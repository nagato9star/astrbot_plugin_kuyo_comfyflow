"""Chinese character-name lookup with a small local index and cached references.

Names: Jannchie/danbooru-tag-index's wiki-name selection and reviewed corrections.
Appearance: AnimaDex's public JSON gallery API. These sources are kept separate.
No post/image downloads or query-time LLM translation are used.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
import unicodedata
from pathlib import Path
from typing import Any

import httpx

NAME_ROOT = "https://raw.githubusercontent.com/Jannchie/danbooru-tag-index/master/data/translations/"
NAME_FILES = ("character_official.json", "character_manual.json", "zh_rejected.json")
MAX_BYTES = 4 * 1024 * 1024


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return "_".join(value.replace("\\(", "(").replace("\\)", ")").split())


def _tags(value: Any) -> list[str]:
    values = (
        value
        if isinstance(value, list)
        else str(value or "").replace("，", ",").split(",")
    )
    return list(
        dict.fromkeys(
            str(v).strip() for v in values if isinstance(v, str) and v.strip()
        )
    )[:32]


def name_entries(official: dict, manual: dict, rejected: dict) -> list[dict]:
    """Reject known bad names while keeping the canonical tag available for search."""
    entries = []
    for tag, value in official.items():
        if not isinstance(tag, str) or not isinstance(value, dict):
            continue
        labels = [
            value.get(language) for language in ("zh_hans", "zh_hant", "ja", "ko")
        ]
        bad = normalize_name(rejected.get(tag, ""))
        names = [
            s
            for s in labels
            if isinstance(s, str) and s.strip() and normalize_name(s) != bad
        ]
        entries.append(
            {"tag": tag, "names": names, "source": "中文名字索引（Wiki 别名选取）"}
        )
    by_tag = {entry["tag"]: entry for entry in entries}
    for tag, label in manual.items():
        if not isinstance(tag, str) or not isinstance(label, str) or not label.strip():
            continue
        entry = by_tag.setdefault(tag, {"tag": tag, "names": []})
        # A correction replaces previous Chinese names, preserving other languages.
        entry["names"] = [
            name
            for name in entry["names"]
            if not any("\u4e00" <= c <= "\u9fff" for c in name)
        ]
        entry["names"].insert(0, label)
        entry["source"] = "中文名字索引（人工校订）"
    return list(by_tag.values())


class CharacterTagLookup:
    def __init__(
        self,
        data_dir: Path,
        *,
        danbooru=None,
        enabled: bool = True,
        names_file: str = "",
        aliases: list[dict] | None = None,
        reference_url: str = "https://animadex.net",
        timeout: float = 6.0,
        cache_hours: float = 168.0,
    ) -> None:
        self.cache_path = Path(data_dir) / "character_tags.json"
        self.danbooru = danbooru
        self.enabled = enabled
        self.names_file = Path(names_file).expanduser() if names_file else None
        self.aliases = aliases or []
        self.reference_url = reference_url.rstrip("/")
        self.timeout = max(1.0, min(float(timeout), 30.0))
        self.ttl = max(60.0, float(cache_hours) * 3600)
        self.entries: list[dict] = []
        self.base_entries: list[dict] = []
        self.exact: dict[str, list[dict]] = {}
        self.overrides: dict[str, list[dict]] = {}
        self.details: dict[str, dict] = {}
        self.loaded = False
        self.loaded_at = 0.0
        self.file_stamp = None
        self._names_task: asyncio.Task | None = None
        self._detail_tasks: dict[str, asyncio.Task] = {}
        self._misses: dict[str, float] = {}
        self._remote_names: dict[str, tuple[float, list[dict]]] = {}
        self._retry_at = 0.0
        self._write_lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None
        self._index([])

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": "AstrBot-ComfyUIDirect/character-lookup"},
            )
        return self._client

    async def _json(self, url: str, params: dict | None = None) -> Any:
        async with self.client.stream("GET", url, params=params) as response:
            response.raise_for_status()
            chunks = bytearray()
            async for chunk in response.aiter_bytes():
                chunks.extend(chunk)
                if len(chunks) > MAX_BYTES:
                    raise ValueError("角色数据响应过大")
            return json.loads(chunks)

    def _index(self, entries: list[dict]) -> None:
        self.base_entries = [
            copy.deepcopy(entry)
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("tag"), str)
            and isinstance(entry.get("names", []), list)
        ]
        merged = {entry["tag"]: copy.deepcopy(entry) for entry in self.base_entries}
        override_tags: dict[str, set[str]] = {}
        for row in self.aliases:
            if not isinstance(row, dict) or not row.get("name") or not row.get("tag"):
                continue
            tag = normalize_name(row["tag"])
            entry = merged.setdefault(tag, {"tag": tag, "names": []})
            entry["names"] = list(
                dict.fromkeys([str(row["name"]).strip(), *entry.get("names", [])])
            )
            entry["source"] = "配置别名"
            override_tags.setdefault(normalize_name(row["name"]), set()).add(tag)
            if row.get("appearance_tags"):
                entry["appearance_tags"] = _tags(row["appearance_tags"])
        self.entries = list(merged.values())
        self.overrides = {
            name: [merged[tag] for tag in tags] for name, tags in override_tags.items()
        }
        self.exact = {}
        for entry in self.entries:
            entry.setdefault("source", "本地角色名字库")
            names = {
                normalize_name(name)
                for name in [entry["tag"], *entry.get("names", [])]
                if isinstance(name, str)
            }
            for name in names:
                self.exact.setdefault(name, []).append(entry)

    async def _persist(self) -> None:
        async with self._write_lock:
            payload = {
                "version": 1,
                "loaded_at": self.loaded_at,
                "entries": self.base_entries,
                "details": self.details,
            }
            text = json.dumps(payload, ensure_ascii=False)

            def write():
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.cache_path.with_suffix(".tmp")
                temporary.write_text(text, encoding="utf-8")
                os.replace(temporary, self.cache_path)

            try:
                await asyncio.to_thread(write)
            except OSError:
                pass  # The in-memory lookup remains usable on a read-only filesystem.

    async def _load_names(self, allow_download: bool = True) -> None:
        if not self.loaded:
            try:
                cached = await asyncio.to_thread(
                    lambda: json.loads(self.cache_path.read_text(encoding="utf-8"))
                )
                if cached.get("version") == 1 and isinstance(
                    cached.get("entries"), list
                ):
                    await asyncio.to_thread(self._index, cached["entries"])
                    self.details = (
                        cached.get("details", {})
                        if isinstance(cached.get("details"), dict)
                        else {}
                    )
                    self.loaded_at = float(cached.get("loaded_at", 0))
            except (OSError, ValueError, TypeError, AttributeError):
                pass
            self.loaded = True
            if not self.entries:
                self._index([])
        if self.names_file:
            try:
                stamp = self.names_file.stat().st_mtime_ns
                if stamp != self.file_stamp:
                    data = await asyncio.to_thread(
                        lambda: json.loads(self.names_file.read_text(encoding="utf-8"))
                    )
                    if isinstance(data, list):
                        entries = data
                    elif isinstance(data, dict):
                        entries = name_entries(data, {}, {})
                    else:
                        return
                    await asyncio.to_thread(self._index, entries)
                    self.file_stamp = stamp
            except (OSError, ValueError, TypeError):
                pass
            return
        if (
            not allow_download
            or not self.enabled
            or time.time() - self.loaded_at < self.ttl
            or time.monotonic() < self._retry_at
        ):
            return
        self._retry_at = time.monotonic() + 60
        try:
            async with asyncio.timeout(self.timeout):
                data = await asyncio.gather(
                    *(self._json(NAME_ROOT + name) for name in NAME_FILES),
                    return_exceptions=True,
                )
            if not all(isinstance(item, dict) for item in data):
                return
            entries = await asyncio.to_thread(name_entries, *data)
            if not entries:
                return
            await asyncio.to_thread(self._index, entries)
            self.loaded_at = time.time()
            await self._persist()
        except (httpx.HTTPError, ValueError, TimeoutError):
            pass  # Keep the previous index if a refresh fails.

    async def _ensure_names(self) -> None:
        if self._names_task is None or self._names_task.done():
            self._names_task = asyncio.create_task(self._load_names())
        # A cancelled caller must not cancel the refresh shared by other lookups.
        await asyncio.shield(self._names_task)

    async def _reference(self, tag: str) -> dict | None:
        try:
            async with asyncio.timeout(self.timeout):
                data = await self._json(
                    self.reference_url + "/api/characters/search", {"q": tag, "page": 1}
                )
            if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                return None
            rows = [
                row
                for row in data["results"]
                if isinstance(row, dict)
                and normalize_name(row.get("slug", "")) == normalize_name(tag)
            ]
            if len(rows) != 1:
                return None
            row = rows[0]
            detail = {
                "saved_at": time.time(),
                "reference_url": self.reference_url,
                "appearance_tags": _tags(row.get("tags")),
                "trigger": str(row.get("trigger") or ""),
                "copyright": str(row.get("copyright") or ""),
            }
            self.details[tag] = detail
            # Bound the disk cache independently of the much larger name index.
            if len(self.details) > 512:
                oldest = min(
                    self.details, key=lambda key: self.details[key].get("saved_at", 0)
                )
                self.details.pop(oldest, None)
            await self._persist()
            return detail
        except (httpx.HTTPError, httpx.InvalidURL, ValueError, TypeError, TimeoutError):
            return None
        finally:
            self._misses[tag] = time.monotonic()

    async def _get_reference(self, tag: str) -> dict | None:
        cached = self.details.get(tag)
        if (
            isinstance(cached, dict)
            and cached.get("reference_url") == self.reference_url
            and time.time() - cached.get("saved_at", 0) < self.ttl
        ):
            return copy.deepcopy(cached)
        if not self.reference_url or time.monotonic() - self._misses.get(tag, -60) < 60:
            return None
        task = self._detail_tasks.get(tag)
        if task is None:
            task = asyncio.create_task(self._reference(tag))
            self._detail_tasks[tag] = task
            task.add_done_callback(lambda done: self._detail_tasks.pop(tag, None))
        return await asyncio.shield(task)

    async def lookup(self, query: str, limit: int = 5) -> dict:
        query = str(query).strip()[:160]
        key = normalize_name(query)
        if not key:
            return {"query": query, "matches": []}
        limit = max(1, min(int(limit or 5), 8))
        if key not in self.overrides:
            await self._ensure_names()
        elif not self.loaded:
            # Read a previous cache before fetching a configured alias's reference.
            # This avoids replacing an existing name index with an empty index.
            await self._load_names(allow_download=False)
        matches = self.overrides.get(key) or self.exact.get(key, [])
        exact = bool(matches)
        if not matches:
            matches = [
                entry
                for entry in self.entries
                if any(
                    key in normalize_name(name)
                    for name in [entry["tag"], *entry.get("names", [])]
                )
            ]
        if not matches and self.danbooru is not None:
            cached = self._remote_names.get(key)
            if cached and time.monotonic() - cached[0] < 60:
                matches = copy.deepcopy(cached[1])
            else:
                try:
                    async with asyncio.timeout(self.timeout):
                        # Keep enough candidates to detect ambiguity even with limit=1.
                        matches = await self.danbooru.character_candidates(query, 8)
                except (httpx.HTTPError, ValueError, TimeoutError):
                    matches = []
                self._remote_names[key] = (time.monotonic(), copy.deepcopy(matches))
                if len(self._remote_names) > 512:
                    self._remote_names.pop(next(iter(self._remote_names)))
            if matches:
                # Only exact aliases are learned; partial searches stay candidates.
                exact_matches = [
                    entry
                    for entry in matches
                    if any(
                        key == normalize_name(name)
                        for name in [entry["tag"], *entry.get("names", [])]
                    )
                ]
                if exact_matches:
                    matches, exact = exact_matches, True
                    merged = {
                        entry["tag"]: copy.deepcopy(entry)
                        for entry in self.base_entries
                    }
                    for entry in matches:
                        previous = merged.get(entry["tag"], {})
                        merged[entry["tag"]] = {
                            **entry,
                            "names": list(
                                dict.fromkeys(
                                    [
                                        *previous.get("names", []),
                                        *entry.get("names", []),
                                    ]
                                )
                            ),
                        }
                    self._index(list(merged.values()))
                    await self._persist()
        matches = sorted(matches, key=lambda entry: (len(entry["tag"]), entry["tag"]))
        total = len(matches)
        selected = copy.deepcopy(matches[:limit])
        if total == 1 and exact:
            entry = selected[0]
            if not entry.get("appearance_tags"):
                detail = await self._get_reference(entry["tag"])
                if detail:
                    entry.update(detail)
                    entry["appearance_source"] = "AnimaDex 角色样例"
            else:
                entry["appearance_source"] = "配置／本地角色库"
        return {"query": query, "matches": selected, "total": total, "exact": exact}

    async def close(self) -> None:
        tasks = [
            task
            for task in [self._names_task, *self._detail_tasks.values()]
            if task and not task.done()
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()


def format_character_result(result: dict) -> str:
    matches = result.get("matches") or []
    if not matches:
        return f"未找到角色：{result['query']}。可换日文／英文名，或在 character_aliases 中添加已核实的映射。"
    if result.get("total", len(matches)) > 1 or not result.get("exact"):
        lines = [
            f"【角色候选 {result['query']}】共 {result.get('total', len(matches))} 项，显示 {len(matches)} 项；请结合所属作品核对 tag："
        ]
        lines.extend(
            f"- {entry['tag']}（{' / '.join(entry.get('names', [])[:2]) or entry['tag']}；{entry.get('source', 'Danbooru')}）"
            for entry in matches
        )
        return "\n".join(lines)
    entry = matches[0]
    lines = [
        f"【角色 {result['query']}】",
        f"角色 tag: {entry['tag']}",
        f"名称来源: {entry.get('source', 'Danbooru')}",
    ]
    if entry.get("copyright"):
        lines.append(f"作品 tag: {entry['copyright']}")
    if entry.get("appearance_tags"):
        colors = {
            "aqua",
            "black",
            "blue",
            "brown",
            "green",
            "grey",
            "gray",
            "orange",
            "pink",
            "purple",
            "red",
            "silver",
            "white",
            "yellow",
            "golden",
            "blonde",
        }
        alternatives = {}
        normalized = {normalize_name(tag) for tag in entry["appearance_tags"]}
        for part in ("eyes", "hair"):
            if part == "eyes" and "heterochromia" in normalized:
                continue
            if part == "hair" and normalized.intersection(
                {
                    "multicolored_hair",
                    "two-tone_hair",
                    "two_tone_hair",
                    "gradient_hair",
                    "split-color_hair",
                    "split_color_hair",
                    "streaked_hair",
                    "colored_inner_hair",
                }
            ):
                continue
            group = [
                tag
                for tag in entry["appearance_tags"]
                if normalize_name(tag) in {f"{color}_{part}" for color in colors}
            ]
            if len(group) > 1:
                alternatives[part] = group
        choices = {tag for group in alternatives.values() for tag in group}
        shared = [tag for tag in entry["appearance_tags"] if tag not in choices]
        lines.append(
            f"外观参考 tag（{entry.get('appearance_source', '本地角色库')}）: "
            + ", ".join(shared[:20])
        )
        for part, group in alternatives.items():
            label = "眼色" if part == "eyes" else "发色"
            lines.append(
                f"{label}候选: " + " / ".join(group) + "（按角色版本选用，勿同时拼入）"
            )
        lines.append("样例标签可能含默认服饰；按本次绘图需求取用。")
    else:
        lines.append("外观标签暂无可用来源；角色 tag 可单独使用。")
    return "\n".join(lines)
