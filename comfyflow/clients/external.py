"""外部资源查询客户端：Danbooru / Gelbooru（画师/角色触发词）+ Civitai 镜像（生成配方）。

2026-08-11 v2.0.0 新增：
- Danbooru 公开 API（需自定义 UA），支持画师/角色触发词与高频 tag 聚合
- Gelbooru DAPI（json=1），artist:/character: 命名空间搜索，支持别名查询
- civitai.red 镜像兼容 Civitai 官方 v1 API（/api/v1/images 的 meta 即完整生成配方），
  可选 API key（Authorization: Bearer）
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import httpx

from astrbot.api import logger

USER_AGENT = "AstrBot-KuyoComfyFlow/2.7 (+astrbot_plugin_kuyo_comfyflow)"

# 聚合 tag 时排除的通用质量/评分词
SKIP_TAGS = {
    "masterpiece", "best quality", "highres", "absurdres", "newest", "sensitive",
    "very aesthetic", "ultra detailed", "high contrast", "simple background",
    "year2025", "year2026",
}
SKIP_PREFIXES = ("artist:", "copyright:", "character:", "meta:", "rating:", "score_")


class DanbooruClient:
    """Danbooru 只读查询客户端（多镜像顺序回退）。

    与 prompt_optimizer 的角色 tag 查询共用同一套站点策略：donmai 镜像逐个尝试，
    失败快速跳过，全部失败时由调用方回退 gelbooru / safebooru.org，避免单点 403
    就让整个画师/角色查询失效。
    """

    def __init__(
        self,
        base_url: str = "https://danbooru.donmai.us",
        timeout: float = 15.0,
        base_urls: tuple[str, ...] = (),
    ) -> None:
        self.timeout = timeout
        urls = tuple(u.rstrip("/") for u in base_urls if u and str(u).strip())
        if not urls:
            urls = (base_url.rstrip("/"),)
        self.base_urls: tuple[str, ...] = urls
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_urls[0],
                timeout=httpx.Timeout(self.timeout),
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,  # 镜像常做 http->https / 主机跳转
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, params: dict) -> Any:
        last_err: str = ""
        for base_url in self.base_urls:
            try:
                resp = await self.client.get(base_url + path, params=params)
                resp.raise_for_status()
                try:
                    return resp.json()
                except ValueError as e:
                    # 镜像失效时可能返回 HTML 提示页而非 JSON
                    last_err = f"{base_url}{path} 响应非 JSON: {e}"
                    logger.warning(
                        f"[ComfyUIDirect] danbooru {base_url}{path} 响应非 JSON"
                        f"（镜像可能失效），尝试下一镜像: {e}"
                    )
                    continue
            except httpx.HTTPError as e:
                last_err = f"{base_url}{path} HTTP 错误: {e}"
                logger.debug(f"[ComfyUIDirect] danbooru GET {base_url}{path} 失败，尝试下一镜像: {e}")
                continue
        logger.error(f"[ComfyUIDirect] danbooru 全部镜像查询失败: {last_err}")
        return None

    async def _find_artist(self, query: str) -> dict | None:
        artists = await self._get(
            "/artists.json", {"search[name_matches]": f"*{query}*", "limit": 5}
        )
        if not artists:
            return None
        q = query.lower()
        for a in artists:
            names = [str(a.get("name", ""))] + [str(n) for n in (a.get("other_names") or [])]
            if any(q in n.lower() for n in names):
                return a
        return artists[0]

    async def _find_character(self, query: str) -> dict | None:
        tags = await self._get(
            "/tags.json",
            {"search[name_matches]": f"*{query}*", "search[category]": 4, "limit": 5},
        )
        if not tags:
            return None
        q = query.lower()
        for t in tags:
            names = [str(t.get("name", ""))] + [str(a) for a in (t.get("aliases") or [])]
            if any(q in n.lower() for n in names):
                return t
        return tags[0]

    async def search_artist(self, query: str, limit: int = 30) -> dict | None:
        """查画师：返回 {artist, aliases, posts, url}；未找到返回 None。"""
        artist = await self._find_artist(query)
        if artist is None:
            return None
        posts = await self._get(
            "/posts.json", {"tags": f"artist:{artist['name']}", "limit": limit}
        ) if limit > 0 else []
        return {
            "artist": artist["name"],
            "aliases": artist.get("other_names") or [],
            "posts": posts or [],
            "url": f"{self.base_urls[0]}/artists/{artist.get('id', '')}",
        }

    async def search_character(self, query: str, limit: int = 30) -> dict | None:
        """查角色：返回 {character, aliases, posts, url}；未找到返回 None。"""
        tag = await self._find_character(query)
        if tag is None:
            return None
        posts = await self._get(
            "/posts.json", {"tags": f"character:{tag['name']}", "limit": limit}
        ) if limit > 0 else []
        return {
            "character": tag["name"],
            "aliases": tag.get("aliases") or [],
            "posts": posts or [],
            "url": f"{self.base_urls[0]}/tags/{tag.get('id', '')}",
        }

    async def character_candidates(self, query: str, limit: int = 5) -> list[dict]:
        """Lookup canonical character tags/real wiki aliases without post sampling."""
        q = "_".join(query.strip().split()).replace("*", "")
        data = await self._get("/tags.json", {
            "search[name_matches]": f"*{q}*", "search[category]": 4,
            "search[order]": "count", "limit": limit,
        })
        if isinstance(data, list):
            candidates = [{"tag": row["name"], "names": [], "source": "Danbooru 角色 tag"}
                          for row in data if isinstance(row, dict) and row.get("name") and str(row.get("category")) == "4"]
            if candidates:
                return candidates
        # Danbooru puts Japanese/Chinese names on wiki pages, not on Tag.aliases.
        data = await self._get("/wiki_pages.json", {
            "search[other_names_match]": q, "search[tag][category]": 4,
            "limit": limit, "only": "title,other_names",
        })
        if not isinstance(data, list):
            return []
        return [{"tag": row["title"], "names": row.get("other_names") or [], "source": "Danbooru Wiki 别名"}
                for row in data if isinstance(row, dict) and row.get("title")]

    @staticmethod
    def aggregate_tags(posts: list[dict], top: int = 15) -> list[tuple[str, int]]:
        """统计作品 tag 出现频率（去掉画师/角色/版权/质量词）。"""
        cnt: Counter[str] = Counter()
        for p in posts:
            for t in (p.get("tag_string") or "").split():
                if not t or t in SKIP_TAGS:
                    continue
                if any(t.startswith(prefix) for prefix in SKIP_PREFIXES):
                    continue
                cnt[t] += 1
        return cnt.most_common(top)


class GelbooruClient:
    """Gelbooru DAPI 只读查询客户端。

    无独立艺术家表：通过 artist:/character: 命名空间搜索作品，
    从作品 tag 中提取规范名与别名（s=artist 接口尽力而为）。
    """

    def __init__(self, base_url: str = "https://gelbooru.com", timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,  # 镜像常做 http->https / 主机跳转
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, params: dict) -> Any:
        params = dict(params)
        params.setdefault("page", "dapi")
        params.setdefault("json", 1)
        try:
            resp = await self.client.get("/index.php", params=params)
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError as e:
                logger.warning(
                    f"[ComfyUIDirect] gelbooru 响应非 JSON（镜像可能失效），"
                    f"请检查 gelbooru_base_url 配置: {e}"
                )
                return None
        except httpx.HTTPError as e:
            logger.error(f"[ComfyUIDirect] gelbooru 请求失败: {e}")
            return None

    async def _posts(self, tags: str, limit: int) -> list[dict]:
        data = await self._get({"s": "post", "q": "index", "tags": tags, "limit": limit})
        if not data:
            return []
        posts = data.get("post") or []
        if isinstance(posts, dict):
            posts = [posts]  # 单条结果时返回 dict
        return posts

    async def _artist_names(self, posts: list[dict]) -> list[str]:
        """从作品 tags 里统计出现最多的 artist: 规范名。"""
        cnt: Counter[str] = Counter()
        for p in posts:
            for t in (p.get("tags") or "").split():
                if t.startswith("artist:"):
                    cnt[t[len("artist:"):]] += 1
        return [name for name, _ in cnt.most_common()]

    async def _character_names(self, posts: list[dict]) -> list[str]:
        cnt: Counter[str] = Counter()
        for p in posts:
            for t in (p.get("tags") or "").split():
                if t.startswith("character:"):
                    cnt[t[len("character:"):]] += 1
        return [name for name, _ in cnt.most_common()]

    @staticmethod
    def _norm_query(query: str) -> str:
        return query.strip().replace(" ", "_")

    async def search_artist(self, query: str, limit: int = 30) -> dict | None:
        """查画师：返回 {artist, aliases, posts, url}；未找到返回 None。"""
        q = self._norm_query(query)
        posts = await self._posts(f"artist:{q}", limit)
        if not posts:
            return None
        # 确认规范名（作品里实际用的 artist: 标签）
        names = await self._artist_names(posts)
        name = names[0] if names else q
        aliases = []
        # 尽力取别名（s=artist 接口，匹配 name 或 alias）
        artist_data = await self._get({"s": "artist", "q": "index", "name": q})
        artists = artist_data.get("artist") if artist_data else None
        if isinstance(artists, dict):
            artists = [artists]
        if artists:
            for a in artists:
                an = (a.get("name") or "").strip()
                al = (a.get("alias") or "").strip()
                if an.lower() == q.lower() or al.lower() == q.lower():
                    if an and an != name:
                        name = an
                    if al:
                        aliases = [al]
                    break
        return {
            "artist": name,
            "aliases": aliases,
            "posts": posts,
            "url": f"{self.base_url}/index.php?page=post&s=list&tags=artist:{name}",
        }

    async def search_character(self, query: str, limit: int = 30) -> dict | None:
        """查角色：返回 {character, aliases, posts, url}；未找到返回 None。"""
        q = self._norm_query(query)
        posts = await self._posts(f"character:{q}", limit)
        if not posts:
            return None
        names = await self._character_names(posts)
        name = names[0] if names else q
        return {
            "character": name,
            "aliases": [],
            "posts": posts,
            "url": f"{self.base_url}/index.php?page=post&s=list&tags=character:{name}",
        }

    @staticmethod
    def normalize_posts(posts: list[dict]) -> list[dict]:
        """把 gelbooru post 转成聚合器可用的 {tag_string} 结构。"""
        return [{"tag_string": p.get("tags") or ""} for p in posts]


class CivitaiClient:
    """civitai.red（Civitai 官方 API 兼容镜像）查询客户端。"""

    def __init__(
        self,
        base_url: str = "https://civitai.red",
        api_key: str = "",
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"User-Agent": USER_AGENT}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers=headers,
                follow_redirects=True,  # 镜像常做 http->https / 主机跳转
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def search_images(
        self, query: str, limit: int = 5, nsfw: bool = False
    ) -> list[dict]:
        """搜索图片（按最多反应排序），返回含 meta（生成配方）的条目列表。"""
        params = {
            "search": query,
            "limit": limit,
            "sort": "Most Reactions",
            "nsfw": "true" if nsfw else "false",
        }
        try:
            resp = await self.client.get("/api/v1/images", params=params)
            resp.raise_for_status()
            try:
                data = resp.json()
                return (data or {}).get("items") or []
            except ValueError as e:
                logger.warning(f"[ComfyUIDirect] civitai 响应非 JSON（镜像可能失效）: {e}")
                return []
        except httpx.HTTPError as e:
            logger.error(f"[ComfyUIDirect] civitai 搜索失败: {e}")
            return []

    async def search_models(
        self, query: str, types: str = "LORA", limit: int = 3
    ) -> list[dict]:
        """按名称搜模型，返回含 modelVersions（trainedWords 触发词）的条目。"""
        params = {"query": query, "types": types, "limit": limit}
        try:
            resp = await self.client.get("/api/v1/models", params=params)
            resp.raise_for_status()
            try:
                data = resp.json()
                return (data or {}).get("items") or []
            except ValueError as e:
                logger.warning(f"[ComfyUIDirect] civitai 响应非 JSON（镜像可能失效）: {e}")
                return []
        except httpx.HTTPError as e:
            logger.error(f"[ComfyUIDirect] civitai 模型搜索失败: {e}")
            return []
