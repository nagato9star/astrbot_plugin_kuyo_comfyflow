"""配方与生成历史。

新版配方只引用模型家族并保存底模、LoRA、画幅与采样参数。工作流及节点槽位
由模型家族和工作流档案管理；旧配方的 workflow/slots 字段继续兼容读取。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger

from ..workflows.slots import (
    ANIMA_DROP_NODES,
    detect_slots,
    looks_like_anima,
    merge_slots,
    parse_node_option,
    read_current_values,
    slots_from_config,
)

_SLUG_RE = re.compile(r"[^\w\u4e00-\u9fff-]+", re.UNICODE)
RECIPE_VALUE_KEYS = (
    "prompt",
    "model",
    "loras",
    "width",
    "height",
    "steps",
    "cfg",
    "sampler_name",
    "scheduler",
    "denoise",
    "trigger_words",
    "artist",
    "quality",
    "negative",
)


def slugify(name: str) -> str:
    text = _SLUG_RE.sub("-", str(name or "").strip()).strip("-")
    return text[:64] or "recipe"


def recipe_template(recipe: dict) -> str:
    """读取旧配方绑定的模板名；新配方通过 family 在运行时解析工作流。"""
    r = recipe or {}
    return str(r.get("template") or r.get("workflow") or "").strip()


def _short_file(name: str) -> str:
    text = str(name or "").replace("\\", "/").split("/")[-1]
    return text[:-12] if text.endswith(".safetensors") else text


# 模型家族启发式：从底模文件名猜结构系（CLIP/引导/分辨率档是否通用）。
# 顺序即优先级——先匹配具体家族关键词，再落宽泛的 sdxl/sd15。
_FAMILY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("krea", "krea"),
    ("qwen", "qwen"),
    ("flux", "flux"),
    ("chroma", "chroma"),
    ("anima", "anima"),
    ("noobai", "illustrious"),
    ("illustrious", "illustrious"),
    ("pony", "pony"),
    ("hunyuan", "hunyuan"),
    ("wan", "wan"),
    ("ltxv", "ltxv"),
    ("hidream", "hidream"),
    ("sd3", "sd3"),
    ("sd_xl", "sdxl"),
    ("sdxl", "sdxl"),
    ("v1-5", "sd15"),
    ("sd15", "sd15"),
    ("sd1", "sd15"),
)


def model_family(model_name: str) -> str:
    """从底模文件名猜模型家族；猜不出返回空串（空串 = 不做家族校验）。"""
    text = str(model_name or "").replace("\\", "/").rsplit("/", 1)[-1].casefold()
    if not text:
        return ""
    for keyword, family in _FAMILY_KEYWORDS:
        if keyword in text:
            return family
    return ""


def recipe_family(recipe: dict) -> str:
    """配方的模型家族：显式 family 字段优先，否则从默认底模名猜。

    兼容完整配方（defaults.model）与 list() 摘要行（顶层 model）两种形态。
    """
    recipe = recipe or {}
    explicit = str(recipe.get("family") or "").strip().casefold()
    if explicit:
        return explicit
    model = ((recipe.get("defaults") or {}).get("model")) or recipe.get("model") or ""
    return model_family(model)


class RecipeStore:
    def __init__(self, data_dir: Path, preferred_default: str = "") -> None:
        self.data_dir = data_dir
        self.preferred_default = str(preferred_default or "").strip()
        self.recipe_dir = data_dir / "recipes"
        self.history_dir = data_dir / "history"
        self.recipe_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._warned_missing: set[str] = set()

    def path_for(self, recipe_id: str) -> Path:
        rid = slugify(recipe_id)
        return self.recipe_dir / f"{rid}.json"

    def list(self) -> list[dict]:
        rows = []
        for path in sorted(self.recipe_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            rows.append(self._public(data, path.stem))
        return rows

    def names(self) -> list[str]:
        return [r.get("name") or r["id"] for r in self.list()]

    def catalog(self, limit: int = 8) -> str:
        """给 LLM 看的短目录：配方名 + 底模 + LoRA + 尺寸。"""
        parts = []
        for row in self.list()[:limit]:
            bits = [row.get("name") or row["id"]]
            if row.get("family"):
                bits.append(f"family={row['family']}")
            if row.get("description"):
                bits.append(str(row["description"])[:24])
            if row.get("model"):
                bits.append(_short_file(row["model"]))
            loras = row.get("loras") or []
            if loras:
                bits.append("lora=" + "+".join(_short_file(x) for x in loras[:3]))
            if row.get("width") and row.get("height"):
                bits.append(f"{row['width']}x{row['height']}")
            parts.append(" ".join(str(b) for b in bits if b))
        return "；".join(parts)

    def get(self, name_or_id: str) -> dict | None:
        if not name_or_id:
            return None
        wanted = str(name_or_id).strip()
        direct = self.path_for(wanted)
        if direct.is_file():
            return self._load(direct)
        for row in self.list():
            if row.get("name") == wanted or row.get("id") == wanted:
                return self._load(self.path_for(row["id"]))
        return None

    def default(self, preferred: str = "") -> dict | None:
        preferred = str(preferred or self.preferred_default or "").strip()
        if preferred:
            found = self.get(preferred)
            if found:
                return found
            if self.list() and preferred not in self._warned_missing:
                self._warned_missing.add(preferred)
                logger.warning(
                    f"[ComfyUIDirect] 配置的默认配方「{preferred}」不存在，回退到其它配方"
                )
        rows = self.list()
        for row in rows:
            if row.get("id") == "default" or row.get("name") in ("默认", "default"):
                return self.get(row["id"])
        if rows:
            return self.get(rows[0]["id"])
        return None

    def base_slots_for(self, workflow_name: str) -> tuple[dict, str]:
        """绑定工作流的基底槽位映射：默认配方优先，其次同工作流且已映射主提示词的配方。

        配方保存时用它继承节点映射，保证新配方不缺主提示词槽位、可直接用于生成。
        返回 (slots, source_name)，无可继承时返回 ({}, "")。
        """
        wanted = str(workflow_name or "").strip()
        if not wanted:
            return {}, ""
        candidates: list[dict] = []
        default_recipe = self.default()
        if default_recipe and recipe_template(default_recipe) == wanted:
            candidates.append(default_recipe)
        for row in self.list():
            if default_recipe and row.get("id") == default_recipe.get("id"):
                continue
            full = self.get(str(row.get("id") or ""))
            if full and recipe_template(full) == wanted:
                candidates.append(full)
        for cand in candidates:
            slots = cand.get("slots") or {}
            if slots.get("prompt"):
                return dict(slots), str(cand.get("name") or cand.get("id") or "")
        return {}, ""

    def save(self, recipe: dict) -> dict:
        name = str(recipe.get("name") or "").strip()
        if not name:
            raise ValueError("配方名不能为空")
        rid = str(recipe.get("id") or "").strip() or slugify(name)
        rid = slugify(rid)
        # name 是配方的唯一标识：同名不同 id 的旧文件直接清掉，
        # 否则 get/delete 按 name 匹配会出现歧义（webui 里"删不掉"的根源）。
        for row in self.list():
            if row.get("name") == name and row.get("id") != rid:
                stale = self.path_for(str(row.get("id") or ""))
                if stale.is_file():
                    stale.unlink()
        family = str(recipe.get("family") or "").strip()
        data = {
            "id": rid,
            "name": name,
            "description": str(recipe.get("description") or "").strip(),
            "defaults": _clean_defaults(recipe.get("defaults") or {}),
            "updated_at": int(time.time()),
        }
        if family:
            # 新格式：配方只引用家族。工作流、槽位与节点清理规则归工作流档案所有。
            data["family"] = family
        else:
            # 旧格式兼容：没有家族时仍保留原绑定，便于升级前的数据继续运行。
            template = str(recipe.get("template") or recipe.get("workflow") or "").strip()
            data.update(
                {
                    "template": template,
                    "workflow": template,
                    "slots": recipe.get("slots") or {},
                    "drop_nodes": list(recipe.get("drop_nodes") or []),
                }
            )
        presets = recipe.get("size_presets")
        if isinstance(presets, dict) and presets:
            data["size_presets"] = presets
        path = self.path_for(rid)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data

    def delete(self, name_or_id: str) -> int:
        """按 id 精确删；按 name 匹配时清掉全部同名配方。

        返回删除数量（0 = 没删到）。真值判断与旧的 bool 返回兼容。
        """
        wanted = str(name_or_id or "").strip()
        if not wanted:
            return 0
        removed = 0
        direct = self.path_for(wanted)
        if direct.is_file():
            direct.unlink()
            removed += 1
        for row in self.list():
            if row.get("id") == wanted or row.get("name") == wanted:
                path = self.path_for(str(row.get("id") or ""))
                if path.is_file():
                    path.unlink()
                    removed += 1
        return removed

    def save_history(self, entry: dict) -> Path:
        pid = str(entry.get("prompt_id") or int(time.time()))
        safe = slugify(pid)
        path = self.history_dir / f"{safe}.json"
        payload = dict(entry)
        payload["saved_at"] = int(time.time())
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._trim_history(40)
        return path

    def list_history(self, limit: int = 20) -> list[dict]:
        files = sorted(self.history_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for path in files[:limit]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                out.append(data)
        return out

    def history_get(self, prompt_id: str) -> dict | None:
        path = self.history_dir / f"{slugify(prompt_id)}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def recipe_from_history(self, prompt_id: str, name: str, description: str = "") -> dict:
        entry = self.history_get(prompt_id)
        if entry is None:
            raise ValueError(f"历史不存在: {prompt_id}")
        defaults = dict(entry.get("values") or entry.get("defaults") or {})
        if not entry.get("family") and entry.get("prompt") and "prompt" not in defaults:
            defaults["prompt"] = entry["prompt"]
        recipe = {
            "id": slugify(name),
            "name": name,
            "description": description or entry.get("description") or "",
            "template": entry.get("template") or entry.get("workflow") or "",
            "workflow": entry.get("template") or entry.get("workflow") or "",
            "slots": entry.get("slots") or {},
            "defaults": _clean_defaults(defaults),
            "drop_nodes": list(entry.get("drop_nodes") or []),
        }
        if entry.get("family"):
            recipe["family"] = entry["family"]
        return self.save(recipe)

    def bootstrap(
        self,
        *,
        workflow_name: str,
        wf: dict,
        config_slots: Any = None,
        config_defaults: dict | None = None,
        family: str = "",
    ) -> dict | None:
        """没有配方时，用自动检测 + 配置下拉生成「默认」配方。"""
        if self.list():
            return self.default()
        try:
            detected = detect_slots(wf)
            configured = slots_from_config(config_slots)
            slots = merge_slots(detected, configured)
            if "prompt" not in slots:
                logger.warning("[ComfyUIDirect] 未能检测到主提示词节点，跳过默认配方")
                return None
            for role, spec in slots.items():
                if "node" in spec and not spec.get("field"):
                    spec["node"] = parse_node_option(spec["node"])
            defaults = read_current_values(wf, slots)
            for key, val in (config_defaults or {}).items():
                if val in (None, "", 0, 0.0, [], {}):
                    continue
                defaults[key] = val
            recipe = {
                "id": "default",
                "name": "默认",
                "description": "导入工作流后自动生成的默认配方",
                "template": workflow_name,
                "workflow": workflow_name,
                "slots": slots,
                "defaults": _clean_defaults(defaults),
                "drop_nodes": list(ANIMA_DROP_NODES) if looks_like_anima(wf) else [],
            }
            if str(family or "").strip():
                recipe["family"] = str(family).strip()
            saved = self.save(recipe)
            logger.info("[ComfyUIDirect] 已创建默认配方")
            return saved
        except Exception as e:
            logger.warning(f"[ComfyUIDirect] 创建默认配方失败: {e}")
            return None

    def _load(self, path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"[ComfyUIDirect] 读取配方失败 {path.name}: {e}")
            return None
        if not isinstance(data, dict):
            return None
        data.setdefault("id", path.stem)
        data.setdefault("slots", {})
        data.setdefault("defaults", {})
        return data

    def _public(self, data: dict, stem: str) -> dict:
        defaults = data.get("defaults") or {}
        return {
            "id": data.get("id") or stem,
            "name": data.get("name") or stem,
            "description": data.get("description") or "",
            "template": recipe_template(data),
            "workflow": recipe_template(data),
            "family": data.get("family") or recipe_family(data),
            "model": defaults.get("model") or "",
            "loras": [
                str(x.get("name") or x)
                for x in (defaults.get("loras") or [])
                if x
            ][:4],
            "width": defaults.get("width") or 0,
            "height": defaults.get("height") or 0,
            "has_slots": bool(data.get("slots")),
        }

    def attach_legacy_family(self, name_or_id: str, family: str) -> bool:
        """给旧配方补家族标签，同时保留旧字段作为回退数据。"""
        target = self.get(name_or_id)
        family = str(family or "").strip()
        if not target or target.get("family") or not family:
            return False
        path = self.path_for(str(target.get("id") or name_or_id))
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return False
            raw["family"] = family
            path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        except (OSError, json.JSONDecodeError):
            return False

    def used_templates(self) -> dict[str, list[str]]:
        """模板名 -> 引用它的配方显示名列表。删模板前的引用完整性检查用。"""
        refs: dict[str, list[str]] = {}
        for row in self.list():
            full = self.get(str(row.get("id") or ""))
            if full and full.get("family"):
                continue
            t = str(row.get("template") or "").strip()
            if t:
                refs.setdefault(t, []).append(str(row.get("name") or row.get("id") or ""))
        return refs

    def _trim_history(self, keep: int) -> None:
        files = sorted(self.history_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files[keep:]:
            try:
                path.unlink()
            except OSError:
                pass


def _clean_defaults(raw: dict) -> dict:
    out: dict[str, Any] = {}
    for key in RECIPE_VALUE_KEYS:
        if key not in raw:
            continue
        val = raw[key]
        if val in (None, "") and not (key == "trigger_words" and val == ""):
            continue
        if val == [] and key != "loras":
            continue
        if key in ("width", "height", "steps") and val == 0:
            continue
        if key in ("cfg", "denoise") and val == 0:
            continue
        out[key] = val
    return out


def materialize_values(recipe: dict, overrides: dict[str, Any]) -> dict[str, Any]:
    """优先级：LLM/请求显式值 > 配方 defaults。种子从不复用配方里的旧值。"""
    values = dict(recipe.get("defaults") or {})
    values.pop("seed", None)
    for key, val in overrides.items():
        if val is None or (val == "" and key != "trigger_words"):
            continue
        values[key] = val
    return values
