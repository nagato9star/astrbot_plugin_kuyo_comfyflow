"""模型资源、角色与画师查询工具。"""

from __future__ import annotations

import json
from typing import Any
from astrbot.api import FunctionTool
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from pydantic import ConfigDict
from pydantic import Field
from pydantic.dataclasses import dataclass
from ..resources.characters import CharacterTagLookup
from ..resources.characters import format_character_result
from ..resources.sources import lookup_animadex
from ..resources.sources import lookup_booru
from ..clients.animadex import AnimaDexClient
from ..clients.comfyui import ComfyUIClient
from ..clients.external import CivitaiClient
from ..clients.external import DanbooruClient
from ..clients.external import GelbooruClient
from ..resources.catalog import canonical_family
from ..resources.catalog import exact_matches
from ..resources.catalog import explicitly_named
from ..resources.catalog import family_summary
from ..resources.catalog import filename_matches
from ..resources.catalog import filter_family
from ..resources.catalog import metadata_for
from ..resources.catalog import page_number
from ..resources.catalog import resource_family
from ..resources.catalog import selection_error
from .common import _RESOURCE_QUERY_PROPERTIES, _as_bool, _match_resource
from .recipe_lookup import lookup_recipes
from ..storage.recipes import RecipeStore


def _usage_tips_text(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return value.strip()[:180]
    if not isinstance(value, dict):
        return ""
    labels = (
        ("strength", "建议权重"),
        ("strength_range", "权重范围"),
        ("clip_strength", "CLIP权重"),
        ("clip_skip", "CLIP跳过层"),
    )
    return ", ".join(
        f"{label}={value[key]}" for key, label in labels if value.get(key) not in (None, "")
    )


def _lora_info_summary(info: dict, detailed: bool = False) -> list[str]:
    """Render the compact normalized LoRA record for an LLM response."""
    lines: list[str] = []
    categories = info.get("categories") or []
    tags = info.get("tags") or []
    if categories:
        lines.append("类别: " + ", ".join(str(x) for x in categories[:8]))
    if tags and detailed:
        lines.append("标签: " + ", ".join(str(x) for x in tags[:16]))
    if info.get("model_name") and detailed:
        lines.append("Civitai名称: " + str(info["model_name"]))
    if info.get("base_model"):
        lines.append("基础模型: " + str(info["base_model"]))
    if info.get("description"):
        description = str(info["description"])
        if not detailed:
            description = " ".join(description.split())
            description = description[:120] + ("…" if len(description) > 120 else "")
        lines.append("用途说明: " + description)
    tips = _usage_tips_text(info.get("usage_tips"))
    if tips:
        lines.append("使用建议: " + tips)
    if info.get("trigger_words"):
        lines.append("触发词: " + ", ".join(str(x) for x in info["trigger_words"][:12]))
    if detailed and info.get("notes"):
        lines.append("备注: " + str(info["notes"]))
    return lines


def _match_lora_resources(
    names: list[str], metadata: dict[str, dict], query: str, limit: int = 8
) -> list[str]:
    """Match LoRAs by filename, LoRA Manager category/tag, or description."""
    q = str(query or "").strip().casefold()
    if not q:
        return []

    category_hits: list[str] = []
    for name in names:
        info = metadata.get(name) or {}
        categories = {str(x).casefold() for x in info.get("categories") or []}
        for category, aliases in ComfyUIClient._LORA_CATEGORY_ALIASES.items():
            if category in categories and any(q == str(alias).casefold() for alias in aliases):
                category_hits.append(name)
                break
    if category_hits:
        return category_hits[:limit]

    q_base = q.replace("\\", "/").rsplit("/", 1)[-1]
    hits: list[str] = []
    for name in names:
        info = metadata.get(name) or {}
        search_text = " ".join(
            str(value)
            for value in (
                name,
                info.get("model_name"),
                info.get("categories"),
                info.get("tags"),
                info.get("description"),
                info.get("notes"),
            )
            if value
        ).casefold()
        if q in search_text or q_base in name.replace("\\", "/").rsplit("/", 1)[-1].casefold():
            hits.append(name)
    return hits[:limit]


def _resource_page(client, resources, kind, query="", family="", limit=5, offset=0,
                   include_unknown=False, detailed=False) -> str:
    all_names = resources.get("lora_name" if kind == "lora" else "unet_name") or []
    names = all_names
    meta = metadata_for(resources, kind)
    rules = getattr(client, "resource_family_rules", [])
    title = "LoRA" if kind == "lora" else "底模"
    if not family and not query:
        return f"【{title}家族摘要】" + family_summary(names, meta, rules, kind) + "。指定 model_family 可浏览文件；填写 query 可跨家族按名称搜索。"
    matches = _resource_matches(client, resources, kind, query, family, include_unknown)
    limit = max(1, page_number(limit, 5, 10))
    offset = page_number(offset)
    shown = matches[offset:offset + limit]
    scope = canonical_family(family) if family else f"名称搜索：{query}"
    lines = [f"【{title} {scope}】共 {len(matches)} 项，显示 {offset + 1 if shown else 0}-{offset + len(shown)}"]
    for name, mode, _ in shown:
        found, source = resource_family(name, meta.get(name), rules, kind)
        compatibility = selection_error(
            resources,
            {"loras": [{"name": name}]} if kind == "lora" else {"model": name},
            family,
            rules,
        ) if family else ""
        warning = f"；兼容性待核对: {compatibility}" if compatibility else ""
        near_note = "；近似名称，请核对" if mode == "near" else ""
        lines.append(f"{name} [家族={found or 'unknown'}；{source}{warning}{near_note}]")
        if kind == "lora":
            lines.extend("  " + line[:240] for line in _lora_info_summary(meta.get(name) or {}, detailed=detailed))
    if offset + limit < len(matches):
        lines.append(f"next_offset={offset + limit}；可进一步用 query 缩小范围。")
    if not shown:
        lines.append("无匹配项；检查家族/关键词，或用 model_family=unknown 查看未识别资源并配置归类规则。")
        if kind == "lora" and not family:
            lines.append("按 LoRA 用途、类别或标签查询时，请同时填写生图工作流的 model_family。")
    if include_unknown or canonical_family(family) == "unknown" or any(
        not resource_family(name, meta.get(name), rules, kind)[0] for name, _, _ in shown
    ):
        lines.append("unknown 项未确认兼容性，请核实元数据或配置归类后选用。")
    return "\n".join(lines)


def _resource_matches(client, resources, kind, query, family="", include_unknown=False):
    names = resources.get("lora_name" if kind == "lora" else "unet_name") or []
    meta = metadata_for(resources, kind)
    rules = getattr(client, "resource_family_rules", [])
    exact = exact_matches(names, query) if query else []
    if exact:
        return [(name, "exact", 1.0) for name in exact]
    if family:
        names = filter_family(names, meta, family, rules, kind, include_unknown)
    else:
        names = sorted(set(names), key=str.casefold)
        if not include_unknown:
            names = [name for name in names if resource_family(name, meta.get(name), rules, kind)[0]
                     or (query and explicitly_named(name, query))]
    if not query:
        return [(name, "list", 1.0) for name in names]
    if kind == "lora" and family:
        semantic = _match_lora_resources(names, meta, query, limit=len(names))
        if semantic:
            return [(name, "keyword", 1.0) for name in semantic]
    return filename_matches(names, query)


def _all_resource_search(client, resources, query, family="", limit=5, offset=0,
                         include_unknown=False, detailed=False) -> str:
    rows = []
    for kind, title in (("model", "底模"), ("lora", "LoRA")):
        for name, mode, score in _resource_matches(client, resources, kind, query, family, include_unknown):
            meta = metadata_for(resources, kind)
            found, source = resource_family(name, meta.get(name),
                                            getattr(client, "resource_family_rules", []), kind)
            rows.append((title, name, found or "unknown", source, mode, score))
    if not family:
        for field, title in (("clip_name", "CLIP"), ("vae_name", "VAE"),
                             ("embeddings", "Embedding")):
            rows.extend((title, name, "", "", mode, score)
                        for name, mode, score in filename_matches(resources.get(field) or [], query))
    priority = {"exact": 0, "name": 1, "keyword": 2, "near": 3}
    rows.sort(key=lambda row: (priority[row[4]], -row[5], row[0] != "底模", row[1].casefold()))
    limit = max(1, page_number(limit, 5, 10))
    offset = page_number(offset)
    shown = rows[offset:offset + limit]
    lines = [f"【资源名称搜索：{query}】共 {len(rows)} 项，显示 {offset + 1 if shown else 0}-{offset + len(shown)}"]
    for title, name, found, source, mode, _ in shown:
        family_note = f" [家族={found}；{source}]" if found else ""
        near_note = "（近似名称，请核对）" if mode == "near" else ""
        lines.append(f"{title}: {name}{family_note}{near_note}")
        if title == "LoRA":
            info = (resources.get("lora_meta") or {}).get(name) or {}
            lines.extend("  " + line[:240] for line in _lora_info_summary(info, detailed=detailed))
    if offset + limit < len(rows):
        lines.append(f"next_offset={offset + limit}")
    if not shown:
        lines.append("无匹配项；可换用文件名片段，或显式查询 model_family=unknown / include_unknown=true。")
        if not family:
            lines.append("按 LoRA 用途、类别或标签查询时，请指定 type=lora 和生图工作流的 model_family。")
    if include_unknown or canonical_family(family) == "unknown" or any(
        found == "unknown" for _, _, found, _, _, _ in shown
    ):
        lines.append("unknown 项未确认兼容性，请核实元数据或配置归类后选用。")
    return "\n".join(lines)


async def _local_model_info(client: ComfyUIClient, name: str, kind: str) -> str:
    """本地模型元数据：LoRA Manager sidecar 优先，回退 safetensors 头部。"""
    # LoRA Manager 保存的 sidecar/Civitai 信息比 safetensors 头部更完整，
    # 尤其是 style/character 分类、用途说明和推荐权重。
    manager_meta = await client.get_lora_manager_metadata(name) if kind == "lora" else None
    manager_info = ComfyUIClient.normalize_lora_metadata(manager_meta)
    if manager_info:
        lines = [f"【LoRA Manager】{name}"]
        lines.extend(_lora_info_summary(manager_info, detailed=True))
        return "\n".join(lines)

    meta = await client.get_model_metadata(
        name, folders=["loras"] if kind == "lora" else ["checkpoints", "diffusion_models", "unet"]
    )
    if not meta:
        return f"本地未找到模型 {name}，或该文件没有元数据头部（可用 comfyui_civitai_search 在线搜索）。"

    parts = [f"【本地模型】{name}"]
    title = (
        meta.get("modelspec.title")
        or meta.get("ss_title")
        or meta.get("sd_models/name")
        or ""
    )
    author = meta.get("modelspec.author") or meta.get("ss_creator") or ""
    tags = meta.get("modelspec.tags") or meta.get("ss_tags") or ""
    if title:
        parts.append(f"标题: {title}")
    if author:
        parts.append(f"作者: {author}")
    if tags:
        parts.append(f"标签: {tags}")
    triggers, _source = ComfyUIClient._extract_trigger_words(meta)
    if triggers:
        parts.append(f"触发词: {', '.join(triggers)}")
    elif not title and not author and not tags:
        # 有元数据但都是技术字段：给个头部键名摘要
        keys = [k for k in meta if not k.startswith("ss_")]
        if keys:
            parts.append("元数据键: " + ", ".join(keys[:12]))
    return "\n".join(parts)


# 本插件家族 → Civitai baseModels 过滤值；无映射的家族不缩小结果。
_CIVITAI_BASE_MODELS: dict[str, list[str]] = {
    "moss": ["MOSS"],
    "flux": ["Flux.1 D", "Flux.1 S", "Flux.1 Krea", "Flux.2 D", "Flux.2 Schnell"],
    "krea2": ["Flux.1 Krea"],
    "qwen": ["Qwen"],
    "chroma": ["Chroma"],
    "illustrious": ["Illustrious"],
    "noobai": ["NoobAI"],
    "pony": ["Pony"],
    "sdxl": ["SDXL 1.0", "SDXL 0.9", "SDXL Turbo", "SDXL 1.0 LCM", "SDXL Lightning"],
    "sd15": ["SD 1.5", "SD 1.4", "SD 1.5 LCM"],
    "sd3": ["SD 3", "SD 3.5", "SD 3.5 Medium", "SD 3.5 Large", "SD 3.5 Large Turbo"],
    "wan": ["Wan Video"],
    "hunyuan": ["Hunyuan 1", "Hunyuan Video"],
    "hidream": ["HiDream"],
    "ltxv": ["LTXV"],
}

_CIVITAI_IMAGE_SORTS = {
    "most reactions": "Most Reactions",
    "reactions": "Most Reactions",
    "most comments": "Most Comments",
    "comments": "Most Comments",
    "newest": "Newest",
}
_CIVITAI_MODEL_SORTS = {
    "highest rated": "Highest Rated",
    "rated": "Highest Rated",
    "most downloaded": "Most Downloaded",
    "downloaded": "Most Downloaded",
    "newest": "Newest",
}
_CIVITAI_PERIODS = {
    "day": "Day", "week": "Week", "month": "Month",
    "year": "Year", "alltime": "AllTime", "all": "AllTime",
}


def _civitai_base_models(kwargs: dict) -> list[str] | None:
    """base_model 精确值优先，否则按本插件家族映射为 civitai baseModels 列表。"""
    explicit = str(kwargs.get("base_model") or "").strip()
    if explicit:
        return [explicit]
    family = canonical_family(kwargs.get("model_family"))
    return _CIVITAI_BASE_MODELS.get(family)


def _civitai_norm_sort(raw: Any, allowed: dict, default: str) -> str:
    return allowed.get(str(raw or "").strip().casefold(), default)


def _civitai_norm_period(raw: Any) -> str:
    return _CIVITAI_PERIODS.get(str(raw or "").strip().casefold(), "AllTime")


def _truncate(s: str, n: int) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[:n] + "…"


def _meta_first(meta: dict, *keys: str) -> Any:
    for key in keys:
        val = meta.get(key)
        if val not in (None, ""):
            return val
    return None


def _has_recipe_meta(item: dict) -> bool:
    meta = item.get("meta")
    if not isinstance(meta, dict) or not meta:
        return False
    return any(
        _meta_first(meta, *keys) is not None
        for keys in (
            ("prompt", "Prompt"),
            ("sampler", "Sampler"),
            ("steps", "Steps"),
            ("cfgScale", "CFG scale", "cfg"),
            ("seed", "Seed"),
            ("Model", "model"),
        )
    )


def _civitai_model_label(item: dict, meta: dict) -> str:
    raw = str(_meta_first(meta, "Model", "model") or "").strip()
    is_air = raw.startswith("urn:air:")
    model = "" if is_air else raw
    if not model:
        for res in meta.get("civitaiResources") or meta.get("resources") or []:
            if not isinstance(res, dict):
                continue
            if str(res.get("type") or "").casefold() in {"checkpoint", "model"}:
                name = str(
                    res.get("modelName") or res.get("name") or res.get("modelVersionName") or ""
                ).strip()
                if name:
                    model = name
                    break
    if not model:
        model = str(item.get("modelName") or "").strip() or raw
    base = str(item.get("baseModel") or meta.get("baseModel") or "").strip()
    if not model:
        return base or "未知模型"
    if base and base.casefold() not in model.casefold():
        return f"{model}（{base}）"
    return model


def _civitai_lora_summary(meta: dict) -> str:
    loras: list[str] = []
    for res in meta.get("civitaiResources") or meta.get("resources") or []:
        if not isinstance(res, dict):
            continue
        if str(res.get("type") or "").casefold() != "lora":
            continue
        name = str(
            res.get("modelName") or res.get("name") or res.get("modelVersionName") or res.get("modelVersionId") or ""
        ).strip()
        if not name:
            continue
        weight = res.get("weight")
        loras.append(f"{name}({weight})" if weight is not None else name)
        if len(loras) >= 4:
            break
    return ", ".join(loras)


def _civitai_image_recipe_lines(item: dict, index: int) -> list[str]:
    """一张 civitai 示例图的生成配方（含版本 id，便于下钻）。"""
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    model = _civitai_model_label(item, meta)
    lines = [f"[{index}] 模型: {model}"]
    prompt = _meta_first(meta, "prompt", "Prompt")
    if prompt:
        lines.append(f"prompt: {_truncate(prompt, 400)}")
    neg = _meta_first(meta, "negativePrompt", "Negative prompt", "negative_prompt")
    if neg:
        lines.append(f"负向: {_truncate(neg, 200)}")
    lora_text = _civitai_lora_summary(meta)
    if lora_text:
        lines.append(f"LoRA: {lora_text}")
    sampler = _meta_first(meta, "sampler", "Sampler")
    steps = _meta_first(meta, "steps", "Steps")
    cfg = _meta_first(meta, "cfgScale", "CFG scale", "cfg")
    seed = _meta_first(meta, "seed", "Seed")
    if any(v is not None for v in (sampler, steps, cfg, seed)):
        lines.append(
            f"sampler/steps/cfg/seed: {sampler if sampler is not None else '?'} / "
            f"{steps if steps is not None else '?'} / "
            f"{cfg if cfg is not None else '?'} / "
            f"{seed if seed is not None else '?'}"
        )
    elif not meta:
        lines.append("配方参数: 原图未附带生成参数")
    vid = item.get("modelVersionId")
    vids = item.get("modelVersionIds")
    if vid:
        lines.append(f"model_version_id={vid}")
    elif isinstance(vids, list) and vids:
        extra = f" (关联版本: {', '.join(str(x) for x in vids[:5])})" if len(vids) > 1 else ""
        lines.append(f"model_version_id={vids[0]}{extra}")
    if item.get("url"):
        lines.append(f"图: {item['url']}")
    return lines


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiCivitaiSearchTool(FunctionTool[AstrAgentContext]):
    """civitai 图片配方 / 模型记录 / 版本详情查询。"""

    name: str = "comfyui_civitai_search"
    description: str = (
        "查询 civitai 在线资源。mode=images（默认）按关键词搜参考图并返回其完整生成配方"
        "（模型、提示词、采样器、步数、cfg、seed），确认家族兼容后可填进 comfyui_draw；"
        "mode=models 搜模型记录，列出各版本 baseModel、官方触发词与 model_version_id；"
        "mode=version 按 model_version_id 取单个版本详情与示例图配方。"
        "用户想参考某风格/某模型的作品、找现成提示词、或查某个 LoRA/底模在 civitai 上的"
        "触发词与兼容性时使用。本地已装模型信息改用 comfyui_lookup。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["images", "models", "version"],
                    "description": "images=搜图取配方（默认），models=搜模型记录，version=按版本 id 取详情",
                },
                "query": {
                    "type": "string",
                    "description": "搜索关键词（风格/角色/模型名）；images/models 模式必填，version 模式省略",
                },
                "limit": {
                    "type": "number",
                    "description": "返回条数（1-10，默认 5）",
                },
                "nsfw": {
                    "type": "boolean",
                    "description": "是否包含 NSFW 内容（默认 false）",
                },
                "sort": {
                    "type": "string",
                    "description": "排序：images 支持 Most Reactions(默认)/Most Comments/Newest；models 支持 Highest Rated(默认)/Most Downloaded/Newest",
                },
                "period": {
                    "type": "string",
                    "enum": ["Day", "Week", "Month", "Year", "AllTime"],
                    "description": "时间范围，默认 AllTime",
                },
                "types": {
                    "type": "string",
                    "description": "models 模式的类型过滤，如 LORA、Checkpoint、TextualInversion；省略搜全部类型",
                },
                "base_model": {
                    "type": "string",
                    "description": "Civitai baseModel 精确值（如 Flux.1 D、Illustrious）；比 model_family 更精确时优先",
                },
                "model_family": _RESOURCE_QUERY_PROPERTIES["model_family"],
                "model_id": {
                    "type": "number",
                    "description": "限定 civitai 模型 id：images 模式只看该模型的示例图",
                },
                "model_version_id": {
                    "type": "number",
                    "description": "civitai 模型版本 id：version 模式必填，images 模式按此过滤该版本的示例图",
                },
            },
            "required": [],
        }
    )
    client: CivitaiClient | None = None

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        try:
            return int(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _filter_version_images(version: dict | None, nsfw: bool, model_name: str = "") -> list[dict]:
        if not isinstance(version, dict):
            return []
        images = version.get("images") or []
        if not nsfw:
            images = [
                img for img in images
                if int(img.get("nsfwLevel") or 1) <= 1 and not img.get("nsfw")
            ]
        m_name = model_name or (version.get("model") or {}).get("name") or ""
        base_model = version.get("baseModel") or ""
        vid = version.get("id")
        enriched: list[dict] = []
        for img in images:
            if not isinstance(img, dict):
                continue
            row = dict(img)
            if vid and not row.get("modelVersionId"):
                row["modelVersionId"] = vid
            if m_name and not row.get("modelName"):
                row["modelName"] = m_name
            if base_model and not row.get("baseModel"):
                row["baseModel"] = base_model
            enriched.append(row)
        return enriched

    @staticmethod
    def _pick_recipe_images(items: list[dict], limit: int) -> list[dict]:
        seen: set[str] = set()
        with_meta: list[dict] = []
        without_meta: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            dedup_key = str(item.get("id") or item.get("url") or id(item))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            if _has_recipe_meta(item):
                with_meta.append(item)
            else:
                without_meta.append(item)
        return (with_meta + without_meta)[:limit]

    async def _images(self, query: str, kwargs: dict, limit: int, nsfw: bool) -> str:
        sort = _civitai_norm_sort(kwargs.get("sort"), _CIVITAI_IMAGE_SORTS, "Most Reactions")
        period = _civitai_norm_period(kwargs.get("period"))
        model_id = self._int_or_none(kwargs.get("model_id"))
        version_id = self._int_or_none(kwargs.get("model_version_id"))
        items: list[dict] = []
        if model_id is not None or version_id is not None:
            items = await self.client.search_images(
                query, limit=min(max(limit * 3, 12), 20), nsfw=nsfw, sort=sort, period=period,
                model_id=model_id, model_version_id=version_id,
            )
            if version_id is not None and not any(_has_recipe_meta(x) for x in items):
                v = await self.client.get_model_version(version_id)
                v_imgs = self._filter_version_images(v, nsfw)
                if v_imgs:
                    items = v_imgs + items
        else:
            # Civitai /api/v1/images ignores free-text `search`; resolve via matching models first.
            base_models = _civitai_base_models(kwargs)
            models = await self.client.search_models(
                query, limit=min(max(limit, 3), 5), sort="Highest Rated",
                period=period, base_models=base_models, nsfw=nsfw,
            )
            for m in models[:4]:
                versions = m.get("modelVersions") or []
                v0 = versions[0] if versions and isinstance(versions[0], dict) else None
                vid = self._int_or_none(v0.get("id") if v0 else None)
                if vid is None:
                    continue
                m_name = str(m.get("name") or "")
                b_model = str(v0.get("baseModel") or "")
                batch = await self.client.search_images(
                    "", limit=min(max(limit * 2, 6), 12), nsfw=nsfw,
                    sort=sort, period=period, model_version_id=vid,
                )
                for img in batch:
                    row = dict(img)
                    row.setdefault("modelName", m_name)
                    if b_model and not row.get("baseModel"):
                        row["baseModel"] = b_model
                    if not row.get("modelVersionId"):
                        row["modelVersionId"] = vid
                    items.append(row)
                if not any(_has_recipe_meta(x) for x in batch):
                    v_detail = await self.client.get_model_version(vid)
                    items.extend(self._filter_version_images(v_detail, nsfw, model_name=m_name))
                if sum(1 for x in items if _has_recipe_meta(x)) >= limit:
                    break
        items = self._pick_recipe_images(items, limit)
        if not items:
            return f"civitai 未找到相关图片：{query or f'version {version_id}'}"
        scope = f"搜索: {query}" if query else f"版本 {version_id or model_id} 的示例图"
        parts = [f"【civitai 配方参考】{scope}（{sort}/{period}）"]
        for i, item in enumerate(items, 1):
            parts.append("")
            parts.extend(_civitai_image_recipe_lines(item, i))
        return "\n".join(parts)

    async def _models(self, query: str, kwargs: dict, limit: int, nsfw: bool) -> str:
        sort = _civitai_norm_sort(kwargs.get("sort"), _CIVITAI_MODEL_SORTS, "Highest Rated")
        period = _civitai_norm_period(kwargs.get("period"))
        types = str(kwargs.get("types") or "").strip()
        base_models = _civitai_base_models(kwargs)
        items = await self.client.search_models(
            query, types=types, limit=limit, sort=sort, period=period,
            base_models=base_models, nsfw=nsfw,
        )
        if not items:
            return f"civitai 未找到匹配的模型：{query}"
        parts = [f"【civitai 模型】搜索: {query}（{sort}/{period}）"]
        if base_models:
            parts.append(f"baseModel 过滤: {', '.join(base_models)}")
        for i, item in enumerate(items, 1):
            stats = item.get("stats") or {}
            creator = (item.get("creator") or {}).get("username") or "?"
            parts.append("")
            parts.append(f"[{i}] {item.get('name') or '?'}（{item.get('type') or '?'}）by {creator} | model_id={item.get('id')}")
            dl = stats.get("downloadCount", 0)
            like = stats.get("thumbsUpCount", 0)
            if dl or like:
                parts.append(f"下载 {dl} | 点赞 {like}")
            for v in (item.get("modelVersions") or [])[:3]:
                base = v.get("baseModel") or "?"
                parts.append(f"  版本: {v.get('name') or '?'}（{base}）model_version_id={v.get('id')}")
                tw = [t for t in (v.get("trainedWords") or []) if t]
                parts.append(f"  触发词: {', '.join(tw[:10])}" if tw else "  触发词: 未记录")
            tags = [str(t).strip() for t in (item.get("tags") or []) if str(t).strip()]
            if tags:
                parts.append(f"标签: {', '.join(tags[:12])}")
            description = _truncate(item.get("description"), 320)
            if description:
                parts.append(f"用途说明: {description}")
        parts.append("")
        parts.append("用 model_version_id 调 mode=version 取该版本详情，或 mode=images+model_version_id 取它的示例图配方。")
        return "\n".join(parts)

    async def _version(self, kwargs: dict, nsfw: bool) -> str:
        version_id = self._int_or_none(kwargs.get("model_version_id"))
        if version_id is None:
            return "查询失败：mode=version 需要 model_version_id。"
        v = await self.client.get_model_version(version_id)
        if not v:
            return f"civitai 未找到版本 {version_id}。"
        model = v.get("model") or {}
        parts = [f"【civitai 版本】{model.get('name') or '?'}（{model.get('type') or '?'}）/ {v.get('name') or '?'}"]
        parts.append(f"model_id={v.get('modelId')} | model_version_id={v.get('id')} | baseModel: {v.get('baseModel') or '?'}")
        tw = [t for t in (v.get("trainedWords") or []) if t]
        parts.append(f"触发词: {', '.join(tw[:16])}" if tw else "触发词: 未记录")
        description = _truncate(v.get("description"), 400)
        if description:
            parts.append(f"说明: {description}")
        files = v.get("files") or []
        if files:
            f0 = files[0]
            size = f0.get("sizeKB")
            size_text = f"{size / 1024:.0f}MB" if isinstance(size, (int, float)) else "?"
            parts.append(f"文件: {f0.get('name') or '?'}（{size_text}）")
        images = v.get("images") or []
        if not nsfw:
            images = [img for img in images
                      if int(img.get("nsfwLevel") or 1) <= 1 and not img.get("nsfw")]
        if images:
            parts.append(f"示例图 {len(images)} 张，配方如下：")
            for i, item in enumerate(images[:3], 1):
                parts.append("")
                parts.extend(_civitai_image_recipe_lines(item, i))
        return "\n".join(parts)

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        if self.client is None:
            return "查询失败：civitai 接口未配置。"
        mode = str(kwargs.get("mode") or "images").strip().lower()
        if mode not in {"images", "models", "version"}:
            return "查询失败：mode 仅支持 images / models / version。"
        query = str(kwargs.get("query") or "").strip()
        try:
            limit = min(max(int(kwargs.get("limit") or 5), 1), 10)
        except (TypeError, ValueError):
            limit = 5
        nsfw = _as_bool(kwargs.get("nsfw", False))
        if mode == "version":
            return await self._version(kwargs, nsfw)
        if not query:
            return "查询失败：query 不能为空。"
        if mode == "models":
            return await self._models(query, kwargs, limit, nsfw)
        return await self._images(query, kwargs, limit, nsfw)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiLookupTool(FunctionTool[AstrAgentContext]):
    """统一查询规范词、作品系列与本地模型资源。"""

    name: str = "comfyui_lookup"
    description: str = (
        "查询角色/画师规范词、作品系列、本机底模/LoRA/CLIP/VAE/Embedding及已保存配方。"
        "model/lora 的 query 可跨家族按名称搜索；按用途挑选 LoRA 时填写生图家族，"
        "结果包含用途、推荐权重和已知触发词。选用文件名与家族须匹配生图工作流。"
        "model_info 按文件名取单个本地模型的完整元数据（标题/作者/标签/触发词/使用建议），civitai 在线记录用 comfyui_civitai_search。"
        "character 默认支持中文名与有来源的外观参考；artist 触发词写进 artist。"
        "detail=true 展开资源元数据、配方参数，或取样统计角色/画师作品 tag；source 可选 danbooru/gelbooru/animadex。"
        "recipe 按名称、用途或家族查已有配方，适合选用现成方案；查询不会修改配方。"
        "copyright 查询 AnimaDex 作品系列，page 指定页码。"
        "all 查询各类资源摘要或跨类别搜索；folder 按指定 ComfyUI 目录精确查找。"
        "refresh=true 刷新资源缓存，offset 按返回的 next_offset 翻页。"
    )
    parameters: dict = Field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["character", "artist", "copyright", "model", "lora",
                         "clip", "vae", "embedding", "all", "folder", "recipe",
                         "model_info"],
                "description": "查询对象；all=资源摘要/跨类别搜索，folder=指定目录，recipe=只读配方目录，model_info=单个本地模型元数据",
            },
            "query": {
                "type": "string",
                "description": "名称或用途关键词；角色/画师/作品系列/model_info 必填，资源浏览可省略",
            },
            "types": {
                "type": "string",
                "enum": ["LORA", "Checkpoint"],
                "description": "model_info 的模型类型限定；省略时按文件名自动识别",
            },
            "folder": {
                "type": "string",
                "description": "type=folder 时必填，如 checkpoints/loras/clip/vae/unet/diffusion_models",
            },
            "refresh": {"type": "boolean", "description": "强制同步资源缓存，默认 false"},
            "source": {
                "type": "string", "enum": ["auto", "danbooru", "gelbooru", "animadex"],
                "description": "角色/画师来源；默认 auto，copyright 使用 animadex",
            },
            "detail": {"type": "boolean", "description": "展开资源元数据、配方参数，或附带角色/画师作品 tag 统计，默认 false"},
            "sample_limit": {"type": "integer", "minimum": 1, "maximum": 50,
                             "description": "详细查询的作品取样数，默认 30"},
            "page": {"type": "integer", "minimum": 1, "description": "AnimaDex 页码，默认 1"},
            **_RESOURCE_QUERY_PROPERTIES,
        },
        "required": ["type"],
    })
    danbooru: DanbooruClient | None = None
    gelbooru: GelbooruClient | None = None
    animadex: AnimaDexClient | None = None
    character_lookup: CharacterTagLookup | None = None
    client: ComfyUIClient | None = None
    store: RecipeStore | None = None

    async def _resources(self, kind: str, query: str, kwargs: dict) -> str:
        if self.client is None:
            return "查询失败：ComfyUI 未配置。"
        refresh = _as_bool(kwargs.get("refresh"))
        if kind == "model" and not refresh:
            resources = await self.client.generation_resources()
        else:
            resources, _ = await self.client.list_resources(force_refresh=refresh)
        family = str(kwargs.get("model_family") or "").strip()
        limit = kwargs.get("limit", 5)
        offset = kwargs.get("offset", 0)
        include_unknown = _as_bool(kwargs.get("include_unknown"))
        detailed = _as_bool(kwargs.get("detail"))
        if kind == "folder":
            folder = str(kwargs.get("folder") or "").strip().lower()
            if not folder:
                return "查询失败：type=folder 时 folder 必填。"
            names = await self.client.list_models_folder(folder)
            if names is None:
                return f"查询失败：无法连接 ComfyUI 或目录 {folder} 不存在。"
            folder_kind = {"loras": "lora", "checkpoints": "model", "unet": "model",
                           "diffusion_models": "model"}.get(folder)
            if folder_kind:
                resources = {**resources, "lora_name" if folder_kind == "lora" else "unet_name": names}
                return _resource_page(self.client, resources, folder_kind, query, family,
                                      limit, offset, include_unknown, detailed)
            return self._file_page(folder, names, query, limit, offset)
        if kind in {"model", "lora"}:
            return _resource_page(self.client, resources, kind, query, family,
                                  limit, offset, include_unknown, detailed)
        fields = {"model": ("unet_name", "底模"), "lora": ("lora_name", "LoRA"),
                  "clip": ("clip_name", "CLIP"), "vae": ("vae_name", "VAE"),
                  "embedding": ("embeddings", "Embedding")}
        if kind == "all":
            if query:
                return _all_resource_search(self.client, resources, query, family,
                                            limit, offset, include_unknown, detailed)
            return "【资源数量摘要】\n" + "\n".join(
                f"{title}: {len(resources.get(field) or [])}" for field, title in fields.values()
            ) + "\n指定 type 和 model_family 浏览资源，填写 query 可跨类别搜索。"
        field, title = fields[kind]
        return self._file_page(title, resources.get(field) or [], query, limit, offset)

    @staticmethod
    def _file_page(title, names, query, limit, offset):
        names = sorted(name for name in names if query.casefold() in name.casefold())
        limit = max(1, page_number(limit, 5, 10))
        offset = page_number(offset)
        more = f"\nnext_offset={offset + limit}" if offset + limit < len(names) else ""
        return f"【{title}】共 {len(names)} 项\n" + "\n".join(names[offset:offset + limit]) + more

    async def _model_info(self, query: str, kwargs: dict) -> str:
        if self.client is None:
            return "查询失败：ComfyUI 未配置。"
        if not query:
            return "查询失败：type=model_info 需要 query 指定模型文件名。"
        resources, _ = await self.client.list_resources()
        requested = str(kwargs.get("types") or "").casefold()
        kinds = ["lora"] if requested == "lora" else ["model"] if requested == "checkpoint" else ["lora", "model"]
        family = str(kwargs.get("model_family") or "")
        matches = []
        for kind in kinds:
            names = resources.get("lora_name" if kind == "lora" else "unet_name") or []
            if family:
                names = filter_family(names, metadata_for(resources, kind), family,
                                      getattr(self.client, "resource_family_rules", []), kind)
            matches.extend((kind, n) for n in _match_resource(names, query))
        if len(matches) != 1:
            return "未找到唯一匹配模型；请指定 types、model_family 和含目录的完整文件名，或用 type=model/lora 浏览。"
        kind, filename = matches[0]
        found, evidence = resource_family(filename, metadata_for(resources, kind).get(filename),
                                           getattr(self.client, "resource_family_rules", []), kind)
        detail = await _local_model_info(self.client, filename, kind)
        return f"家族={found or 'unknown'}（{evidence}）\n{detail}"

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        kind = str(kwargs.get("type") or "").strip().lower()
        query = str(kwargs.get("query") or "").strip()
        if kind == "model_info":
            return await self._model_info(query, kwargs)
        if kind == "recipe":
            return lookup_recipes(self.store, query, kwargs)
        if kind in {"model", "lora", "clip", "vae", "embedding", "all", "folder"}:
            return await self._resources(kind, query, kwargs)
        if kind not in {"character", "artist", "copyright"}:
            return "查询失败：请从 type 的可用类型中选择查询对象。"
        if not query:
            return "查询角色/画师/作品系列时 query 必填。"
        source = str(kwargs.get("source") or "auto").strip().lower()
        detail = _as_bool(kwargs.get("detail"))
        if source not in {"auto", "danbooru", "gelbooru", "animadex"}:
            return "查询失败：source 仅支持 auto/danbooru/gelbooru/animadex。"
        if source == "animadex" or kind == "copyright":
            if kind == "copyright" and source not in {"auto", "animadex"}:
                return "作品系列查询请使用 source=animadex。"
            return await lookup_animadex(self.animadex, kind, query, kwargs.get("page", 1))
        if source == "auto" and kind == "character":
            if not detail:
                if self.character_lookup is not None:
                    limit = max(1, page_number(kwargs.get("limit"), 5, 8))
                    return format_character_result(await self.character_lookup.lookup(query, limit))
                if self.animadex is not None:
                    return await lookup_animadex(self.animadex, kind, query, kwargs.get("page", 1))
            elif self.character_lookup is not None and any(ord(c) > 127 for c in query):
                limit = max(1, page_number(kwargs.get("limit"), 5, 8))
                resolved = await self.character_lookup.lookup(query, limit)
                if resolved.get("exact") and resolved.get("total") == 1 and resolved.get("matches"):
                    query = resolved["matches"][0]["tag"]
                elif resolved.get("matches"):
                    return (
                        format_character_result(resolved)
                        + "\n提示：请用以上英文角色 tag 重新查询（detail=true）以统计作品常用标签。"
                    )
        return await lookup_booru(
            self.danbooru, self.gelbooru, kind, query,
            source="danbooru" if source == "auto" else source,
            detail=detail, sample_limit=kwargs.get("sample_limit", 30),
        )
