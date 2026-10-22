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


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiCivitaiSearchTool(FunctionTool[AstrAgentContext]):
    """civitai 搜图查生成配方。"""

    name: str = "comfyui_civitai_search"
    description: str = (
        "在 civitai 搜索参考图并返回其完整生成配方（模型、正向/负向提示词、采样器、"
        "步数、cfg、seed），可在确认家族和兼容性后填进 comfyui_draw 的参数。"
        "用户想参考某风格/某模型的作品或找现成提示词时使用。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词（风格/角色/模型名等）",
                },
                "limit": {
                    "type": "number",
                    "description": "返回配方条数（1-10，默认 5）",
                },
                "nsfw": {
                    "type": "boolean",
                    "description": "是否包含 NSFW 内容（默认 false）",
                },
            },
            "required": ["query"],
        }
    )
    client: CivitaiClient | None = None

    @staticmethod
    def _truncate(s: str, n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[:n] + "…"

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return "查询失败：query 不能为空。"
        try:
            limit = min(max(int(kwargs.get("limit") or 5), 1), 10)
        except (TypeError, ValueError):
            limit = 5
        nsfw = _as_bool(kwargs.get("nsfw", False))

        items = await self.client.search_images(query, limit, nsfw)
        if not items:
            return f"civitai 未找到相关图片：{query}"

        parts = [f"【civitai 配方参考】搜索: {query}（按最多反应排序）"]
        for i, item in enumerate(items, 1):
            meta = item.get("meta") or {}
            model = (
                meta.get("Model")
                or meta.get("model")
                or item.get("modelName")
                or "未知模型"
            )
            parts.append("")
            parts.append(f"[{i}] 模型: {model}")
            if meta.get("prompt"):
                parts.append(f"prompt: {self._truncate(meta['prompt'], 400)}")
            if meta.get("negativePrompt"):
                parts.append(f"负向: {self._truncate(meta['negativePrompt'], 200)}")
            sampler = meta.get("sampler") or "?"
            steps = meta.get("steps") or "?"
            cfg = meta.get("cfgScale") or "?"
            seed = meta.get("seed") or "?"
            parts.append(f"sampler/steps/cfg/seed: {sampler} / {steps} / {cfg} / {seed}")
            url = item.get("url") or ""
            if url:
                parts.append(f"图: {url}")
        return "\n".join(parts)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiModelInfoTool(FunctionTool[AstrAgentContext]):
    """模型/LoRA 元数据与触发词查询（本地 safetensors 头部 / civitai trainedWords）。"""

    name: str = "comfyui_model_info"
    description: str = (
        "查询模型或 LoRA 的元数据与触发词：source=local 读 本机 上已装模型的 "
        "safetensors 头部信息（标题/作者/标签/训练触发词）；source=civitai 按名称搜索 "
        "civitai 模型记录（含官方 trainedWords 触发词）。用户想知道某个模型/LoRA 是干嘛的、"
        "触发词是什么时使用。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "模型文件名（local，如 miaomiaoRealskin_anima11.safetensors）或模型名（civitai，如 anima）",
                },
                "source": {
                    "type": "string",
                    "enum": ["local", "civitai"],
                    "description": "查询源：local=本地已装模型（默认），civitai=在线搜索",
                },
                "types": {
                    "type": "string",
                    "enum": ["LORA", "Checkpoint"],
                    "description": "模型类型：LORA 或 Checkpoint（底模）；本地可省略并自动识别",
                },
                "model_family": _RESOURCE_QUERY_PROPERTIES["model_family"],
            },
            "required": ["name"],
        }
    )
    client: ComfyUIClient | None = None
    civitai: CivitaiClient | None = None

    @staticmethod
    def _extract_trigger_words(meta: dict) -> list[str]:
        """从 safetensors 元数据提取触发词，兼容字符串/嵌套频率表。"""
        words, _source = ComfyUIClient._extract_trigger_words(meta)
        return words

    async def _query_local(self, name: str, kind: str = "lora") -> str:
        # LoRA Manager 保存的 sidecar/Civitai 信息比 safetensors 头部更完整，
        # 尤其是 style/character 分类、用途说明和推荐权重。
        manager_meta = await self.client.get_lora_manager_metadata(name) if kind == "lora" else None
        manager_info = ComfyUIClient.normalize_lora_metadata(manager_meta)
        if manager_info:
            lines = [f"【LoRA Manager】{name}"]
            lines.extend(_lora_info_summary(manager_info, detailed=True))
            return "\n".join(lines)

        meta = await self.client.get_model_metadata(
            name, folders=["loras"] if kind == "lora" else ["checkpoints", "diffusion_models", "unet"]
        )
        if not meta:
            return f"本地未找到模型 {name}，或该文件没有元数据头部（可试 source=civitai 在线搜索）。"

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
        triggers = self._extract_trigger_words(meta)
        if triggers:
            parts.append(f"触发词: {', '.join(triggers)}")
        elif not title and not author and not tags:
            # 有元数据但都是技术字段：给个头部键名摘要
            keys = [k for k in meta if not k.startswith("ss_")]
            if keys:
                parts.append("元数据键: " + ", ".join(keys[:12]))
        return "\n".join(parts)

    @staticmethod
    def _query_civitai_items(items: list[dict], query: str) -> str:
        if not items:
            return f"civitai 未找到匹配的模型：{query}"
        parts = [f"【civitai 模型】搜索: {query}"]
        for i, item in enumerate(items[:3], 1):
            name = item.get("name") or "?"
            mtype = item.get("type") or "?"
            creator = (item.get("creator") or {}).get("username") or "?"
            stats = item.get("stats") or {}
            parts.append("")
            parts.append(f"[{i}] {name}（{mtype}）by {creator}")
            dl = stats.get("downloadCount", 0)
            like = stats.get("thumbsUpCount", 0)
            if dl or like:
                parts.append(f"下载 {dl} | 点赞 {like}")
            versions = item.get("modelVersions") or []
            if versions:
                v = versions[0]
                vname = v.get("name") or "?"
                base = v.get("baseModel") or "?"
                parts.append(f"版本: {vname}（{base}）")
                tw = [t for t in (v.get("trainedWords") or []) if t]
                if tw:
                    parts.append(f"触发词: {', '.join(tw[:10])}")
                else:
                    parts.append("触发词: 未记录")
            tags = [str(tag).strip() for tag in (item.get("tags") or []) if str(tag).strip()]
            if tags:
                parts.append(f"标签: {', '.join(tags[:12])}")
            description = " ".join(str(item.get("description") or "").split())
            if description:
                suffix = "…" if len(description) > 320 else ""
                parts.append(f"用途说明: {description[:320]}{suffix}")
        return "\n".join(parts)

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        name = str(kwargs.get("name") or "").strip()
        if not name:
            return "查询失败：name 不能为空。"
        source = str(kwargs.get("source") or "local").strip().lower()

        if source == "local":
            if self.client is None:
                return "查询失败：本地模型接口未配置。"
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
                matches.extend((kind, n) for n in _match_resource(names, name))
            if len(matches) != 1:
                return "未找到唯一匹配模型；请指定 types、model_family 和含目录的完整文件名，先用 comfyui_lookup 查询。"
            kind, filename = matches[0]
            found, evidence = resource_family(filename, metadata_for(resources, kind).get(filename),
                                               getattr(self.client, "resource_family_rules", []), kind)
            detail = await self._query_local(filename, kind)
            return f"家族={found or 'unknown'}（{evidence}）\n{detail}"

        if source == "civitai":
            if self.civitai is None:
                return "查询失败：civitai 接口未配置。"
            types = str(kwargs.get("types") or "LORA").strip()
            items = await self.civitai.search_models(name, types=types, limit=3)
            family = canonical_family(kwargs.get("model_family"))
            if family:
                filtered = []
                for item in items:
                    versions = [v for v in item.get("modelVersions") or []
                                if resource_family("", {"base_model": v.get("baseModel")})[0] == family]
                    if versions:
                        filtered.append({**item, "modelVersions": versions})
                items = filtered
            return self._query_civitai_items(items, name)

        return "查询失败：source 仅支持 local / civitai。"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiLookupTool(FunctionTool[AstrAgentContext]):
    """统一查询规范词、作品系列与本地模型资源。"""

    name: str = "comfyui_lookup"
    description: str = (
        "查询角色/画师规范词、作品系列、本机底模/LoRA/CLIP/VAE/Embedding及已保存配方。"
        "model/lora 的 query 可跨家族按名称搜索；按用途挑选 LoRA 时填写生图家族，"
        "结果包含用途、推荐权重和已知触发词。选用文件名与家族须匹配生图工作流。"
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
                         "clip", "vae", "embedding", "all", "folder", "recipe"],
                "description": "查询对象；all=资源摘要/跨类别搜索，folder=指定目录，recipe=只读配方目录",
            },
            "query": {
                "type": "string",
                "description": "名称或用途关键词；角色/画师/作品系列必填，资源浏览可省略",
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

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        kind = str(kwargs.get("type") or "").strip().lower()
        query = str(kwargs.get("query") or "").strip()
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
        if source == "auto" and kind == "character" and not detail:
            if self.character_lookup is not None:
                limit = max(1, page_number(kwargs.get("limit"), 5, 8))
                return format_character_result(await self.character_lookup.lookup(query, limit))
            if self.animadex is not None:
                return await lookup_animadex(self.animadex, kind, query, kwargs.get("page", 1))
        return await lookup_booru(
            self.danbooru, self.gelbooru, kind, query,
            source="danbooru" if source == "auto" else source,
            detail=detail, sample_limit=kwargs.get("sample_limit", 30),
        )
