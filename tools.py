"""LLM 工具定义（dataclass FunctionTool 模式，v4.5.7+ 推荐）。

comfyui_draw：按模型家族选择工作流，自由生图（默认启用）
comfyui_recipe_draw：复用保存的配方参数快捷生图（默认启用）
comfyui_lookup：查询角色/画师/底模/LoRA，支持按用途选择 LoRA（默认启用）
comfyui_list_models：查询模型/LoRA/CLIP/VAE/Embedding 清单（自动同步缓存）
comfyui_generate：生成图片（可选模型/LoRA/KSampler 参数）
comfyui_interrupt：中断生成 / 取消排队任务
comfyui_queue：查询队列与 GPU 状态
comfyui_booru：danbooru/gelbooru 查画师/角色触发词
comfyui_civitai_search：civitai 搜图查生成配方
comfyui_model_info：查询模型/LoRA 元数据与触发词（本地 / civitai）
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from astrbot.api import FunctionTool, logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import Image, Reply
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from character_tags import CharacterTagLookup, format_character_result
from generation_support import fill_trigger_words
from animadex import AnimaDexClient
from comfy_client import ComfyUIClient, execution_error_message, image_dimensions, image_media_type
from image_cache import save_image
from external_search import CivitaiClient, DanbooruClient, GelbooruClient
from model_families import (
    EditWorkflowRegistry,
    ModelFamily,
    ModelFamilyRegistry,
    WorkflowProfileStore,
)
from recipe_store import (
    RecipeStore,
    materialize_values,
    recipe_family,
    recipe_template,
    resolve_generation_entry,
)
from slot_mapping import (
    ANIMA_DROP_NODES,
    apply_slots,
    detect_slots,
    looks_like_anima,
    parse_lora,
    read_current_values,
    resolve_size,
    source_image_slots,
)
from resource_catalog import (
    canonical_family, exact_matches, explicitly_named, family_summary,
    filename_matches, filter_family, metadata_for, page_number, resource_family,
    selection_error,
)
from workflow_builder import WorkflowBuilder

MAX_LLM_LIST_ITEMS = 30
MAX_LLM_DETAIL_ITEMS = 8
_ASYNC_GENERATION_HINT = (
    "提交后立即返回排队确认，可以继续聊天；完成后自动发送图片并通知。"
    "自然回应用户，无需播报内部编号、轮询或重复提交。"
)


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse tool booleans safely when a provider sends JSON values as text."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n", ""}:
        return False
    return default


def _bounded_limit(value: Any, default: int = MAX_LLM_LIST_ITEMS) -> int:
    try:
        return min(max(int(value), 1), 50)
    except (TypeError, ValueError):
        return default


def _number(value: Any, label: str, *, integer: bool = False,
            minimum: float | None = None, maximum: float | None = None) -> int | float:
    """Validate a generation number and return a plain finite int/float."""
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{label} 必须是数字") from e
    if not math.isfinite(number):
        raise ValueError(f"{label} 不能是 NaN 或无穷大")
    if integer and not number.is_integer():
        raise ValueError(f"{label} 必须是整数")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} 不能小于 {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} 不能大于 {maximum:g}")
    return int(number) if integer else number


def _validate_generation_values(values: dict[str, Any]) -> dict[str, Any]:
    """Normalize numeric generation values before workflow mutation."""
    result = dict(values)
    limits = {
        "seed": (True, 0, 2**63 - 1),
        "steps": (True, 1, 1000),
        "cfg": (False, 0, 100),
        "denoise": (False, 0, 1),
        "width": (True, 64, 8192),
        "height": (True, 64, 8192),
        "megapixels": (False, 0.1, 64),
    }
    for key, (integer, minimum, maximum) in limits.items():
        value = result.get(key)
        if value in (None, ""):
            continue
        result[key] = _number(
            value,
            key,
            integer=integer,
            minimum=minimum,
            maximum=maximum,
        )
    return result


def _edit_canvas_dimensions(width: int, height: int, resolution: int) -> tuple[int, int]:
    """Fit the reference image into a square resolution bound, preserving its ratio."""
    if resolution == 0:
        return width, height
    scale = resolution / max(width, height)
    scaled_width = max(8, int(round(width * scale / 8) * 8))
    scaled_height = max(8, int(round(height * scale / 8) * 8))
    return scaled_width, scaled_height


def _custom_edit_canvas_dimensions(
    source_dimensions: tuple[int, int] | None,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    """Apply an explicit canvas size; infer one missing side from the reference ratio."""
    if width is None and height is None:
        raise ValueError("请至少指定 width 或 height")
    if width is None or height is None:
        if source_dimensions is None:
            raise ValueError("无法读取来源图片宽高，不能按参考图比例补齐画布尺寸")
        source_width, source_height = source_dimensions
        if width is None:
            width = int(round(height * source_width / source_height / 8) * 8)
        if height is None:
            height = int(round(width * source_height / source_width / 8) * 8)
    width, height = max(8, int(round(width / 8) * 8)), max(8, int(round(height / 8) * 8))
    if not (64 <= width <= 8192 and 64 <= height <= 8192):
        raise ValueError("推算后的画布宽高需在 64 到 8192 之间，请同时指定 width 和 height")
    return width, height


def _event_scope(context: ContextWrapper[AstrAgentContext]) -> str:
    """Return a conversation scope for per-session task state."""
    event = getattr(getattr(context, "context", None), "event", None)
    if event is None:
        return "global"
    umo = getattr(event, "unified_msg_origin", None)
    if umo:
        return str(umo)
    get_session_id = getattr(event, "get_session_id", None)
    if callable(get_session_id):
        try:
            value = get_session_id()
            if value:
                return str(value)
        except Exception:
            pass
    try:
        sender = event.get_sender_id()
        if sender:
            return f"sender:{sender}"
    except Exception:
        pass
    return "global"


def _remember_prompt_id(
    shared: dict, context: ContextWrapper[AstrAgentContext], prompt_id: str
) -> None:
    """Keep the most recent task per conversation, not one global task."""
    by_scope = shared.setdefault("last_prompt_ids", {})
    by_scope[_event_scope(context)] = str(prompt_id)


def _last_prompt_id(shared: dict, context: ContextWrapper[AstrAgentContext]) -> str:
    by_scope = shared.get("last_prompt_ids") or {}
    return str(by_scope.get(_event_scope(context)) or "")


def _remember_image_path(shared: dict, context: ContextWrapper[AstrAgentContext], path: Path) -> None:
    shared.setdefault("last_image_paths", {})[_event_scope(context)] = str(path)


async def _enqueue_generation(shared: dict, context, workflow: dict, history: dict,
                              saved_recipe: dict | None = None) -> str | None:
    queue = shared.get("generation_queue")
    if queue is None:
        return None
    try:
        return await queue.enqueue(workflow, context.context.event, history=history, saved_recipe=saved_recipe)
    except (OSError, ValueError) as error:
        return f"加入队列失败：{error}"


def _message_images(context: ContextWrapper[AstrAgentContext]) -> list[Image]:
    event = getattr(getattr(context, "context", None), "event", None)
    message = getattr(getattr(event, "message_obj", None), "message", None) or []
    images: list[Image] = []
    for component in message:
        if isinstance(component, Image):
            images.append(component)
        elif isinstance(component, Reply):
            images.extend(item for item in (component.chain or []) if isinstance(item, Image))
    return images


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
    if detailed and info.get("description"):
        lines.append("用途说明: " + str(info["description"]))
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


_RESOURCE_QUERY_PROPERTIES = {
    "model_family": {"type": "string", "description": "资源家族，如 anima/krea2/sdxl/flux/illustrious；unknown 查未识别项。按名称搜索时可省略，按 LoRA 用途挑选时请填写生图家族"},
    "limit": {"type": "integer", "description": "每页数量，默认 5，最多 10"},
    "offset": {"type": "integer", "description": "分页偏移，默认 0；使用返回的 next_offset"},
    "include_unknown": {"type": "boolean", "description": "同时显示家族未知项，默认 false；未知项需核实兼容性"},
}


def _resource_page(client, resources, kind, query="", family="", limit=5, offset=0,
                   include_unknown=False) -> str:
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
            lines.extend("  " + line[:240] for line in _lora_info_summary(meta.get(name) or {}, detailed=True))
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
                         include_unknown=False) -> str:
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
            lines.extend("  " + line[:240] for line in _lora_info_summary(info, detailed=True))
    if offset + limit < len(rows):
        lines.append(f"next_offset={offset + limit}")
    if not shown:
        lines.append("无匹配项；可换用文件名片段，或显式查询 model_family=unknown / include_unknown=true。")
        if not family:
            lines.append("按 LoRA 用途、类别或标签查询时，请指定 kind=lora 和生图工作流的 model_family。")
    if include_unknown or canonical_family(family) == "unknown" or any(
        found == "unknown" for _, _, found, _, _, _ in shown
    ):
        lines.append("unknown 项未确认兼容性，请核实元数据或配置归类后选用。")
    return "\n".join(lines)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiListModelsTool(FunctionTool[AstrAgentContext]):
    """查询 ComfyUI 可用模型/LoRA/CLIP 清单。"""

    name: str = "comfyui_list_models"
    description: str = (
        "查询本机上ComfyUI可用的UNET底模、LoRA、CLIP、VAE、Embedding列表。"
        "给出 query 可按模型名称跨家族搜索；缺少 kind 时同时搜索各类资源，结果标注家族。"
        "LoRA 会附带触发词，以及 LoRA Manager/Civitai 的 style、character 等类别、标签和使用建议。"
        "清单会自动同步并本地缓存，ComfyUI离线时返回最近一次同步结果。完整已知文件名可跨家族查询；类别/标签筛选需指定 model_family。"
        "用户询问可用资源，或绘图时需要按画风、角色、服饰、效果挑选已安装 LoRA 时使用。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "refresh": {
                    "type": "boolean",
                    "description": "是否强制重新从 ComfyUI 同步一次，默认 false（用本地缓存）",
                },
                "kind": {
                    "type": "string",
                    "enum": ["all", "model", "unet", "lora", "clip", "vae", "embedding"],
                    "description": "只查看某一类资源，model/unet 都表示底模，默认 all",
                },
                "query": {
                    "type": "string",
                    "description": "模型名或文件名片段支持跨家族近似搜索；LoRA 类别/标签搜索需指定 model_family",
                },
                "limit": {
                    "type": "number",
                    "description": "每页数量，默认 5，最多 10",
                },
                **_RESOURCE_QUERY_PROPERTIES,
            },
        }
    )
    client: ComfyUIClient | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        force = _as_bool(kwargs.get("refresh", False))
        kind = str(kwargs.get("kind") or "all").strip().lower()
        if kind == "model":
            kind = "unet"
        field_titles = {
            "unet": ("unet_name", "UNET 底模"),
            "lora": ("lora_name", "LoRA"),
            "clip": ("clip_name", "CLIP"),
            "vae": ("vae_name", "VAE"),
            "embedding": ("embeddings", "Embedding"),
        }
        if kind != "all" and kind not in field_titles:
            return "查询失败：kind 仅支持 all/unet/lora/clip/vae/embedding。"
        resources, from_cache = await self.client.list_resources(force_refresh=force)
        family = str(kwargs.get("model_family") or "").strip()
        query = str(kwargs.get("query") or "").strip()
        if kind == "all":
            if query:
                return _all_resource_search(self.client, resources, query, family,
                                            kwargs.get("limit", 5), kwargs.get("offset", 0),
                                            _as_bool(kwargs.get("include_unknown")))
            return "【资源数量摘要】\n" + "\n".join(
                f"{title}: {len(resources.get(field) or [])}" for field, title in field_titles.values()
            ) + "\n填写 query 可跨类别按名称搜索；指定 kind 和 model_family 可浏览底模或 LoRA。"
        if kind in {"unet", "lora"}:
            return _resource_page(self.client, resources, "model" if kind == "unet" else "lora",
                                  query, family, kwargs.get("limit", 5), kwargs.get("offset", 0),
                                  _as_bool(kwargs.get("include_unknown")))
        field, title = field_titles[kind]
        items = sorted(n for n in resources.get(field) or [] if query.casefold() in n.casefold())
        limit = max(1, page_number(kwargs.get("limit"), 5, 10))
        offset = page_number(kwargs.get("offset"))
        shown = items[offset:offset + limit]
        more = f"\nnext_offset={offset + limit}" if offset + limit < len(items) else ""
        return f"【{title}】共 {len(items)} 项\n" + "\n".join(shown) + more


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiGenerateTool(FunctionTool[AstrAgentContext]):
    """通过 ComfyUI 生成图片。"""

    name: str = "comfyui_generate"
    description: str = (
        "高级兼容生成入口，按默认配方或指定工作流生成图片并发送到当前会话。"
        "日常自由生图使用 comfyui_draw，快捷配方生图使用 comfyui_recipe_draw。"
        "prompt 必填；未覆盖的参数沿用配方、插件配置或模板默认值，width/height 可按构图需求填写。"
        "当 LoRA 有助于实现用户要求的画风、角色、服饰或效果时，可主动查询并选用，用户无需点名 LoRA 或提供文件名。"
        "按用途查询 LoRA 时使用 comfyui_lookup(type=\"lora\", model_family=所选家族, query=需求关键词)，"
        "依据用途说明、兼容性和推荐权重选择，再把实际文件名写入 lora 对象数组。"
        "使用已记录的触发词时同步填写 trigger_words，保留原词格式；查询未提供触发词时可省略该字段并继续使用 LoRA。"
        "省略 lora 会沿用默认设置，传入列表会覆盖映射的可选 LoRA；已有独立加速节点的模板始终沿用其加速设置。"
        "用户明确要求关闭可选 LoRA 时传空数组，该操作只关闭映射槽位。"
        "角色/画师名称不确定时用 comfyui_lookup 查询；底模、采样参数等按用户要求调整，其余沿用默认值。"
        "recipe 与 workflow 是两个独立入口：传 recipe 时按其模型家族解析当前工作流并写入保存参数；"
        "传 workflow 时按该模板生成、不套配方；两者都省略时优先默认配方，没有默认配方才用配置的默认工作流模板。"
        "生成成功后回复结果即可，图片已由插件发送。"
        + _ASYNC_GENERATION_HINT
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "主提示词，按所选工作流或家族的 prompt_style 编写；人物、动作、场景与构图依用户要求填写（必填）",
                },
                "artist": {
                    "type": "string",
                    "description": "画师串，格式如 @画师名，逗号分隔。用户指定画师时填写，可用 comfyui_lookup 查询规范名称；省略则沿用默认画师。一般画风需求也可通过提示词或 LoRA 实现",
                },
                "trigger_words": {
                    "type": "string",
                    "description": "所选 LoRA 的已知触发词，从 lookup/model_info 返回值或用户提供的信息中取用，保留原始格式，多个用逗号分隔。选用 LoRA 时可同步填写，无需用户另外提出；查询未提供时可省略并继续使用 LoRA",
                },
                "quality": {
                    "type": "string",
                    "description": "质量词串（masterpiece, best quality, score_9 等）。用户要求画质/光影时填，否则留空用模板默认",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "负向提示词。用户有特殊负向要求时填，否则留空用模板默认",
                },
                "model": {
                    "type": "string",
                    "description": "底模文件名。必须用 comfyui_list_models 返回的完整名字（可能带子目录前缀，如 Anima\\miaomiaoHarem_anima16.safetensors）；传短名会自动匹配，匹配到多份会要求重填。用户指定底模时填，不知道文件名先查 comfyui_list_models",
                },
                "lora": {
                    **_lora_array_schema(
                        "按查询结果提供 LoRA 对象数组，可附推荐 strength。按已映射槽位覆盖可选 LoRA；独立加速 LoRA 保留。省略沿用默认，传空数组关闭可选 LoRA"
                    ),
                },
                "steps": {
                    "type": "number",
                    "description": "采样步数。不传用插件配置默认",
                },
                "cfg": {
                    "type": "number",
                    "description": "CFG。不传用插件配置默认",
                },
                "sampler_name": {
                    "type": "string",
                    "description": "采样器名，如 er_sde。不传用插件配置默认",
                },
                "scheduler": {
                    "type": "string",
                    "description": "调度器，如 normal。不传用插件配置默认",
                },
                "denoise": {
                    "type": "number",
                    "description": "降噪强度 0~1。不传用插件配置默认",
                },
                "width": {
                    "type": "number",
                    "description": "图片宽度。不传用插件配置默认",
                },
                "height": {
                    "type": "number",
                    "description": "图片高度。不传用插件配置默认",
                },
                "seed": {
                    "type": "number",
                    "description": "随机种子。不传随机",
                },
                "workflow": {
                    "type": "string",
                    "description": "工作流模板入口：传模板名或 JSON 路径时按该模板生成，不套配方（显式 recipe 优先于 workflow）。两者都省略时先用默认配方，没有默认配方再用插件配置的默认模板",
                },
                "recipe": {
                    "type": "string",
                    "description": "配方入口：已保存的配方名。传入后通过配方的模型家族选择工作流，配方参数兜底，本次显式参数优先；不传则用当前默认配方",
                },
            },
            "required": ["prompt"],
        }
    )
    client: ComfyUIClient | None = None
    builder: WorkflowBuilder | None = None
    output_dir: Path | None = None
    shared: dict = Field(default_factory=dict)  # 跨工具共享状态（如 last_prompt_id）
    auto_trigger_words: bool = False
    defaults: dict = Field(default_factory=dict)  # 插件配置里的生成默认值（LLM 不传时使用）
    store: RecipeStore | None = None  # 配方存储（统一用 RecipeStore）
    families: ModelFamilyRegistry | None = None
    profiles: WorkflowProfileStore | None = None

    @staticmethod
    def _pick(defaults: dict, key: str, value: Any) -> Any:
        """参数优先级：LLM 传值 > 配置默认 > None（模板原值）。

        "" / 0 视为"未配置"，保持模板原值。
        """
        if value is not None:
            return value
        d = defaults.get(key)
        if d in (None, "", 0, 0.0):
            return None
        return d

    def _load_recipe(self, name: str) -> dict | None:
        """读配方并转成 generate 参数；省略 name 时读取当前默认配方。"""
        recipe = self._load_recipe_data(name)
        if recipe is None:
            return None
        # 把 RecipeStore 格式转成 generate 工具期望的 flat kwargs
        defaults = dict(recipe.get("defaults") or {})
        flat: dict[str, Any] = {}
        flat["workflow"] = recipe_template(recipe)
        # defaults 里的 loras 要转回 lora 参数串
        loras = defaults.pop("loras", None) if isinstance(defaults, dict) else None
        for k, v in defaults.items():
            flat[k] = v
        if "negative" in flat and "negative_prompt" not in flat:
            flat["negative_prompt"] = flat.pop("negative")
        if loras is not None:
            flat["lora"] = json.dumps(loras, ensure_ascii=False)
        return flat

    def _load_recipe_data(self, name: str) -> dict | None:
        """读取完整配方，供 generate 保留 workflow 与槽位映射。"""
        if not self.store:
            return None
        return self.store.get(name) if name else self.store.default()

    async def _wait_outputs(self, prompt_id: str) -> tuple[dict | None, str | None]:
        return await _wait_outputs(self.client, prompt_id)


    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        prepared_resources = None
        async def resources_for_request():
            nonlocal prepared_resources
            if prepared_resources is None:
                prepared_resources = await self.client.generation_resources()
            return prepared_resources

        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            return "生成失败：prompt（主提示词）不能为空。"

        # 双入口判定：显式 recipe > 显式 workflow > 默认配方 > 配置默认模板。
        # 显式 workflow 必须强制走模板入口，避免被默认配方静默吞掉。
        recipe_name = str(kwargs.get("recipe") or "").strip()
        workflow_param = str(kwargs.get("workflow") or "").strip()
        has_default = False
        if not recipe_name and not workflow_param and self.store is not None:
            has_default = self.store.default() is not None
        entry, entry_name = resolve_generation_entry(
            recipe_name, workflow_param, has_default_recipe=has_default
        )
        explicit_kwargs = dict(kwargs)
        recipe_data = None
        if entry == "recipe":
            recipe_data = self._load_recipe_data(entry_name)
            if recipe_data is None:
                return f"生成失败：配方不存在（{entry_name}）。可先 comfyui_recipe list 查看。"
            recipe_family_entry = (
                self.families.resolve_recipe(recipe_data) if self.families is not None else None
            )
            if recipe_family_entry is None and not (recipe_data.get("slots") or {}).get("prompt"):
                display = str(recipe_data.get("name") or entry_name or "默认")
                return (
                    f"生成失败：配方「{display}」还没指定主提示词节点，"
                    "请在 Workflow Studio 工作流页或配置下拉框中确认节点映射。"
                )
            merged = dict(self._load_recipe(entry_name) or {})
            merged.pop("name", None)
            merged.pop("prompt", None)  # prompt 以本参数为准
            # 把配方值塞进 kwargs（LLM 传的值覆盖），供底模短名解析沿用
            kwargs = {**merged, **kwargs}
            prompt = str(kwargs.get("prompt") or "").strip() or prompt

        seed = kwargs.get("seed")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        try:
            pick = lambda key: self._pick(self.defaults, key, kwargs.get(key))  # noqa: E731
            # 底模名解析：传短名/basename 时在资源列表里匹配出完整路径（如 Anima\xxx.safetensors），
            # 避免 ComfyUI 校验 Value not in list；列表拉不到则保持原值交给提交阶段报错。
            model_val = pick("model")
            if model_val and self.client is not None:
                try:
                    resources = await resources_for_request()
                    hits = _match_resource(resources.get("unet_name") or [], str(model_val))
                except Exception:
                    hits = []
                if len(hits) == 1:
                    kwargs["model"] = hits[0]
                elif len(hits) > 1:
                    preview = "、".join(hits[:5])
                    return (
                        f"生成失败：底模「{model_val}」匹配到多份：{preview}。"
                        "请让用户选一个或填完整文件名。"
                    )
            lora_val = pick("lora")
            trigger_val = pick("trigger_words")
            generation_values = _validate_generation_values(
                {
                    "seed": seed,
                    "width": pick("width"),
                    "height": pick("height"),
                    "steps": pick("steps"),
                    "cfg": pick("cfg"),
                    "denoise": pick("denoise"),
                }
            )
            seed = generation_values["seed"]
            # 自动填 lora 触发词已禁用（33号要求，lora_meta 触发词乱提示），需要时显式传 trigger_words
            # Keep output-node inputs stable so repeated requests can reuse its cache.
            prefix = None
            if recipe_data is not None:
                # 配方生成必须走保存的 workflow + slots。此前这里把配方压平成
                # builder.build() 参数，导致 Power Lora Loader 的动态 lora_N
                # 插槽完全绕过，WebUI 能选到的 LoRA 在机器人调用时不会提交。
                values = materialize_values(
                    recipe_data,
                    {
                        "prompt": prompt,
                        "artist": explicit_kwargs.get("artist"),
                        "trigger_words": explicit_kwargs.get("trigger_words"),
                        "quality": explicit_kwargs.get("quality"),
                        "negative": explicit_kwargs.get("negative_prompt")
                        or explicit_kwargs.get("negative"),
                        # model_val 已完成资源名解析；配方里保存的短名也要沿用
                        # 解析后的完整路径，避免 ComfyUI 校验时再次丢失。
                        "model": kwargs.get("model") if model_val is not None else None,
                        "loras": (
                            parse_lora(
                                explicit_kwargs.get(
                                    "lora",
                                    explicit_kwargs.get("loras"),
                                )
                            )
                            if (
                                "lora" in explicit_kwargs
                                or "loras" in explicit_kwargs
                            )
                            and explicit_kwargs.get(
                                "lora",
                                explicit_kwargs.get("loras"),
                            )
                            not in (None, "")
                            else None
                        ),
                        "width": explicit_kwargs.get("width"),
                        "height": explicit_kwargs.get("height"),
                        "steps": explicit_kwargs.get("steps"),
                        "cfg": explicit_kwargs.get("cfg"),
                        "sampler_name": explicit_kwargs.get("sampler_name"),
                        "scheduler": explicit_kwargs.get("scheduler"),
                        "denoise": explicit_kwargs.get("denoise"),
                        "seed": generation_values.get("seed"),
                    },
                )
                for key, val in self.defaults.items():
                    if key not in values and val not in (None, "", 0, 0.0, []):
                        values["negative" if key == "negative_prompt" else key] = val
                values = _validate_generation_values(values)
                family_entry = (
                    self.families.resolve_recipe(recipe_data)
                    if self.families is not None
                    else None
                )
                if family_entry is not None and (values.get("model") or values.get("loras")):
                    resources = await resources_for_request()
                    error = selection_error(resources, values, family_entry.name,
                                            getattr(self.client, "resource_family_rules", []))
                    if error:
                        return error
                workflow_name = (
                    family_entry.workflow if family_entry is not None else recipe_template(recipe_data)
                )
                wf = self.builder.load_template(workflow_name or None)
                if family_entry is not None and self.profiles is not None:
                    profile = self.profiles.effective(workflow_name, wf)
                    recipe_slots = profile.get("slots") or {}
                    recipe_drop_nodes = list(profile.get("drop_nodes") or [])
                else:
                    recipe_slots = recipe_data.get("slots") or {}
                    recipe_drop_nodes = list(recipe_data.get("drop_nodes") or [])
                if self.auto_trigger_words and recipe_slots.get("trigger_words"):
                    trigger_resources = await resources_for_request()
                    names = [str(item.get("name")) for item in parse_lora(values.get("loras")) if item.get("name")]
                    if names:
                        trigger_resources = await self.client.selected_lora_metadata(trigger_resources, names, require_trusted=True)
                    values = fill_trigger_words(values, trigger_resources, True)
                apply_slots(
                    wf,
                    recipe_slots,
                    values,
                    prefix=prefix,
                    drop_nodes=recipe_drop_nodes,
                )
            else:
                wf = self.builder.build(
                    workflow=kwargs.get("workflow"),
                    prompt=prompt,
                    artist=pick("artist"),
                    trigger_words=trigger_val,
                    quality=pick("quality"),
                    negative_prompt=pick("negative_prompt"),
                    model=pick("model"),
                    lora=lora_val,
                    width=generation_values.get("width"),
                    height=generation_values.get("height"),
                    seed=generation_values.get("seed"),
                    steps=generation_values.get("steps"),
                    cfg=generation_values.get("cfg"),
                    sampler_name=pick("sampler_name"),
                    scheduler=pick("scheduler"),
                    denoise=generation_values.get("denoise"),
                    prefix=prefix,
                )
                family_entry = self.families.by_workflow(
                    kwargs.get("workflow") or self.builder.default_workflow
                ) if self.families is not None else None
                if family_entry is not None and (pick("model") or lora_val):
                    resources = await resources_for_request()
                    error = selection_error(resources,
                                            {"model": pick("model"), "loras": parse_lora(lora_val)},
                                            family_entry.name, getattr(self.client, "resource_family_rules", []))
                    if error:
                        return error
        except FileNotFoundError as e:
            return f"生成失败：{e}"
        except (ValueError, json.JSONDecodeError) as e:
            return f"生成失败：参数错误（{e}）"

        queued = await _enqueue_generation(self.shared, context, wf, {
            "entry": "legacy", "prompt": prompt,
            "workflow": kwargs.get("workflow") or self.builder.default_workflow,
            "values": generation_values,
        })
        if queued is not None:
            return queued
        pid, submit_err = await self.client.submit_prompt_detail(wf)
        if submit_err:
            return f"生成失败：{submit_err}"
        if not pid:
            return (
                "生成失败：无法连接 ComfyUI，请确认本机已开机且ComfyUI已启动、"
                "插件配置的地址（IP/端口）正确。"
            )
        _remember_prompt_id(self.shared, context, pid)

        outputs, wait_err = await self._wait_outputs(pid)
        if wait_err:
            return wait_err if wait_err.startswith(("等待超时", "等待中断")) else f"生成失败：{wait_err}"
        if outputs is None:
            return "生成失败：未获取到执行结果。"

        images = []
        for node_out in outputs.values():
            images.extend(node_out.get("images", []))
        if not images:
            return "生成似乎已完成，但未找到输出图片文件。"

        img = next((item for item in images if item.get("type", "output") == "output"), images[-1])
        filename = img["filename"]
        subfolder = img.get("subfolder", "")
        content = await self.client.download_image(filename, subfolder, image_type=img.get("type", "output"))
        if not content:
            return (
                f"图片已在ComfyUI生成（{filename}），但下载到本地失败。\n"
                f"可手动访问 {self.client.base_url}/view?filename={filename}&type=output 查看。"
            )

        try:
            local_path = await asyncio.to_thread(save_image, self.output_dir, filename, content)
        except (OSError, ValueError) as e:
            logger.error(f"[ComfyUIDirect] 写入图片失败: {e}")
            return f"图片已生成但本地保存失败（{e}）。文件名: {filename}"
        _remember_image_path(self.shared, context, local_path)

        try:
            event: AstrMessageEvent = context.context.event
            await event.send(MessageChain().file_image(str(local_path)))
        except Exception as e:
            logger.error(f"[ComfyUIDirect] 图片发送失败: {e}")
            return (
                f"图片已生成但自动发送失败（{e}）。\n"
                f"本地路径: {local_path}\n"
                f"请用 send_message_to_user 发送这张图。"
            )

        return (
            f"图片已生成并直接发送到会话。\n"
            f"本地路径: {local_path}\n"
            f"文件名: {filename}\n"
            f"prompt_id: {pid}"
        )


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiInterruptTool(FunctionTool[AstrAgentContext]):
    """中断生成 / 取消排队任务。"""

    name: str = "comfyui_interrupt"
    description: str = (
        "中断 ComfyUI 正在执行的生成任务（用户要求停止/改图时用）。"
        "不传 prompt_id 时中断最近一次由本插件提交的任务；可同时从待执行队列移除该任务。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt_id": {
                    "type": "string",
                    "description": "要中断的任务 ID（生成结果里返回的 prompt_id），可选；不传用最近一次生成的任务",
                },
                "remove_from_queue": {
                    "type": "boolean",
                    "description": "是否同时把该任务从待执行队列移除（默认 false，仅中断运行中的）",
                },
            },
        }
    )
    client: ComfyUIClient | None = None
    shared: dict = Field(default_factory=dict)

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        pid = str(kwargs.get("prompt_id") or "").strip() or _last_prompt_id(
            self.shared, context
        )
        queue = self.shared.get("generation_queue")
        identifier = str(kwargs.get("prompt_id") or "").strip()
        if queue is not None:
            return await queue.action("cancel", identifier, _event_scope(context))
        remove = _as_bool(kwargs.get("remove_from_queue", False))

        if remove and pid:
            await self.client.delete_queue_items([pid])
        if not pid:
            return "中断失败：当前会话没有可中断的生成任务。请传入明确的 prompt_id。"
        ok = await self.client.interrupt(prompt_id=pid or None)
        if not ok:
            return "中断失败：无法连接 ComfyUI。"
        parts = [f"已请求中断任务 {pid}。"]
        if remove:
            parts.append("该任务已从待执行队列移除。")
        parts.append("若任务已执行完毕则无需处理。")
        return "".join(parts)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiQueueTool(FunctionTool[AstrAgentContext]):
    """查询 ComfyUI 队列与 GPU 状态。"""

    name: str = "comfyui_queue"
    description: str = (
        "查询 ComfyUI 当前队列（运行中/待执行任务数）与 GPU 显存占用。"
        "用户想知道生成进度、为什么慢、显存够不够时使用。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "include_gpu": {
                    "type": "boolean",
                    "description": "是否附带 GPU/显存状态（默认 true）",
                },
            },
        }
    )
    client: ComfyUIClient | None = None

    @staticmethod
    def _gb(n: int | float) -> str:
        try:
            return f"{n / (1024 ** 3):.1f}GB"
        except (TypeError, ValueError):
            return "?"

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        queue = await self.client.get_queue()
        if queue is None:
            return "查询失败：无法连接 ComfyUI。"
        running = queue.get("queue_running") or []
        pending = queue.get("queue_pending") or []
        running_ids = [str(item[1]) for item in running if isinstance(item, (list, tuple)) and len(item) > 1]
        pending_ids = [str(item[1]) for item in pending if isinstance(item, (list, tuple)) and len(item) > 1]

        parts = [f"【ComfyUI 队列】运行中 {len(running_ids)} 个，待执行 {len(pending_ids)} 个"]
        if running_ids:
            parts.append(f"运行中: {', '.join(running_ids[:3])}")
        if pending_ids:
            parts.append(f"待执行: {', '.join(pending_ids[:5])}" + ("…" if len(pending_ids) > 5 else ""))

        if _as_bool(kwargs.get("include_gpu", True), default=True):
            stats = await self.client.get_system_stats()
            if stats:
                sysinfo = stats.get("system", {})
                devices = stats.get("devices", [])
                parts.append("")
                parts.append(f"ComfyUI 版本: {sysinfo.get('comfyui_version', '?')}")
                ram_t = self._gb(sysinfo.get("ram_total"))
                ram_f = self._gb(sysinfo.get("ram_free"))
                parts.append(f"内存: 空闲 {ram_f} / 共 {ram_t}")
                for d in devices[:2]:
                    name = d.get("name", "?")
                    vram_t = self._gb(d.get("vram_total"))
                    vram_f = self._gb(d.get("vram_free"))
                    parts.append(f"GPU {d.get('index', 0)} {name}: 显存空闲 {vram_f} / 共 {vram_t}")
            else:
                parts.append("")
                parts.append("（GPU 状态获取失败）")
        return "\n".join(parts)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiBooruTool(FunctionTool[AstrAgentContext]):
    """danbooru / gelbooru 画师/角色触发词查询。"""

    name: str = "comfyui_booru"
    description: str = (
        "从 danbooru（默认）或 gelbooru 查询画师或角色的触发词、别名和常用 tag。"
        "danbooru 查询失败或无结果时自动回退 gelbooru（结果里会标注真实来源）。"
        "用户指定画师风格/角色时，先调用本工具查到真实触发词，"
        "再把 @画师 串填进 comfyui_draw 的 artist/trigger_words 参数，不要凭记忆编 tag。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["danbooru", "gelbooru"],
                    "description": "查询源：danbooru（默认）/ gelbooru；danbooru 无结果会自动回退 gelbooru",
                },
                "type": {
                    "type": "string",
                    "enum": ["artist", "character"],
                    "description": "查询类型：artist=画师，character=角色",
                },
                "query": {
                    "type": "string",
                    "description": "画师名或角色名（中文名/罗马音/日文均可，如 初音ミク）",
                },
                "limit": {
                    "type": "number",
                    "description": "取样作品数（1-50，默认 30），越多统计越准但越慢",
                },
            },
            "required": ["type", "query"],
        }
    )
    danbooru: DanbooruClient | None = None
    gelbooru: GelbooruClient | None = None

    @staticmethod
    def _fmt_tags(tags: list[tuple[str, int]]) -> str:
        if not tags:
            return "  (无样本)"
        return "  " + ", ".join(f"{t}({n})" for t, n in tags)

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        source = str(kwargs.get("source") or "danbooru").strip().lower()
        kind = str(kwargs.get("type") or "").strip().lower()
        query = str(kwargs.get("query") or "").strip()
        if source not in ("danbooru", "gelbooru") or kind not in ("artist", "character") or not query:
            return "查询失败：需要 source（danbooru/gelbooru）、type（artist/character）与 query 参数。"
        try:
            limit = min(max(int(kwargs.get("limit") or 30), 1), 50)
        except (TypeError, ValueError):
            limit = 30

        # 主源查询失败（danbooru 镜像 403/无结果）时自动回退 gelbooru，不用让 LLM 手动重试
        primary = self.danbooru if source == "danbooru" else self.gelbooru
        fallback = self.gelbooru if source == "danbooru" else self.danbooru
        if primary is None:
            return f"查询失败：{source} 未配置。"

        data = None
        used_source = source
        if kind == "artist":
            data = await primary.search_artist(query, limit)
            if data is None and fallback is not None:
                data = await fallback.search_artist(query, limit)
                used_source = "gelbooru" if source == "danbooru" else "danbooru"
            if data is None:
                return f"{source} 未找到画师：{query}"
            parts = [f"【画师 @{data['artist']}】（来源: {used_source}）"]
            if data["aliases"]:
                parts.append(f"别名: {', '.join(data['aliases'])}")
            parts.append("常用画风 tag（作品取样统计）:")
            posts = (
                GelbooruClient.normalize_posts(data["posts"])
                if used_source == "gelbooru"
                else data["posts"]
            )
            parts.append(self._fmt_tags(DanbooruClient.aggregate_tags(posts)))
            parts.append(f"触发词建议: @{data['artist']}")
            parts.append(f"参考: {data['url']}")
            return "\n".join(parts)

        data = await primary.search_character(query, limit)
        if data is None and fallback is not None:
            data = await fallback.search_character(query, limit)
            used_source = "gelbooru" if source == "danbooru" else "danbooru"
        if data is None:
            return f"{source} 未找到角色：{query}"
        parts = [f"【角色 {data['character']}】（来源: {used_source}）"]
        if data["aliases"]:
            parts.append(f"别名: {', '.join(data['aliases'])}")
        parts.append("常用 tag（作品取样统计）:")
        posts = (
            GelbooruClient.normalize_posts(data["posts"])
            if used_source == "gelbooru"
            else data["posts"]
        )
        parts.append(self._fmt_tags(DanbooruClient.aggregate_tags(posts)))
        parts.append(f"触发词建议: {data['character']}（别名见上）")
        parts.append(f"参考: {data['url']}")
        return "\n".join(parts)





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
class ComfyuiAnimadexTool(FunctionTool[AstrAgentContext]):
    """本地 AnimaDex 角色库查询（角色/画师/系列）。"""

    name: str = "comfyui_animadex"
    description: str = (
        "从本地 AnimaDex 角色库（36,000+ 动漫游戏角色，离线 SQLite）查询角色/画师/系列的"
        "规范触发词(trigger)、特征标签与关联 LoRA。"
        "本工具是系统自带 search-characters / get-character / search-artists / "
        "search-copyrights 等 MCP 工具的本地封装，二者任选其一即可，不要重复调用。"
        "用户点名作品角色（如 忍野忍/妃咲/铃兰/初音ミク）或指定画师风格时，"
        "先调用本工具查到规范触发词，再把结果填进 comfyui_draw 的 prompt/artist 参数，"
        "不要凭记忆编 tag。搜索时优先用日文原名或英文罗马音（如 himari、初音ミク），罗马音命中率最高；中文虽可搜但自动映射不保证全中，中文查不到就换罗马音重搜。"
        "想拿角色详细设定/关联 LoRA 时，用 type=character 搜索后在结果里取 slug，"
        "再调 get_character 拉完整信息。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["character", "artist", "copyright"],
                    "description": "查询类型：character=角色（默认），artist=画师，copyright=系列/版权",
                },
                "query": {
                    "type": "string",
                    "description": "角色名/画师名/系列名。最好用罗马音或日文原名（如 himari、ヒマリ）；中文能用但映射不全",
                },
                "page": {
                    "type": "number",
                    "description": "结果页码（默认 1）",
                },
            },
            "required": ["query"],
        }
    )
    client: AnimaDexClient | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return "查询失败：query 不能为空。"
        if self.client is None:
            return "查询失败：AnimaDex 客户端未配置。"
        kind = str(kwargs.get("type") or "character").strip().lower()
        if kind not in ("character", "artist", "copyright"):
            return "查询失败：type 仅支持 character / artist / copyright。"
        try:
            page = max(int(kwargs.get("page") or 1), 1)
        except (TypeError, ValueError):
            page = 1
        if kind == "artist":
            text = await self.client.search_artists(query, page=page)
        elif kind == "copyright":
            text = await self.client.search_copyrights(query, page=page)
        else:
            text = await self.client.search_characters(query, page=page)
        if not text:
            return "查询失败：无法连接本地 AnimaDex 服务（127.0.0.1:11451）。"
        parts = [f"【AnimaDex 本地角色库】{kind}: {query}"]
        parts.append(text)
        return "\n".join(parts)

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiRunWorkflowTool(FunctionTool[AstrAgentContext]):
    """直接运行任意工作流 JSON（等价 MCP run_workflow，不依赖拉模板）。"""

    name: str = "comfyui_run_workflow"
    description: str = (
        "直接运行一个工作流 JSON（ComfyUI API 格式）并返回结果。"
        "等价于原生 MCP 的 run_workflow，但不依赖在线模板库。"
        "workflow 参数可以是：JSON 文件路径（本地或本机上已存在的路径）、"
        "或直接传 JSON 字符串（dict 格式，节点 id -> {class_type, inputs}）。"
        "用户有现成工作流 JSON / 想跑非内置模板的工作流时使用。"
        "生成的图片会自动下载到本地并发送到当前会话。"
        + _ASYNC_GENERATION_HINT
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workflow": {
                    "type": "string",
                    "description": "工作流 JSON：文件路径或 JSON 字符串（必填）",
                },
                "wait": {
                    "type": "boolean",
                    "description": "兼容参数；插件运行时统一后台排队并返回排队确认",
                },
            },
            "required": ["workflow"],
        }
    )
    client: ComfyUIClient | None = None
    output_dir: Path | None = None
    shared: dict = Field(default_factory=dict)

    @staticmethod
    def _load_wf(raw: str) -> dict | None:
        raw = (raw or "").strip()
        if not raw:
            return None
        # 先当路径试
        p = Path(raw).expanduser()
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                raise ValueError(f"读取工作流文件失败: {e}") from e
        # 再当 JSON 字符串试
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"workflow 路径与 JSON 格式均无效: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("workflow JSON 必须是对象（节点 id -> 节点）")
        return data

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        raw = str(kwargs.get("workflow") or "").strip()
        if not raw:
            return "运行失败：workflow 不能为空。"
        try:
            wf = self._load_wf(raw)
        except ValueError as e:
            return f"运行失败：{e}"
        if wf is None:
            return "运行失败：workflow 解析失败。"

        queued = await _enqueue_generation(self.shared, context, wf, {"entry": "raw_workflow"})
        if queued is not None:
            return queued
        pid, submit_err = await self.client.submit_prompt_detail(wf)
        if submit_err:
            return f"运行失败：{submit_err}"
        if not pid:
            return "运行失败：无法连接 ComfyUI。"
        _remember_prompt_id(self.shared, context, pid)

        wait = _as_bool(kwargs.get("wait", True), default=True)
        if not wait:
            return f"工作流已提交，prompt_id: {pid}（可稍后用 comfyui_job 查状态）"

        # 轮询
        while True:
            entry, wait_error = await self.client.wait_for_history(pid)
            if wait_error:
                return wait_error
            if entry is not None:
                st = entry.get("status") or {}
                if st.get("status_str") == "error":
                    return f"工作流执行失败: {execution_error_message(st, '详见 ComfyUI 日志')}"
                outputs = entry.get("outputs", {})
                images = []
                for node_out in outputs.values():
                    images.extend(node_out.get("images", []))
                if not images:
                    return f"工作流执行完成（prompt_id: {pid}），但无图片输出。"
                img = next((item for item in images if item.get("type", "output") == "output"), images[-1])
                filename = img["filename"]
                subfolder = img.get("subfolder", "")
                content = await self.client.download_image(filename, subfolder, image_type=img.get("type", "output"))
                if not content:
                    return f"工作流执行完成，图片下载失败（{filename}）。prompt_id: {pid}"
                try:
                    local_path = await asyncio.to_thread(save_image, self.output_dir, filename, content)
                except (OSError, ValueError) as e:
                    return f"工作流执行完成但本地保存失败（{e}）。prompt_id: {pid}"
                _remember_image_path(self.shared, context, local_path)
                try:
                    event: AstrMessageEvent = context.context.event
                    await event.send(MessageChain().file_image(str(local_path)))
                except Exception as e:
                    return f"工作流执行完成，图片发送失败（{e}）。本地路径: {local_path}"
                return (
                    f"工作流执行完成，图片已发送。\n"
                    f"本地路径: {local_path}\nprompt_id: {pid}"
                )
            await asyncio.sleep(2)
        return f"工作流执行超时（{int(self.client.timeout)}s）。prompt_id: {pid}"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiJobTool(FunctionTool[AstrAgentContext]):
    """按 prompt_id 查询任务状态 / 等待 / 取消（等价 MCP job）。"""

    name: str = "comfyui_job"
    description: str = (
        "按任务编号（如 C000001）或 prompt_id 查询生成任务状态或取消任务。"
        "action=status：查询状态与输出；action=wait：返回队列任务当前状态；"
        "action=cancel：中断任务；action=queue：查看本会话的生成队列。"
        "不传编号时查询本会话最近的任务。插件队列任务自动通知完成，wait 也立即返回状态，无需反复查询。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "wait", "cancel", "queue"],
                    "description": "操作：status=查状态（默认），wait=等待完成，cancel=取消，queue=看队列",
                },
                "prompt_id": {
                    "type": "string",
                    "description": "插件任务编号或 ComfyUI prompt_id，可选；不传用本会话最近任务",
                },
                "task_id": {
                    "type": "string",
                    "description": "插件任务编号，例如 C000001；与 prompt_id 二选一",
                },
            },
        }
    )
    client: ComfyUIClient | None = None
    shared: dict = Field(default_factory=dict)

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        action = str(kwargs.get("action") or "status").strip().lower()
        queue = self.shared.get("generation_queue")
        identifier = str(kwargs.get("task_id") or kwargs.get("prompt_id") or "").strip()
        if queue is not None:
            if action not in {"status", "wait", "cancel", "queue"}:
                return "操作无效：支持 status / wait / cancel / queue。"
            return await queue.action(action, identifier, _event_scope(context))
        pid = str(kwargs.get("prompt_id") or "").strip() or _last_prompt_id(
            self.shared, context
        )

        if action == "queue":
            queue = await self.client.get_queue()
            if queue is None:
                return "查询失败：无法连接 ComfyUI。"
            running = [str(item[1]) for item in queue.get("queue_running") or [] if isinstance(item, (list, tuple)) and len(item) > 1]
            pending = [str(item[1]) for item in queue.get("queue_pending") or [] if isinstance(item, (list, tuple)) and len(item) > 1]
            return f"运行中: {len(running)} | 待执行: {len(pending)}\n运行中任务: {', '.join(running) or '无'}\n待执行任务: {', '.join(pending) or '无'}"

        if not pid:
            return "查询失败：没有可用的 prompt_id（先运行一次生成，或显式传入 prompt_id）。"

        if action == "cancel":
            ok = await self.client.interrupt(prompt_id=pid)
            return "已请求取消任务 " + pid + "。" if ok else "取消失败：无法连接 ComfyUI。"

        if action == "wait":
            while True:
                entry, wait_error = await self.client.wait_for_history(pid)
                if wait_error:
                    return wait_error
                if entry is not None:
                    st = entry.get("status") or {}
                    if st.get("status_str") == "error":
                        return f"任务 {pid}: {execution_error_message(st, '详见 ComfyUI 日志')}"
                    outputs = entry.get("outputs", {})
                    n = sum(len(o.get("images", [])) for o in outputs.values())
                    return f"任务 {pid} 已完成，输出图片 {n} 张。"
                await asyncio.sleep(2)
            return f"任务 {pid} 等待超时（{int(self.client.timeout)}s）。"

        entry = await self.client.get_history_entry(pid)
        if entry is None:
            # 可能还在排队/运行中
            queue = await self.client.get_queue()
            if queue is not None:
                running = [str(item[1]) for item in queue.get("queue_running") or [] if isinstance(item, (list, tuple)) and len(item) > 1]
                pending = [str(item[1]) for item in queue.get("queue_pending") or [] if isinstance(item, (list, tuple)) and len(item) > 1]
                if pid in running:
                    return f"任务 {pid} 正在运行中。"
                if pid in pending:
                    return f"任务 {pid} 在待执行队列中。"
            return f"任务 {pid} 状态未知（可能已过期或不存在）。"
        st = entry.get("status") or {}
        if st.get("status_str") == "error":
            return f"任务 {pid}: {execution_error_message(st, '详见 ComfyUI 日志')}"
        outputs = entry.get("outputs", {})
        images = []
        for node_out in outputs.values():
            images.extend(node_out.get("images", []))
        if not images:
            return f"任务 {pid} 已完成，无图片输出。"
        f = images[0]
        return f"任务 {pid} 已完成。图片: {f.get('filename')}（subfolder: {f.get('subfolder', '')}）"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiFetchOutputsTool(FunctionTool[AstrAgentContext]):
    """下载 ComfyUI 输出文件到本地（等价 MCP fetch_outputs）。"""

    name: str = "comfyui_fetch_outputs"
    description: str = (
        "按 prompt_id 把 ComfyUI 生成的输出图片下载到本地，并返回本地路径。"
        "任务已完成但图片没收到、或想再拿一次输出时使用。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt_id": {
                    "type": "string",
                    "description": "任务 ID（必填）",
                },
            },
            "required": ["prompt_id"],
        }
    )
    client: ComfyUIClient | None = None
    output_dir: Path | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        pid = str(kwargs.get("prompt_id") or "").strip()
        if not pid:
            return "下载失败：prompt_id 不能为空。"
        entry = await self.client.get_history_entry(pid)
        if entry is None:
            return f"任务 {pid} 未找到（可能未完成或不存在）。"
        st = entry.get("status") or {}
        if st.get("status_str") == "error":
            return f"任务 {pid}: {execution_error_message(st, '详见 ComfyUI 日志')}"
        outputs = entry.get("outputs", {})
        images = []
        for node_out in outputs.values():
            images.extend(node_out.get("images", []))
        if not images:
            return f"任务 {pid} 无图片输出。"
        saved = []
        for img in images:
            filename = img["filename"]
            subfolder = img.get("subfolder", "")
            content = await self.client.download_image(filename, subfolder, image_type=img.get("type", "output"))
            if not content:
                continue
            try:
                local_path = await asyncio.to_thread(save_image, self.output_dir, filename, content)
                saved.append(str(local_path))
            except (OSError, ValueError):
                continue
        if not saved:
            return f"任务 {pid} 图片全部下载失败（链路问题？）。"
        return "已下载输出图片:\n" + "\n".join(saved)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiSystemStatsTool(FunctionTool[AstrAgentContext]):
    """查询 ComfyUI 系统/显存状态（等价 MCP system_stats）。"""

    name: str = "comfyui_system_stats"
    description: str = (
        "查询 ComfyUI 的系统状态：设备（GPU）、显存占用、系统内存。"
        "用户想知道显存占用、能不能跑大图、为什么变慢时使用。"
    )
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}})
    client: ComfyUIClient | None = None

    @staticmethod
    def _gb(n: int | float | None) -> str:
        try:
            return f"{n / (1024 ** 3):.1f}GB"
        except (TypeError, ValueError):
            return "?"

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        stats = await self.client.get_system_stats()
        if stats is None:
            return "查询失败：无法连接 ComfyUI。"
        parts = ["【ComfyUI 系统状态】"]
        sysinfo = stats.get("system", {})
        if sysinfo:
            os_ = sysinfo.get("os", "")
            parts.append(f"系统: {os_}")
            ram_total = sysinfo.get("ram_total")
            ram_free = sysinfo.get("ram_free")
            if ram_total:
                parts.append(f"内存: 总 {self._gb(ram_total)} / 空闲 {self._gb(ram_free)}")
        for dev in stats.get("devices", []):
            name = dev.get("name", "?")
            parts.append(f"设备: {name}")
            vr = dev.get("vram_total")
            vf = dev.get("vram_free")
            if vr:
                parts.append(f"显存: 总 {self._gb(vr)} / 空闲 {self._gb(vf)}")
        return "\n".join(parts)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiFreeMemoryTool(FunctionTool[AstrAgentContext]):
    """释放 ComfyUI 显存（等价 MCP free_memory）。"""

    name: str = "comfyui_free_memory"
    description: str = (
        "请求 ComfyUI 卸载模型/清空执行器缓存以释放显存。"
        "显存不足跑不动、或想腾出显存跑大图时使用。"
        "不影响正在运行的任务（下次队列迭代时生效）。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "unload_models": {
                    "type": "boolean",
                    "description": "是否卸载模型（默认 true）",
                },
            },
        }
    )
    client: ComfyUIClient | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        unload = _as_bool(kwargs.get("unload_models", True), default=True)
        ok = await self.client.free_memory(unload_models=unload, free_cache=True)
        return "已请求释放显存（卸载模型+清缓存）。" if ok else "释放失败：无法连接 ComfyUI。"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiNodesTool(FunctionTool[AstrAgentContext]):
    """查询 ComfyUI 节点类信息（等价 MCP nodes）。"""

    name: str = "comfyui_nodes"
    description: str = (
        "查询 ComfyUI 节点类（class_type）信息：action=search 按关键词搜节点类，"
        "action=get 查某个节点类的输入/输出 schema，action=list 列出全部节点类。"
        "写自定义工作流、确认某个节点类是否存在/参数名时使用。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "get", "list"],
                    "description": "search=搜索（默认），get=查单个类 schema，list=全部节点类",
                },
                "query": {
                    "type": "string",
                    "description": "搜索关键词（action=search 时必填）或节点类名（action=get 时必填）",
                },
            },
        }
    )
    client: ComfyUIClient | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        action = str(kwargs.get("action") or "search").strip().lower()
        query = str(kwargs.get("query") or "").strip()
        obj = await self.client.get_object_info()
        if not obj:
            return "查询失败：无法连接 ComfyUI。"
        if action == "list":
            names = sorted(obj.keys())
            shown = names[:MAX_LLM_LIST_ITEMS]
            suffix = f"\n（共 {len(names)} 个，仅显示前 {MAX_LLM_LIST_ITEMS} 个；用 search 精确查找）" if len(names) > len(shown) else ""
            return "全部节点类 (" + str(len(names)) + "):\n" + ", ".join(shown) + suffix
        if action == "get":
            if not query:
                return "查询失败：query（节点类名）不能为空。"
            info = obj.get(query)
            if not info:
                return f"节点类不存在: {query}"
            inp = info.get("input", {})
            req = inp.get("required", {})
            opt = inp.get("optional", {})
            parts = [f"【节点类 {query}】"]
            if req:
                parts.append("必填输入:")
                for k, v in req.items():
                    types = v[0] if isinstance(v, list) and v else "?"
                    parts.append(f"  {k}: {types}")
            if opt:
                parts.append("可选输入:")
                for k, v in opt.items():
                    types = v[0] if isinstance(v, list) and v else "?"
                    parts.append(f"  {k}: {types}")
            return "\n".join(parts)
        if not query:
            return "查询失败：query（关键词）不能为空。"
        hits = [k for k in obj if query.lower() in k.lower()]
        if not hits:
            return f"未找到包含 '{query}' 的节点类。"
        return "匹配节点类:\n" + "\n".join(sorted(hits)[:50])


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiValidateWorkflowTool(FunctionTool[AstrAgentContext]):
    """校验工作流 JSON 的节点类是否存在于 ComfyUI（等价 MCP validate_workflow 本地版）。"""

    name: str = "comfyui_validate_workflow"
    description: str = (
        "提交前校验一个工作流 JSON：检查节点 class_type 是否都存在、"
        "必填输入是否齐全。返回 valid 与错误列表。"
        "写自定义工作流、担心提交 400 时先用它检查。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workflow": {
                    "type": "string",
                    "description": "工作流 JSON：文件路径或 JSON 字符串（必填）",
                },
            },
            "required": ["workflow"],
        }
    )
    client: ComfyUIClient | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        raw = str(kwargs.get("workflow") or "").strip()
        if not raw:
            return "校验失败：workflow 不能为空。"
        try:
            wf = ComfyuiRunWorkflowTool._load_wf(raw)
        except ValueError as e:
            return f"校验失败：{e}"
        if wf is None:
            return "校验失败：workflow 解析失败。"
        obj = await self.client.get_object_info()
        if not obj:
            return "校验失败：无法连接 ComfyUI。"
        errors = []
        for nid, node in wf.items():
            ct = node.get("class_type")
            if not ct:
                errors.append(f"节点 {nid}: 缺少 class_type")
                continue
            info = obj.get(ct)
            if not info:
                errors.append(f"节点 {nid}: 节点类不存在 {ct}")
                continue
            req = info.get("input", {}).get("required", {})
            ins = node.get("inputs", {})
            for k in req:
                if k not in ins:
                    errors.append(f"节点 {nid} ({ct}): 缺少必填输入 {k}")
        if errors:
            return "校验结果: invalid\n" + "\n".join(errors[:20])
        return "校验结果: valid（全部节点类与必填输入均通过）"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiUploadFileTool(FunctionTool[AstrAgentContext]):
    """上传图片到 ComfyUI input 目录（等价 MCP upload_file，图生图素材）。"""

    name: str = "comfyui_upload_file"
    description: str = (
        "把本地图片上传到 ComfyUI 的 input 目录，返回文件名，供图生图/ControlNet 工作流引用。"
        "已有本地图片文件、需要作为生成素材时使用。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "本地图片文件路径（必填）",
                },
            },
            "required": ["path"],
        }
    )
    client: ComfyUIClient | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        path = str(kwargs.get("path") or "").strip()
        if not path:
            return "上传失败：path 不能为空。"
        p = Path(path).expanduser()
        if not p.is_file():
            return f"上传失败：文件不存在 {p}"
        try:
            content = p.read_bytes()
        except OSError as e:
            return f"上传失败：读取文件出错（{e}）"
        name, err = await self.client.upload_image(p.name, content)
        if err:
            return f"上传失败：{err}"
        return f"上传成功，ComfyUI 文件名: {name}"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiModelsSearchTool(FunctionTool[AstrAgentContext]):
    """搜索 ComfyUI 已安装的模型文件（等价 MCP search_models 本地版）。"""

    name: str = "comfyui_models_search"
    description: str = (
        "按目录列出/搜索 ComfyUI 已安装的模型文件。folder 如 checkpoints/loras/clip/vae/unet。"
        "用户想确认某模型是否已安装、文件名是什么时使用。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "模型目录：checkpoints / loras / clip / vae / unet（默认 loras）",
                },
                "query": {
                    "type": "string",
                    "description": "文件名或模型名称；支持跨家族近似搜索（可选）",
                },
                **_RESOURCE_QUERY_PROPERTIES,
            },
        }
    )
    client: ComfyUIClient | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        folder = str(kwargs.get("folder") or "loras").strip().lower()
        query = str(kwargs.get("query") or "").strip().lower()
        names = await self.client.list_models_folder(folder)
        if names is None:
            return f"查询失败：无法连接 ComfyUI 或目录 {folder} 不存在。"
        if folder in {"loras", "checkpoints", "unet", "diffusion_models"}:
            resources, _ = await self.client.list_resources()
            resources = dict(resources)
            kind = "lora" if folder == "loras" else "model"
            resources["lora_name" if kind == "lora" else "unet_name"] = names
            return _resource_page(self.client, resources, kind, query,
                                  str(kwargs.get("model_family") or ""), kwargs.get("limit", 5),
                                  kwargs.get("offset", 0), _as_bool(kwargs.get("include_unknown")))
        names = sorted(n for n in names if query in n.casefold())
        offset = page_number(kwargs.get("offset"))
        limit = max(1, page_number(kwargs.get("limit"), 5, 10))
        suffix = f"\nnext_offset={offset + limit}" if offset + limit < len(names) else ""
        return f"【{folder}】共 {len(names)} 项\n" + "\n".join(names[offset:offset + limit]) + suffix


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiRecipeTool(FunctionTool[AstrAgentContext]):
    """保存、读取和列出与工作流解耦的快捷配方。"""

    name: str = "comfyui_recipe"
    description: str = (
        "把实验好的底模、LoRA、画幅和采样参数保存为命名配方。配方引用 model_family，"
        "工作流和节点映射由家族配置统一管理。"
        "action=save（保存，需 name + model_family）/ action=list（列出）/"
        "action=load（读取一个配方，返回全部参数）/ action=delete（删除）。"
        "实际快捷生图使用 comfyui_recipe_draw，只需传配方名和本次 prompt。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "list", "load", "delete"],
                    "description": "save=保存（默认），list=列出，load=读取，delete=删除",
                },
                "name": {
                    "type": "string",
                    "description": "配方名（save/load/delete 必填）",
                },
                "description": {"type": "string", "description": "配方用途说明，帮助模型判断何时复用此配方"},
                "model_family": {
                    "type": "string",
                    "description": "配方适用的模型家族（save 时必填）",
                },
                "artist": {"type": "string", "description": "画师串"},
                "quality": {"type": "string", "description": "质量词"},
                "trigger_words": {"type": "string", "description": "lora触发词"},
                "negative_prompt": {"type": "string", "description": "负向提示词"},
                "model": {"type": "string", "description": "底模文件名"},
                "lora": _lora_array_schema("LoRA 对象数组，省略沿用已有设置；旧版 JSON 字符串仍可解析"),
                "steps": {"type": "number", "description": "采样步数"},
                "cfg": {"type": "number", "description": "CFG"},
                "sampler_name": {"type": "string", "description": "采样器"},
                "scheduler": {"type": "string", "description": "调度器"},
                "denoise": {"type": "number", "description": "降噪强度"},
                "width": {"type": "number", "description": "宽度"},
                "height": {"type": "number", "description": "高度"},
                "seed": {"type": "number", "description": "种子"},
            },
        }
    )
    store: RecipeStore | None = None
    allow_delete: bool = False
    builder: WorkflowBuilder | None = None
    default_workflow: str = ""
    families: ModelFamilyRegistry | None = None

    def refresh_schema(self) -> None:
        actions = ["save", "list", "load"] + (["delete"] if self.allow_delete else [])
        action = self.parameters["properties"]["action"]
        action["enum"] = actions
        action["description"] = "选择保存、列出或读取配方" + ("；仅用户明确要求时删除" if self.allow_delete else "")
        self.parameters["required"] = ["action"]
        self.description = (
            "保存、列出和读取静态配方。保存时提供 name、model_family 和可选生成参数；"
            "description 描述配方用途，不会固定后续生图提示词。生图调用 comfyui_recipe_draw。"
            + ("用户明确要求删除时可使用 action=delete。" if self.allow_delete else "")
        )
        family = self.parameters["properties"]["model_family"]
        names = self.families.names() if self.families is not None else []
        if names:
            family["enum"] = names
        else:
            family.pop("enum", None)

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        action = str(kwargs.get("action") or "save").strip().lower()
        name = str(kwargs.get("name") or "").strip()
        if action not in ("save", "list", "load", "delete"):
            return "操作失败：action 仅支持 save/list/load/delete。"

        if action == "list":
            names = _recipe_enum(self.store)
            if not names:
                return "没有已保存的配方。"
            parts = ["【已保存配方】"]
            if self.store is not None:
                for row in self.store.list():
                    desc = (row.get("description") or "")[:40]
                    parts.append(f"- {row.get('name') or row['id']}: {desc}")
            return "\n".join(parts)

        if not name:
            return "操作失败：name 不能为空。"

        if action == "delete":
            if not self.allow_delete:
                return "为避免模型误删配方，删除操作请在 Workflow Studio 中手动完成。"
            if self.store is not None and self.store.delete(name):
                return f"已删除配方: {name}"
            return f"配方不存在: {name}"

        if action == "load":
            recipe = self.store.get(name) if self.store else None
            if recipe is None:
                return f"配方不存在: {name}"
            defaults = recipe.get("defaults") or {}
            info = {
                "name": recipe.get("name") or name,
                "description": recipe.get("description") or "",
                "model_family": recipe.get("family") or recipe_family(recipe),
                "model": defaults.get("model") or "",
                "lora": defaults.get("loras") or [],
                "width": defaults.get("width") or "",
                "height": defaults.get("height") or "",
                "steps": defaults.get("steps") or "",
                "cfg": defaults.get("cfg") or "",
                "sampler_name": defaults.get("sampler_name") or "",
                "scheduler": defaults.get("scheduler") or "",
                "denoise": defaults.get("denoise") or "",
                "trigger_words": defaults.get("trigger_words") or "",
                "artist": defaults.get("artist") or "",
                "quality": defaults.get("quality") or "",
                "negative_prompt": defaults.get("negative") or "",
            }
            return "配方参数:\n" + json.dumps(info, ensure_ascii=False, indent=2)

        # save
        family_name = str(kwargs.get("model_family") or "").strip()
        if not family_name:
            return "保存失败：model_family 不能为空。"
        family = self.families.get(family_name) if self.families is not None else None
        if family is None:
            available = "、".join(self.families.names()) if self.families else "无"
            return f"保存失败：模型家族「{family_name}」不存在。可用家族：{available}"
        lora_val = kwargs.get("lora")
        if lora_val:
            try:
                parsed_loras = parse_lora(lora_val)
            except ValueError as e:
                return f"保存失败：{e}"
        else:
            parsed_loras = []
        defaults: dict[str, Any] = {}
        for key in ("artist", "quality", "model", "sampler_name", "scheduler"):
            v = kwargs.get(key)
            if v is not None and v != "":
                defaults[key] = v
        if kwargs.get("negative_prompt"):
            defaults["negative"] = kwargs["negative_prompt"]
        if lora_val not in (None, ""):
            defaults["loras"] = parsed_loras
        for key in ("steps", "cfg", "denoise", "width", "height"):
            v = kwargs.get(key)
            if v is not None:
                defaults[key] = v

        try:
            assert self.store is not None
            self.store.save({
                "name": name,
                "description": str(kwargs.get("description") or kwargs.get("prompt") or "")[:160],
                "family": family.name,
                "defaults": defaults,
            })
        except (ValueError, OSError) as e:
            return f"保存失败：{e}"
        return f"配方已保存: {name}（模型家族 {family.name}）"


_DRAW_DESC = (
    "从文字生成新图片并发送；需要修改现有图片时使用 comfyui_edit。model_family、prompt 必填，提示词遵循家族 prompt_style。"
    "省略可选参数沿用工作流。可按画风、角色、服饰或效果需求主动用 comfyui_lookup 查询并选用 LoRA；"
    "分辨率选择器工作流可用 aspect_ratio 和 megapixels 覆盖比例与目标百万像素数；"
    "按用途、类别或标签查询资源时传 model_family；完整已知文件名可跨家族查找。使用前核对资源与当前家族兼容，并采用返回的文件名、推荐权重和已知触发词。"
    "复用配方用 comfyui_recipe_draw。"
    + _ASYNC_GENERATION_HINT
)


def _recipe_enum(store: RecipeStore | None) -> list[str]:
    if store is None:
        return []
    try:
        return store.names()
    except Exception:
        return []


def _lora_array_schema(description: str) -> dict[str, Any]:
    """Expose structured LoRA arguments while the runtime keeps legacy parsing."""
    return {
        "type": "array",
        "description": description,
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "comfyui_lookup 返回的实际文件名"},
                "strength": {"type": "number", "description": "查询结果给出的推荐权重；省略则使用工作流默认权重"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    }


def _match_resource(names: list[str], query: str, limit: int = 8) -> list[str]:
    q = str(query or "").strip().casefold().replace("\\", "/")
    if not q:
        return []
    exact = exact_matches(names, q)
    if exact:
        return exact[:limit]
    return [n for n in names if q in n.casefold().replace("\\", "/")][:limit]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiDrawTool(FunctionTool[AstrAgentContext]):
    """按配置的模型家族选择工作流，自由覆盖本次生成参数。"""

    name: str = "comfyui_draw"
    description: str = _DRAW_DESC
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "要画的内容，必填。根据所选模型家族的 prompt_style 组织提示词",
                },
                "model_family": {
                    "type": "string",
                    "description": "模型家族，必填。只能填写配置中公开的家族名；插件据此选择对应工作流",
                },
                "model": {
                    "type": "string",
                    "description": "换底模。用户没点名就不要填。关键词或文件名",
                },
                "lora": {
                    **_lora_array_schema("按实际需求用 comfyui_lookup 查询并选用 LoRA，传对象数组，可附推荐 strength。按工作流映射顺序覆盖可选槽位，独立加速 LoRA 保留。省略沿用工作流，空数组关闭可选 LoRA；旧版字符串调用仍兼容"),
                },
                "size": {
                    "type": "string",
                    "enum": ["portrait", "landscape", "square", "same"],
                    "description": "portrait竖图 landscape横图 square方图。用户没提画幅就不要填",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "分辨率选择器支持时设置画幅比例，例如 1:1、16:9；用户没指定时沿用工作流",
                },
                "megapixels": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 64,
                    "description": "分辨率选择器支持时设置目标 MP，例如 1.0 MP；Qwen Image 2.1 原生 2K 方图约 4.0 MP。用户没指定时沿用工作流",
                },
                "artist": {
                    "type": "string",
                    "description": "画师风格。用户没点名画师就不要填",
                },
                "quality": {
                    "type": "string",
                    "description": "画质词。用户没要求画质就不要填",
                },
                "negative": {
                    "type": "string",
                    "description": "不要出现的东西。用户没说就不要填",
                },
                "trigger_words": {
                    "type": "string",
                    "description": "所选 LoRA 的已知触发词，从 lookup/model_info 返回值或用户提供的信息中取用，保留原始格式，多个用逗号分隔。选用 LoRA 时可同步填写，无需用户另外提出；查询未提供时可省略并继续使用 LoRA",
                },
                "steps": {
                    "type": "number",
                    "description": "采样步数。用户说更精细/更快/改步数才填",
                },
                "cfg": {
                    "type": "number",
                    "description": "CFG。用户明确说改才填",
                },
                "width": {
                    "type": "number",
                    "description": "精确宽度；用户指定像素尺寸时填写",
                },
                "height": {
                    "type": "number",
                    "description": "精确高度；用户指定像素尺寸时填写",
                },
                "sampler_name": {
                    "type": "string",
                    "description": "采样器；仅在用户指定或明确需要调整时填写",
                },
                "scheduler": {
                    "type": "string",
                    "description": "调度器；仅在用户指定或明确需要调整时填写",
                },
                "denoise": {
                    "type": "number",
                    "description": "降噪强度 0 到 1；仅在需要调整时填写",
                },
                "seed": {
                    "type": "number",
                    "description": "种子。用户要复现某张图才填，否则不填",
                },
                "save_as": {
                    "type": "string",
                    "description": "用户说记住这套/存成某某时，填新配方名",
                },
            },
            "required": ["model_family", "prompt"],
        }
    )
    client: ComfyUIClient | None = None
    builder: WorkflowBuilder | None = None
    store: RecipeStore | None = None
    output_dir: Path | None = None
    shared: dict = Field(default_factory=dict)
    auto_trigger_words: bool = False
    families: ModelFamilyRegistry | None = None
    profiles: WorkflowProfileStore | None = None
    on_schema_change: Any = None

    def refresh_schema(self) -> None:
        names = self.families.names() if self.families is not None else []
        catalog = self.families.catalog() if self.families is not None else ""
        self.description = _DRAW_DESC + (f" 可用家族：{catalog}" if catalog else " 当前没有可用模型家族配置。")
        props = self.parameters.setdefault("properties", {})
        family_prop = props.setdefault("model_family", {"type": "string"})
        if names:
            family_prop["enum"] = names
            family_prop["description"] = "必填，从 enum 选择模型家族"
        else:
            family_prop.pop("enum", None)
            family_prop["description"] = "必填。请先由管理员在插件配置中添加模型家族"

    async def _resource_lists(self) -> dict:
        if self.client is None:
            return {"unet_name": [], "lora_name": []}
        return await self.client.generation_resources()


    def _family_candidates(self, resources, kind, query, family):
        names = resources.get("lora_name" if kind == "lora" else "unet_name") or []
        meta = metadata_for(resources, kind)
        rules = getattr(self.client, "resource_family_rules", [])
        if not family:
            return names
        # An exact filename remains usable when metadata is absent; never treat
        # an unclassified fuzzy match as a recommendation.
        exact = exact_matches(names, query)
        if exact:
            return [n for n in exact if not selection_error(
                resources, {"loras": [{"name": n}]} if kind == "lora" else {"model": n}, family, rules
            )]
        return filter_family(names, meta, family, rules, kind)

    async def _resolve_model(self, query: str, family: str = "", resources: dict | None = None) -> tuple[str | None, str | None]:
        resources = resources if resources is not None else await self._resource_lists()
        names = self._family_candidates(resources, "model", query, family)
        hits = _match_resource(names, query)
        if len(hits) == 1:
            return hits[0], None
        if not hits:
            return None, f"家族 {family} 中未找到可用底模「{query}」。请用 comfyui_lookup(type=model, model_family=家族) 查询；未知资源需核实后填写完整文件名。"
        return None, f"底模名称有歧义：{'、'.join(hits[:6])}。请填写含目录的完整文件名。"


    async def _resolve_loras(self, raw: Any, family: str = "", resources: dict | None = None) -> tuple[list[dict] | None, str | None]:
        resources = resources if resources is not None else await self._resource_lists()
        metadata = resources.get("lora_meta") or {}
        try:
            parsed = parse_lora(raw)
        except ValueError as error:
            if not isinstance(raw, str):
                return None, f"LoRA 参数无效：{error}"
            parsed = [{"name": part.strip()} for part in str(raw).split(",") if part.strip()]
        if not parsed:
            return [], None
        resolved: list[dict] = []
        for item in parsed:
            query = str(item.get("name") or "").strip()
            if not query:
                continue
            names = self._family_candidates(resources, "lora", query, family)
            hits = exact_matches(names, query) or _match_lora_resources(names, metadata, query)
            if len(hits) == 1:
                resolved.append({**item, "name": hits[0]})
                continue
            if not hits:
                return None, f"家族 {family} 中未找到可用 LoRA「{query}」。请按 model_family 查询；未知资源需核实后填写完整文件名。"
            return None, f"LoRA 名称有歧义：{'、'.join(hits[:6])}。请填写含目录的完整文件名。"
        return resolved, None

    async def _execute(
        self,
        context: ContextWrapper[AstrAgentContext],
        *,
        prompt: str,
        values: dict[str, Any],
        family: ModelFamily | None,
        recipe: dict | None = None,
        size_token: str = "",
        save_as: str = "",
        legacy_workflow: str = "",
        legacy_slots: dict | None = None,
        legacy_drop_nodes: list[str] | None = None,
        resources: dict | None = None,
    ) -> str:
        """执行一条已解析的家族或旧配方生成任务。"""
        if (
            self.builder is None
            or self.client is None
            or self.store is None
            or self.output_dir is None
        ):
            return "生成失败：插件未初始化完成。"

        if values.get("model") or values.get("loras"):
            resources = resources if resources is not None else await self._resource_lists()
            selected = [str(item.get("name")) for item in parse_lora(values.get("loras")) if item.get("name")]
            if selected:
                resources = await self.client.selected_lora_metadata(resources, selected, require_trusted=self.auto_trigger_words and values.get("trigger_words") is None)
        resources = resources or {}
        if family is not None and (values.get("model") or values.get("loras")):
            error = selection_error(resources, values, family.name,
                                    getattr(self.client, "resource_family_rules", []))
            if error:
                return error

        workflow_name = family.workflow if family is not None else legacy_workflow
        if not workflow_name:
            return "生成失败：没有可用工作流。请先在模型家族配置中选择工作流。"
        try:
            wf = self.builder.load_template(workflow_name)
        except FileNotFoundError as e:
            return f"生成失败：{e}"

        if family is not None and self.profiles is not None:
            profile = self.profiles.effective(workflow_name, wf, getattr(self.client, "_object_info", None))
            slots = profile.get("slots") or {}
            drop_nodes = list(profile.get("drop_nodes") or [])
        elif legacy_slots:
            slots = legacy_slots
            drop_nodes = list(legacy_drop_nodes or [])
        else:
            slots = detect_slots(wf, getattr(self.client, "_object_info", None))
            drop_nodes = list(ANIMA_DROP_NODES) if looks_like_anima(wf) else []

        if not slots.get("prompt"):
            return (
                f"工作流「{workflow_name}」还没指定主提示词节点。"
                "请在 Workflow Studio 工作流页确认这张工作流的槽位映射。"
            )
        if "model" in values and values.get("model") not in (None, "") and not slots.get("model"):
            return f"工作流「{workflow_name}」没有映射底模槽位，无法替换底模。"
        if "loras" in values and values.get("loras") is not None and not slots.get("loras"):
            return f"工作流「{workflow_name}」没有映射 LoRA 槽位，无法写入 LoRA。"
        if any(values.get(k) not in (None, "") for k in ("steps", "cfg", "sampler_name", "scheduler", "denoise")):
            if not slots.get("sampler") and not slots.get("sampler_2"):
                return f"工作流「{workflow_name}」没有映射采样槽位，无法写入采样参数。"
        size_token = str(size_token or "").strip().lower()
        change_size = bool(size_token and size_token != "same")
        if (
            change_size
            or values.get("width") not in (None, "")
            or values.get("height") not in (None, "")
        ) and not slots.get("size"):
            return f"工作流「{workflow_name}」没有映射画面大小槽位，无法修改画幅。"
        for role in ("aspect_ratio", "megapixels"):
            if values.get(role) not in (None, "") and not slots.get(role):
                return f"工作流「{workflow_name}」没有映射 {role} 输入，无法修改分辨率选择器。"

        seed = values.get("seed")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        values = dict(values)
        if "trigger_words" in (recipe or {}).get("defaults", {}):
            values.setdefault("trigger_words", recipe["defaults"]["trigger_words"])
        values = fill_trigger_words(values, resources, self.auto_trigger_words and bool(slots.get("trigger_words")))
        values["prompt"] = prompt
        values["seed"] = seed
        try:
            values = _validate_generation_values(values)
            seed = values["seed"]
            if change_size:
                current = read_current_values(wf, slots)
                width = values.get("width") or current.get("width")
                height = values.get("height") or current.get("height")
                values["width"], values["height"] = resolve_size(
                    int(width) if width else None,
                    int(height) if height else None,
                    size_token,
                    presets=(recipe or {}).get("size_presets"),
                )
                values = _validate_generation_values(values)
            apply_slots(
                wf,
                slots,
                values,
                drop_nodes=drop_nodes,
            )
        except (TypeError, ValueError) as e:
            return f"生成失败：参数错误（{e}）"

        if self.shared.get("generation_queue") is not None:
            used = read_current_values(wf, slots)
            used.update({k: v for k, v in values.items() if v is not None and (v != "" or k == "trigger_words")})
            family_name = family.name if family is not None else str((recipe or {}).get("family") or "")
            recipe_defaults = {
                k: v for k, v in used.items()
                if k not in {"prompt", "seed"} and v is not None
                and (v != "" or k == "trigger_words") and (v != [] or k == "loras")
            }
            saved_recipe = {
                "name": save_as, "description": f"从 {family_name} 家族自由生图保存",
                "family": family_name, "defaults": recipe_defaults,
            } if save_as and family is not None else None
            return await _enqueue_generation(self.shared, context, wf, {
                "entry": "recipe" if recipe else "family", "family": family_name,
                "recipe": (recipe or {}).get("name"), "recipe_id": (recipe or {}).get("id"),
                "workflow": workflow_name, "slots": slots, "drop_nodes": drop_nodes,
                "prompt": prompt, "values": {**recipe_defaults, "seed": seed},
            }, saved_recipe)
        pid, submit_err = await self.client.submit_prompt_detail(wf)
        if submit_err:
            return f"生成失败：{submit_err}"
        if not pid:
            return "生成失败：无法连接 ComfyUI。"
        _remember_prompt_id(self.shared, context, pid)

        outputs, wait_err = await _wait_outputs(self.client, pid)
        if wait_err:
            return wait_err if wait_err.startswith(("等待超时", "等待中断")) else f"生成失败：{wait_err}"
        if outputs is None:
            return "生成失败：未获取到执行结果。"

        images: list[dict] = []
        for node_out in outputs.values():
            images.extend(node_out.get("images", []))
        if not images:
            return "生成完成，但没有输出图片。"
        img = next((item for item in images if item.get("type") == "output"), images[-1])
        filename = str(img.get("filename") or "")
        if not filename:
            return "生成完成，但没有有效图片。"
        content = await self.client.download_image(
            filename,
            subfolder=img.get("subfolder", ""),
            image_type=img.get("type", "output"),
        )
        if not content:
            return f"图片已生成但下载失败（{filename}）。"
        try:
            local_path = await asyncio.to_thread(save_image, self.output_dir, filename, content)
        except (OSError, ValueError) as e:
            return f"图片已生成但本地保存失败（{e}）。"
        _remember_image_path(self.shared, context, local_path)

        actual = read_current_values(wf, slots)
        used = dict(actual)
        used.update({k: v for k, v in values.items() if v is not None and (v != "" or k == "trigger_words")})
        used["prompt"] = prompt
        used["seed"] = seed
        family_name = family.name if family is not None else str((recipe or {}).get("family") or "")
        self.store.save_history(
            {
                "prompt_id": pid,
                "entry": "recipe" if recipe else "family",
                "family": family_name,
                "recipe": (recipe or {}).get("name"),
                "recipe_id": (recipe or {}).get("id"),
                "workflow": workflow_name,
                "slots": slots,
                "drop_nodes": drop_nodes,
                "prompt": prompt,
                "values": {
                    k: v
                    for k, v in used.items()
                    if k != "prompt"
                    and v is not None and (v != "" or k == "trigger_words")
                    and (v != [] or k == "loras")
                },
                "filename": filename,
                "local_path": str(local_path),
            }
        )

        saved_note = ""
        if save_as:
            if family is None:
                saved_note = " 当前是旧配方兼容路径，未另存新配方。"
            else:
                recipe_defaults = {
                    k: v
                    for k, v in used.items()
                    if k not in {"prompt", "seed"}
                    and v is not None and (v != "" or k == "trigger_words")
                    and (v != [] or k == "loras")
                }
                try:
                    self.store.save(
                        {
                            "name": save_as,
                            "description": f"从 {family.name} 家族自由生图保存",
                            "family": family.name,
                            "defaults": recipe_defaults,
                        }
                    )
                    if callable(self.on_schema_change):
                        self.on_schema_change()
                    saved_note = f" 已保存快捷配方「{save_as}」。"
                except (OSError, ValueError) as e:
                    saved_note = f" 保存配方失败：{e}"

        try:
            event: AstrMessageEvent = context.context.event
            await event.send(MessageChain().file_image(str(local_path)))
        except Exception as e:
            logger.error(f"[ComfyUIDirect] 图片发送失败: {e}")
            return f"图片已生成但发送失败（{e}）。路径: {local_path}"

        return f"图片已发送。本地路径: {local_path}\nseed={seed} prompt_id={pid}.{saved_note}"

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            return "生成失败：prompt 不能为空。"
        if self.families is None:
            return "生成失败：模型家族配置未初始化。"

        requested_family = str(kwargs.get("model_family") or "").strip()
        if not requested_family:
            available = "、".join(self.families.names()) or "无"
            return f"生成失败：model_family 必填。可用家族：{available}"
        family = self.families.get(requested_family)
        if family is None:
            available = "、".join(self.families.names()) or "无"
            return f"生成失败：模型家族「{requested_family}」不存在。可用家族：{available}"

        values: dict[str, Any] = {
            "prompt": prompt,
            "seed": kwargs.get("seed"),
        }
        model_raw = str(kwargs.get("model") or "").strip()
        resources = await self._resource_lists() if model_raw or _has_lora_input(kwargs) else None
        if model_raw:
            resolved_model, err = await self._resolve_model(model_raw, family.name, resources)
            if err:
                return err
            values["model"] = resolved_model
        lora_present = (
            kwargs.get("lora") not in (None, "")
            or kwargs.get("loras") not in (None, "")
        )
        if lora_present:
            raw_loras = (
                kwargs.get("lora")
                if kwargs.get("lora") not in (None, "")
                else kwargs.get("loras")
            )
            resolved_loras, err = await self._resolve_loras(raw_loras, family.name, resources)
            if err:
                return err
            values["loras"] = resolved_loras

        for source, target in (
            ("artist", "artist"),
            ("quality", "quality"),
            ("trigger_words", "trigger_words"),
            ("negative", "negative"),
            ("negative_prompt", "negative"),
            ("width", "width"),
            ("height", "height"),
            ("aspect_ratio", "aspect_ratio"),
            ("megapixels", "megapixels"),
            ("steps", "steps"),
            ("cfg", "cfg"),
            ("sampler_name", "sampler_name"),
            ("scheduler", "scheduler"),
            ("denoise", "denoise"),
        ):
            if kwargs.get(source) not in (None, ""):
                values[target] = kwargs[source]

        if "trigger_words" in kwargs and kwargs["trigger_words"] is not None:
            values["trigger_words"] = kwargs["trigger_words"]
        return await self._execute(
            context,
            prompt=prompt,
            values=values,
            family=family,
            size_token=str(kwargs.get("size") or "").strip(),
            save_as=str(kwargs.get("save_as") or "").strip(),
            resources=resources,
        )


_EDIT_DESC = (
    "按独立编辑工作流路由修改已有图片并直接发送结果。会按工作流输入顺序使用当前消息或引用消息中的多张图片；附件超过输入口时用 image_indices 选择；"
    "没有附图时可使用本插件上次生成的图片，或填写本插件此前回执的本地路径。"
    "局部改动默认沿用参考图原始宽高。需要按参考图比例缩放时使用 resolution；需要新画布/抠出素材时使用 custom_size 和 width/height。"
    "这些画布参数只写入当前工作流已映射的输入。"
    + _ASYNC_GENERATION_HINT
)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiEditTool(FunctionTool[AstrAgentContext]):
    """Upload an attached image and run a selected independent edit workflow."""

    name: str = "comfyui_edit"
    description: str = _EDIT_DESC
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "edit_workflow": {
                    "type": "string",
                    "description": "独立的编辑工作流路由名，与生图模型家族无关；只有一个路由时可省略",
                },
                "prompt": {
                    "type": "string",
                    "description": "对来源图片的修改要求，必填",
                },
                "resolution": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 8192,
                    "description": "按来源图比例缩放的最长边像素数；局部修改默认行为等同于 0（原始宽高）。只有明确要缩放时填写；不能与 width/height 同时填写",
                },
                "custom_size": {
                    "type": "boolean",
                    "description": "新画布或抠出素材时开启；工作流已映射 custom_size 开关时会启用分辨率选择器画布。width/height 可以自动开启此模式",
                },
                "width": {
                    "type": "integer",
                    "minimum": 64,
                    "maximum": 8192,
                    "description": "自定义画布宽度；需有 size 槽位。只填写一边时按参考图比例计算另一边；与 resolution 互斥。",
                },
                "height": {
                    "type": "integer",
                    "minimum": 64,
                    "maximum": 8192,
                    "description": "自定义画布高度；需有 size 槽位。只填写一边时按参考图比例计算另一边；与 resolution 互斥。",
                },
                "image_path": {
                    "type": "string",
                    "description": "可选单张本地图路径；支持本插件回执路径或 AstrBot data/temp 中的消息图片路径。附图时省略",
                },
                "image_index": {
                    "type": "integer",
                    "description": "单图工作流或只选一张图时，选择第几张（从 1 开始）；多图工作流可用 image_indices",
                },
                "image_indices": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "description": "多图工作流中按输入顺序选择多张消息图片的 1-based 序号，例如 [1,2]；省略时按消息顺序使用附件",
                },
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "多图工作流使用本插件回执路径或 AstrBot data/temp 消息图片路径数组，最多与工作流参考图输入数相同",
                },
            },
            "required": ["prompt"],
        }
    )
    client: ComfyUIClient | None = None
    builder: WorkflowBuilder | None = None
    store: RecipeStore | None = None
    output_dir: Path | None = None
    shared: dict = Field(default_factory=dict)
    families: ModelFamilyRegistry | None = None
    edit_workflows: EditWorkflowRegistry | None = None
    profiles: WorkflowProfileStore | None = None

    def refresh_schema(self) -> None:
        names = (
            self.edit_workflows.names()
            if self.edit_workflows is not None
            else (self.families.editable_names() if self.families else [])
        )
        prop = self.parameters["properties"]["edit_workflow"]
        if names:
            prop["enum"] = names
            self.parameters["required"] = ["prompt"] + (
                ["edit_workflow"] if len(names) > 1 else []
            )
            descriptions = []
            if self.edit_workflows is not None:
                descriptions = [
                    f"{row['name']}：{row['description']}"
                    for row in self.edit_workflows.list()
                    if row.get("workflow") and row.get("description")
                ]
            suffix = " 路由说明：" + "；".join(descriptions) if descriptions else ""
            self.description = _EDIT_DESC + " 可用编辑路由：" + "、".join(names) + "。" + suffix
        else:
            prop.pop("enum", None)
            self.parameters["required"] = ["prompt"]
            self.description = _EDIT_DESC + " 当前尚未配置编辑工作流。"

    async def _source_paths(
        self,
        context: ContextWrapper[AstrAgentContext],
        kwargs: dict,
        *,
        max_inputs: int,
    ) -> tuple[list[Path] | None, str | None]:
        """Resolve one or more attached images in workflow input order."""
        requested_many = kwargs.get("image_paths")
        requested_single = str(kwargs.get("image_path") or "").strip()
        if requested_many not in (None, "") and requested_single:
            return None, "image_path 与 image_paths 只能选一个。"
        if (requested_many not in (None, "") or requested_single) and (
            kwargs.get("image_indices") not in (None, "")
            or kwargs.get("image_index") not in (None, "")
        ):
            return None, "image_path/image_paths 与 image_index/image_indices 只能选一类输入。"

        raw_paths: list[str] = []
        if requested_many not in (None, ""):
            if not isinstance(requested_many, list) or not all(isinstance(x, str) for x in requested_many):
                return None, "image_paths 必须是本插件此前回执的本地路径数组。"
            raw_paths = [path.strip() for path in requested_many if path.strip()]
        elif requested_single:
            raw_paths = [requested_single]
        if requested_many is not None and not raw_paths:
            return None, "image_paths 不能为空，请提供明确图片路径。"

        def resolve_allowed_paths(values: list[str]) -> tuple[list[Path] | None, str | None]:
            resolved: list[Path] = []
            roots = [self.output_dir.resolve()]
            try:
                from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

                roots.append(Path(get_astrbot_temp_path()).resolve())
            except (ImportError, OSError, RuntimeError):
                pass
            for value in values:
                try:
                    path = Path(value).expanduser().resolve(strict=True)
                except (OSError, ValueError):
                    return None, "image_path/image_paths 无法读取；请确认路径有效。"
                allowed = False
                for root in roots:
                    try:
                        path.relative_to(root)
                        allowed = True
                        break
                    except ValueError:
                        continue
                if not allowed:
                    return None, "image_path/image_paths 仅支持本插件输出目录或 AstrBot data/temp 中的文件。"
                if not path.is_file():
                    return None, "来源图片文件不存在。"
                resolved.append(path)
            if len(resolved) > max_inputs:
                return None, f"工作流有 {max_inputs} 个参考图输入，image_paths 不能超过该数量。"
            return (resolved or None), (None if resolved else "没有提供有效的来源图片路径。")

        if raw_paths:
            return resolve_allowed_paths(raw_paths)

        images = _message_images(context)
        if images:
            raw_indices = kwargs.get("image_indices")
            raw_index = kwargs.get("image_index")
            if raw_indices not in (None, "") and raw_index not in (None, ""):
                return None, "image_index 与 image_indices 只能选一个。"
            indices: list[int]
            if raw_indices not in (None, ""):
                if not isinstance(raw_indices, list) or not raw_indices:
                    return None, "image_indices 必须是非空的 1-based 图片序号数组。"
                try:
                    if any(isinstance(index, bool) for index in raw_indices):
                        raise ValueError("boolean index")
                    indices = [int(_number(index, "image_indices", integer=True, minimum=1)) for index in raw_indices]
                except (TypeError, ValueError):
                    return None, "image_indices 必须只包含整数。"
            elif raw_index not in (None, ""):
                try:
                    if isinstance(raw_index, bool):
                        raise ValueError("boolean index")
                    indices = [int(_number(raw_index, "image_index", integer=True, minimum=1))]
                except (TypeError, ValueError):
                    return None, "image_index 必须是从 1 开始的整数。"
            else:
                if len(images) > max_inputs:
                    return None, f"当前消息有 {len(images)} 张图片，但工作流只有 {max_inputs} 个参考图输入；请用 image_index 或 image_indices 选择。"
                indices = list(range(1, len(images) + 1))

            if len(indices) > max_inputs:
                return None, f"image_indices 不能超过工作流的 {max_inputs} 个参考图输入。"
            if any(index < 1 or index > len(images) for index in indices):
                return None, f"image_indices 超出范围；当前消息有 {len(images)} 张图片。"

            paths: list[Path] = []
            for index in indices:
                try:
                    paths.append(Path(await images[index - 1].convert_to_file_path()).resolve(strict=True))
                except Exception as e:  # noqa: BLE001 - media resolver may raise adapter errors
                    return None, f"无法读取第 {index} 张消息图片（{e}）。"
            return paths, None

        if kwargs.get("image_index") is not None or kwargs.get("image_indices") is not None:
            return None, "当前消息及引用消息没有可读取图片，image_index/image_indices 无法选择；请使用 image_path/image_paths。"
        last = str((self.shared.get("last_image_paths") or {}).get(_event_scope(context)) or "")
        if last and Path(last).is_file():
            return [Path(last)], None
        return None, "当前消息没有图片，且本会话没有可用的上次生成图片；请附图后重试。"

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            return "编辑失败：prompt 不能为空。"
        if not all((self.client, self.builder, self.store, self.output_dir)):
            return "编辑失败：插件未初始化完成。"
        workflows = self.edit_workflows
        if workflows is None:
            legacy = [
                {"name": row.get("name"), "workflow": row.get("edit_workflow")}
                for row in (self.families.list() if self.families else [])
                if row.get("edit_workflow")
            ]
            workflows = EditWorkflowRegistry(raw=legacy)
        route_name = str(
            kwargs.get("edit_workflow") or kwargs.get("model_family") or ""
        ).strip()
        if not route_name:
            available = workflows.names()
            if len(available) == 1:
                route_name = available[0]
            elif len(available) > 1:
                return "编辑失败：请从可用 edit_workflow 中选择一个编辑路由：" + "、".join(available)
        edit_route = workflows.get(route_name)
        if edit_route is None or not edit_route.workflow:
            return f"编辑失败：编辑路由「{route_name or '(未指定)'}」没有可用工作流；请在 Workflow Studio 中绑定编辑工作流。"
        try:
            wf = self.builder.load_template(edit_route.workflow)
        except (FileNotFoundError, ValueError) as e:
            return f"编辑失败：{e}"
        profile = self.profiles.effective(edit_route.workflow, wf) if self.profiles else {"slots": detect_slots(wf), "drop_nodes": []}
        slots = profile.get("slots") or {}
        try:
            source_specs = source_image_slots(wf, slots)
        except ValueError as e:
            return f"编辑失败：{e}。"
        if not source_specs:
            return f"编辑失败：工作流「{edit_route.workflow}」缺少有效的来源图片 LoadImage 槽位映射。"
        if not slots.get("prompt"):
            return f"编辑失败：工作流「{edit_route.workflow}」缺少提示词槽位映射。"
        size_node = wf.get(str((slots.get("size") or {}).get("node") or ""))
        size_inputs = (size_node or {}).get("inputs") or {}
        size_has_dimensions = (
            isinstance(size_node, dict)
            and "width" in size_inputs
            and "height" in size_inputs
        )

        resolution = kwargs.get("resolution")
        resolution_provided = resolution not in (None, "")
        if resolution_provided:
            if isinstance(resolution, bool):
                return "编辑失败：resolution 必须是 0 到 8192 之间的整数。"
            try:
                resolution = _number(
                    resolution, "resolution", integer=True, minimum=0, maximum=8192,
                )
            except ValueError as e:
                return f"编辑失败：{e}。"
        width, height = kwargs.get("width"), kwargs.get("height")
        width_provided = width not in (None, "")
        height_provided = height not in (None, "")
        width = width if width_provided else None
        height = height if height_provided else None
        if width_provided:
            if isinstance(width, bool):
                return "编辑失败：width 必须是 64 到 8192 之间的整数。"
            try:
                width = _number(width, "width", integer=True, minimum=64, maximum=8192)
            except ValueError as e:
                return f"编辑失败：{e}。"
        if height_provided:
            if isinstance(height, bool):
                return "编辑失败：height 必须是 64 到 8192 之间的整数。"
            try:
                height = _number(height, "height", integer=True, minimum=64, maximum=8192)
            except ValueError as e:
                return f"编辑失败：{e}。"
        dimensions_provided = width_provided or height_provided
        if resolution_provided and dimensions_provided:
            return "编辑失败：resolution 与 width/height 是两种画布设置方式，请只选一种。"
        if resolution_provided and not slots.get("resolution") and not size_has_dimensions:
            return "编辑失败：此工作流尚未映射 resolution 输入或画面大小节点；请映射 resolution 或 EmptyLatentImage.width/height。"
        if dimensions_provided and not size_has_dimensions:
            return "编辑失败：此工作流尚未映射画面大小节点，无法写入 width/height。"
        custom_size = kwargs.get("custom_size")
        if custom_size is not None:
            if not isinstance(custom_size, bool):
                return "编辑失败：custom_size 必须是布尔值。"
            if custom_size and not slots.get("custom_size"):
                return "编辑失败：此工作流尚未映射 custom_size 开关；若只需指定 EmptyLatentImage 画布，请传 width/height。"
        if dimensions_provided and custom_size is False:
            return "编辑失败：填写 width/height 时请开启 custom_size 或省略该开关。"

        source_paths, source_error = await self._source_paths(
            context, kwargs, max_inputs=len(source_specs),
        )
        if source_error or not source_paths:
            return f"编辑失败：{source_error or '没有可用的来源图片。'}"
        contents: list[bytes] = []
        for source_path in source_paths:
            try:
                if source_path.stat().st_size > 25 * 1024 * 1024:
                    return "编辑失败：单张来源图片不能超过 25 MiB。"
                contents.append(await asyncio.to_thread(source_path.read_bytes))
            except OSError as e:
                return f"编辑失败：读取来源图片失败（{e}）。"
        suffixes = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}
        uploads: list[str] = []
        for content in contents:
            suffix = suffixes.get(image_media_type(content))
            if not suffix:
                return "编辑失败：来源文件需为 PNG、JPEG、WebP 或 GIF 图片。"
            uploads.append(f"astrbot_edit_{hashlib.sha256(content).hexdigest()}{suffix}")
        if resolution_provided:
            effective_resolution = resolution
        elif dimensions_provided:
            effective_resolution = 0
        else:
            effective_resolution = 0
        canvas_dimensions = None
        needs_canvas = dimensions_provided or (
            resolution_provided and (custom_size is True or not slots.get("resolution"))
        ) or not (slots.get("resolution") or slots.get("custom_size"))
        if size_has_dimensions and needs_canvas:
            source_dimensions = image_dimensions(contents[0])
            if dimensions_provided:
                try:
                    canvas_dimensions = _custom_edit_canvas_dimensions(
                        source_dimensions, width, height,
                    )
                except ValueError as e:
                    return f"编辑失败：{e}。"
            else:
                if source_dimensions is None:
                    return "编辑失败：无法读取来源图片宽高，不能按参考图比例设置画布。"
                canvas_dimensions = _edit_canvas_dimensions(*source_dimensions, int(effective_resolution))
        effective_custom_size = custom_size
        if (
            effective_custom_size is None and slots.get("custom_size")
            and (dimensions_provided or (resolution_provided and not slots.get("resolution")))
        ):
            effective_custom_size = True
        seed = random.randint(0, 2**31 - 1) if slots.get("sampler") or slots.get("sampler_2") else None
        apply_values = {"prompt": prompt, "seed": seed}
        if slots.get("source_images"):
            apply_values["source_images"] = uploads
        else:
            apply_values["source_image"] = uploads[0]
        if slots.get("resolution"):
            apply_values["resolution"] = effective_resolution
        if canvas_dimensions is not None:
            apply_values["width"], apply_values["height"] = canvas_dimensions
        if slots.get("custom_size"):
            apply_values["custom_size"] = (
                effective_custom_size if effective_custom_size is not None else False
            )
        try:
            apply_slots(
                wf, slots, apply_values,
                drop_nodes=profile.get("drop_nodes") or [],
            )
        except (TypeError, ValueError) as e:
            return f"编辑失败：工作流节点映射无效（{e}）。"
        for index, (filename, content) in enumerate(zip(uploads, contents)):
            upload_name, upload_error = await self.client.upload_image(filename, content)
            if upload_error or not upload_name:
                return f"编辑失败：上传第 {index + 1} 张来源图片失败（{upload_error or 'ComfyUI 未返回文件名'}）。"
            uploads[index] = upload_name
        apply_slots(wf, slots, {"source_images": uploads})
        queued = await _enqueue_generation(self.shared, context, wf, {
            "entry": "edit", "family": edit_route.name, "edit_route": edit_route.name,
            "workflow": edit_route.workflow, "prompt": prompt,
            "source_path": str(source_paths[0]), "source_paths": [str(path) for path in source_paths],
            "values": apply_values,
        })
        if queued is not None:
            return queued
        pid, submit_error = await self.client.submit_prompt_detail(wf)
        if submit_error or not pid:
            return f"编辑失败：{submit_error or '无法连接 ComfyUI'}"
        _remember_prompt_id(self.shared, context, pid)
        outputs, wait_error = await _wait_outputs(self.client, pid)
        if wait_error:
            return wait_error if wait_error.startswith(("等待超时", "等待中断")) else f"编辑失败：{wait_error}"
        images = [img for output in (outputs or {}).values() for img in output.get("images", [])]
        if not images:
            return "编辑完成，但没有图片输出。"
        image = next((img for img in images if img.get("type") == "output"), images[-1])
        filename = str(image.get("filename") or "")
        if not filename:
            return "编辑完成，但输出图片缺少文件名。"
        data = await self.client.download_image(filename, image.get("subfolder", ""), image_type=image.get("type", "output"))
        if not data:
            return f"编辑完成，但下载图片失败（{filename}）。"
        try:
            local_path = await asyncio.to_thread(save_image, self.output_dir, filename, data)
        except (OSError, ValueError) as e:
            return f"编辑完成，但保存图片失败（{e}）。"
        _remember_image_path(self.shared, context, local_path)
        self.store.save_history({
            "prompt_id": pid, "entry": "edit", "family": edit_route.name,
            "edit_route": edit_route.name, "workflow": edit_route.workflow,
            "prompt": prompt, "source_path": str(source_paths[0]),
            "source_paths": [str(path) for path in source_paths],
            "filename": filename, "local_path": str(local_path),
        })
        try:
            event: AstrMessageEvent = context.context.event
            await event.send(MessageChain().file_image(str(local_path)))
        except Exception as e:
            return f"图片已编辑但发送失败（{e}）。本地路径: {local_path}"
        seed_note = f"seed={seed} " if seed is not None else ""
        image_note = ""
        if len(source_specs) > 1:
            image_note = (
                f" 已填入 {len(uploads)}/{len(source_specs)} 个参考图输入；"
                "其余输入沿用工作流默认值。"
            )
        return f"图片已编辑并发送。本地路径: {local_path}\n{seed_note}prompt_id={pid}.{image_note}"


_RECIPE_DRAW_DESC = (
    "使用已经实验并保存好的配方快捷生成新图片，完成后直接发送到当前会话；修改现有图片使用 comfyui_edit。"
    "prompt 必填，recipe 在用户点名配方时填写；省略 recipe 使用配置的默认配方。"
    "配方保存底模、LoRA、画幅和采样参数，并通过 model family 使用当前配置的工作流。"
    "本工具用于复用固定方案；需要自由选择底模、LoRA 或采样参数时调用 comfyui_draw。"
    + _ASYNC_GENERATION_HINT
)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiRecipeDrawTool(FunctionTool[AstrAgentContext]):
    """按配方参数快捷生图。"""

    name: str = "comfyui_recipe_draw"
    description: str = _RECIPE_DRAW_DESC
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "本次要画的内容，必填；会写入配方所属家族工作流的主提示词槽位",
                },
                "recipe": {
                    "type": "string",
                    "description": "已保存的配方名。用户点名时填写，省略则使用默认配方",
                },
                "size": {
                    "type": "string",
                    "enum": ["portrait", "landscape", "square", "same"],
                    "description": "可选的临时画幅方向；省略时完整沿用配方",
                },
                "seed": {
                    "type": "number",
                    "description": "仅在复现结果时填写；省略则随机",
                },
            },
            "required": ["prompt"],
        }
    )
    draw_tool: ComfyuiDrawTool | None = None
    store: RecipeStore | None = None
    families: ModelFamilyRegistry | None = None

    def refresh_schema(self) -> None:
        names = _recipe_enum(self.store)
        self.description = _RECIPE_DRAW_DESC + ("" if names else " 当前还没有配方。")
        prop = self.parameters.setdefault("properties", {}).setdefault(
            "recipe", {"type": "string"}
        )
        if names:
            prop["enum"] = names
            prop["description"] = "用户点名时从 enum 选择；省略用默认配方"
        else:
            prop.pop("enum", None)
            prop["description"] = "当前没有配方，请先在 Workflow Studio 静态配方页保存"

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            return "生成失败：prompt 不能为空。"
        if self.draw_tool is None or self.store is None or self.families is None:
            return "生成失败：插件未初始化完成。"

        recipe_name = str(kwargs.get("recipe") or "").strip()
        recipe = self.store.get(recipe_name) if recipe_name else self.store.default()
        if recipe is None:
            if recipe_name:
                return f"生成失败：配方「{recipe_name}」不存在。"
            return "生成失败：还没有可用配方，请先在 Workflow Studio 静态配方页保存一套。"

        family = self.families.resolve_recipe(recipe)
        explicit_family = str(recipe.get("family") or "").strip()
        has_legacy_route = bool(
            recipe_template(recipe) and (recipe.get("slots") or {}).get("prompt")
        )
        if explicit_family and family is None and not has_legacy_route:
            available = "、".join(self.families.names())
            return (
                f"生成失败：配方「{recipe.get('name')}」引用的模型家族「{explicit_family}」"
                f"未配置。当前可用：{available}"
            )
        legacy_workflow = ""
        legacy_slots: dict | None = None
        legacy_drop_nodes: list[str] | None = None
        if family is None:
            legacy_workflow = recipe_template(recipe)
            legacy_slots = recipe.get("slots") or {}
            legacy_drop_nodes = list(recipe.get("drop_nodes") or [])
            if not legacy_workflow:
                return (
                    f"生成失败：配方「{recipe.get('name')}」没有模型家族。"
                    "请在 Workflow Studio 静态配方页为它选择家族后重新保存。"
                )

        seed = kwargs.get("seed")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        values = materialize_values(
            recipe,
            {
                "prompt": prompt,
                "seed": seed,
            },
        )
        return await self.draw_tool._execute(
            context,
            prompt=prompt,
            values=values,
            family=family,
            recipe=recipe,
            size_token=str(kwargs.get("size") or "").strip(),
            legacy_workflow=legacy_workflow,
            legacy_slots=legacy_slots,
            legacy_drop_nodes=legacy_drop_nodes,
        )


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiLookupTool(FunctionTool[AstrAgentContext]):
    """查角色/画师/LoRA 触发词，短回包。"""

    name: str = "comfyui_lookup"
    description: str = (
        "查询角色/画师规范词和已安装底模/LoRA。model/lora 指定 query 可跨家族按名称搜索；"
        "浏览清单时传与生图一致的 model_family，两个参数都省略时返回家族数量。"
        "绘图需要某种画风、角色、服饰或效果时，可主动查询匹配的 LoRA，用户无需点名 LoRA 或提供文件名。"
        "character 支持中文名并返回有来源的外观参考；同名和颜色候选须核对版本。artist 触发词写进 artist。"
        "model/lora：选择符合需求的结果，把实际文件名填进 comfyui_draw 的 model/lora。LoRA 的 query 支持 LoRA Manager/Civitai "
        "分类或标签，例如 style/character/concept/风格/角色；结果会带用途说明、推荐权重和触发词。"
        "选用 LoRA 时可同步传入已记录的触发词；查询未提供触发词时可省略该字段并继续使用 LoRA。"
        "查无结果时可换类别或用途关键词搜索，也可继续使用配方默认值。文件名和规范词以查询结果为准。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["character", "artist", "model", "lora"],
                    "description": "character=角色，artist=画师，model=底模文件名，lora=LoRA 文件名与触发词",
                },
                "query": {
                    "type": "string",
                    "description": "角色/画师名或模型名称；模型名可跨家族近似搜索。LoRA 用途标签或 style/character/concept 分类需指定 model_family。支持中文、日文、罗马音",
                },
                "limit": {
                    "type": "number",
                    "description": "type=lora 时最多返回几项（1-8，默认 5）",
                },
                **_RESOURCE_QUERY_PROPERTIES,
            },
            "required": ["type"],
        }
    )
    danbooru: DanbooruClient | None = None
    gelbooru: GelbooruClient | None = None
    animadex: AnimaDexClient | None = None
    character_lookup: CharacterTagLookup | None = None
    client: ComfyUIClient | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        kind = str(kwargs.get("type") or "").strip().lower()
        query = str(kwargs.get("query") or "").strip()
        if kind not in ("character", "artist", "model", "lora"):
            return "查询失败：需要 type（character/artist/model/lora）。"
        if kind in {"model", "lora"}:
            if self.client is None:
                return "查询失败：ComfyUI 未配置。"
            if kind == "model":
                # Model names do not need a full LoRA metadata/header scan.
                resources = await self.client.generation_resources()
            else:
                resources, _ = await self.client.list_resources()
            return _resource_page(self.client, resources, kind, query,
                                  str(kwargs.get("model_family") or ""), kwargs.get("limit", 5),
                                  kwargs.get("offset", 0), _as_bool(kwargs.get("include_unknown")))
        if not query:
            return "查询角色/画师时 query 必填。"

        if kind == "character" and self.character_lookup is not None:
            try:
                limit = max(1, min(int(kwargs.get("limit") or 5), 8))
            except (TypeError, ValueError):
                limit = 5
            return format_character_result(await self.character_lookup.lookup(query, limit))

        if kind == "character" and self.animadex is not None:
            text = await self.animadex.search_characters(query, page=1)
            if text:
                clipped = text.strip()
                if len(clipped) > 800:
                    clipped = clipped[:800] + "…"
                return f"【角色 {query}】\n{clipped}"

        if kind == "artist":
            data = None
            source = "danbooru"
            if self.danbooru is not None:
                data = await self.danbooru.search_artist(query, 0)
            if data is None and self.gelbooru is not None:
                data = await self.gelbooru.search_artist(query, 20)
                source = "gelbooru"
            if data is None:
                return f"未找到画师：{query}"
            aliases = ", ".join((data.get("aliases") or [])[:6])
            trigger = data.get("artist") or query
            extra = f" 别名: {aliases}" if aliases else ""
            return f"【画师 @{trigger}】来源 {source}{extra}\n触发词: @{trigger}"

        data = None
        source = "danbooru"
        if self.danbooru is not None:
            data = await self.danbooru.search_character(query, 0)
        if data is None and self.gelbooru is not None:
            data = await self.gelbooru.search_character(query, 20)
            source = "gelbooru"
        if data is None:
            return f"未找到角色：{query}"
        name = data.get("character") or query
        aliases = ", ".join((data.get("aliases") or [])[:8])
        extra = f"\n别名: {aliases}" if aliases else ""
        return f"【角色 {name}】来源 {source}{extra}\n触发词: {name}"



async def _wait_outputs(client: ComfyUIClient, prompt_id: str) -> tuple[dict | None, str | None]:
    entry, error = await client.wait_for_history(prompt_id)
    if error or entry is None:
        return None, error
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        return None, execution_error_message(status, "执行出错（详见 ComfyUI 日志）")
    return entry.get("outputs", {}), None


def _has_lora_input(kwargs: dict[str, Any]) -> bool:
    value = _lora_input(kwargs)
    return value is not None and value != ""


def _lora_input(kwargs: dict[str, Any]) -> Any:
    """Read canonical `lora` input and retain the older `loras` alias."""
    value = kwargs.get("lora")
    if value is not None and value != "":
        return value
    return kwargs.get("loras")
