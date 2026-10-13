"""工具共用的参数解析与会话状态。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from astrbot.api.message_components import Image
from astrbot.api.message_components import Reply
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from ..generation.receipts import mark_task_reply
from ..clients.comfyui import ComfyUIClient
from ..clients.comfyui import execution_error_message
from ..storage.recipes import RecipeStore
from ..resources.catalog import exact_matches


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
    mark_task_reply(context, shared)


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
        receipt = await queue.enqueue(workflow, context.context.event, history=history, saved_recipe=saved_recipe)
        mark_task_reply(context, shared)
        return receipt
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


_RESOURCE_QUERY_PROPERTIES = {
    "model_family": {"type": "string", "description": "资源家族，如 anima/krea2/sdxl/flux/illustrious；unknown 查未识别项。按名称搜索时可省略，按 LoRA 用途挑选时请填写生图家族"},
    "limit": {"type": "integer", "description": "每页数量，默认 5，最多 10"},
    "offset": {"type": "integer", "description": "分页偏移，默认 0；使用返回的 next_offset"},
    "include_unknown": {"type": "boolean", "description": "同时显示家族未知项，默认 false；未知项需核实兼容性"},
}


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
