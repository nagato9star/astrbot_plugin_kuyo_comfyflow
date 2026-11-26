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
    "按当前人格结合用户要画的内容简短接话，不照读排队回执；无需播报内部编号、轮询或重复提交。"
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


def _allowed_image_roots(output_dir: Path | None) -> list[Path]:
    roots: list[Path] = []
    if output_dir is not None:
        roots.append(output_dir.resolve())
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

        roots.append(Path(get_astrbot_temp_path()).resolve())
    except (ImportError, OSError, RuntimeError):
        pass
    return roots


def _resolve_allowed_image_paths(
    values: list[str],
    output_dir: Path | None,
    *,
    max_inputs: int = 1,
) -> tuple[list[Path] | None, str | None]:
    resolved: list[Path] = []
    roots = _allowed_image_roots(output_dir)
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


async def _resolve_source_images(
    context: ContextWrapper[AstrAgentContext],
    kwargs: dict[str, Any],
    *,
    output_dir: Path | None,
    shared: dict,
    max_inputs: int = 1,
) -> tuple[list[Path] | None, str | None]:
    """解析来源图片（显式路径 / 当前与引用消息附图 / 会话上次生成图）。"""
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

    if raw_paths:
        return _resolve_allowed_image_paths(raw_paths, output_dir, max_inputs=max_inputs)

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
                hint = "image_index 或 image_indices" if max_inputs > 1 else "image_index"
                return None, f"当前消息有 {len(images)} 张图片，但当前仅支持 {max_inputs} 张输入；请用 {hint} 选择。"
            indices = list(range(1, len(images) + 1))

        if len(indices) > max_inputs:
            return None, f"image_indices 不能超过当前支持的 {max_inputs} 张输入。"
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
        return None, "当前消息及引用消息没有可读取图片，image_index/image_indices 无法选择；请使用 image_path。"
    last = str((shared.get("last_image_paths") or {}).get(_event_scope(context)) or "")
    if last and Path(last).is_file():
        return [Path(last)], None
    return None, "当前消息没有图片，且本会话没有可用的上次生成图片；请附图后重试。"


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
