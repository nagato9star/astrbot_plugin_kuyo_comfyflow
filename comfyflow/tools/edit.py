"""来源图片解析与工作流编辑生成。"""

from __future__ import annotations

import asyncio
import hashlib
import random
from pathlib import Path
from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event import MessageChain
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from pydantic import ConfigDict
from pydantic import Field
from pydantic.dataclasses import dataclass
from ..generation.receipts import hide_task_ids
from ..generation.receipts import task_reference
from ..clients.comfyui import ComfyUIClient
from ..clients.comfyui import image_dimensions
from ..clients.comfyui import image_media_type
from ..storage.images import save_image
from ..workflows.registry import EditWorkflowRegistry
from ..workflows.registry import ModelFamilyRegistry
from ..workflows.registry import WorkflowProfileStore
from ..storage.recipes import RecipeStore
from ..workflows.slots import apply_slots
from ..workflows.slots import detect_slots
from ..workflows.slots import source_image_slots
from ..workflows.builder import WorkflowBuilder
from .common import _ASYNC_GENERATION_HINT, _custom_edit_canvas_dimensions, _edit_canvas_dimensions, _enqueue_generation, _event_scope, _message_images, _number, _remember_image_path, _remember_prompt_id, _wait_outputs


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
            return hide_task_ids(wait_error, [pid]) if wait_error.startswith(("等待超时", "等待中断")) else f"编辑失败：{wait_error}"
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
        return f"图片已编辑并发送。本地路径: {local_path}\n{seed_note}{image_note}\n" + task_reference({"prompt_id": pid})
