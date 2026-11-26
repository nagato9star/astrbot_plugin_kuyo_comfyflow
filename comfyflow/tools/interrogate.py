"""图片提示词反推工具：优先读取内嵌生成元数据，未命中时调用云端 VLM 反推。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from astrbot.api import FunctionTool
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from ..clients.vision import VisionInterrogateClient
from ..resources.interrogate import format_reverse_prompt_result, reverse_prompt
from ..workflows.registry import ModelFamilyRegistry
from .common import _as_bool, _resolve_source_images

MAX_SOURCE_BYTES = 25 * 1024 * 1024


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiReversePromptTool(FunctionTool[AstrAgentContext]):
    """从消息附图、本地输出图或引用图片中提取原图元数据或通过云端 VLM 反推提示词。"""

    name: str = "comfyui_reverse_prompt"
    description: str = (
        "对当前消息、引用消息、上次生图或指定路径的图片反推生图提示词。"
        "优先毫秒级读取 PNG/ComfyUI/A1111/NovelAI 内嵌生成参数；未命中时调用云端视觉模型（不占本地显卡）。"
        "支持按目标 model_family 或 style 输出自然语言描述或 Danbooru 标签串。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "style": {
                    "type": "string",
                    "enum": ["auto", "danbooru", "natural"],
                    "description": "输出风格：danbooru=英文标签串，natural=自然语言描述，auto=同时输出两段或跟随 model_family",
                },
                "model_family": {
                    "type": "string",
                    "description": "可选目标生图家族；style=auto 时会自动匹配该家族偏好的 prompt_style",
                },
                "focus": {
                    "type": "string",
                    "description": "可选聚焦要求，例如「只描述服装和动作」「重点提取构图、视角和光影」",
                },
                "read_metadata": {
                    "type": "boolean",
                    "description": "是否优先读取图片内嵌提示词元数据，默认 true；想强制走视觉分析时传 false",
                },
                "image_path": {
                    "type": "string",
                    "description": "可选本地图片路径（支持本插件输出目录或 AstrBot data/temp 文件）；附图时省略",
                },
                "image_index": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "当前消息有多张附图时指定第几张（从 1 开始）",
                },
            },
        }
    )
    vision: VisionInterrogateClient | None = None
    output_dir: Path | None = None
    shared: dict = Field(default_factory=dict)
    families: ModelFamilyRegistry | None = None

    def refresh_schema(self) -> None:
        family_prop = self.parameters["properties"]["model_family"]
        names = self.families.names() if self.families is not None else []
        if names:
            family_prop["enum"] = names
        else:
            family_prop.pop("enum", None)

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs: Any) -> str:
        paths, source_error = await _resolve_source_images(
            context,
            kwargs,
            output_dir=self.output_dir,
            shared=self.shared,
            max_inputs=1,
        )
        if source_error or not paths:
            return f"反推失败：{source_error or '没有可用的来源图片。'}"

        source_path = paths[0]
        try:
            if source_path.stat().st_size > MAX_SOURCE_BYTES:
                return "反推失败：单张来源图片不能超过 25 MiB。"
            image_bytes = await asyncio.to_thread(source_path.read_bytes)
        except OSError as exc:
            return f"反推失败：读取来源图片失败（{exc}）。"

        style = str(kwargs.get("style") or "auto").strip().lower()
        if style not in {"auto", "danbooru", "natural"}:
            return "反推失败：style 仅支持 auto、danbooru 或 natural。"
        model_family = str(kwargs.get("model_family") or "").strip()
        focus = str(kwargs.get("focus") or "").strip()
        read_metadata = _as_bool(kwargs.get("read_metadata", True), default=True)

        try:
            result = await reverse_prompt(
                image_path=source_path,
                image_bytes=image_bytes,
                style=style,
                model_family=model_family,
                focus=focus,
                read_metadata=read_metadata,
                families=self.families,
                vision=self.vision,
            )
        except ValueError as exc:
            return f"反推失败：{exc}"
        except Exception as exc:
            return f"反推失败：{exc}"

        return format_reverse_prompt_result(result)
