"""配方管理与快捷配方生成。"""

from __future__ import annotations

import json
import random
from typing import Any
from astrbot.api import FunctionTool
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from pydantic import ConfigDict
from pydantic import Field
from pydantic.dataclasses import dataclass
from ..workflows.registry import ModelFamilyRegistry
from ..storage.recipes import RecipeStore
from ..storage.recipes import materialize_values
from ..storage.recipes import recipe_family
from ..storage.recipes import recipe_template
from ..workflows.slots import parse_lora
from ..workflows.builder import WorkflowBuilder
from .common import _ASYNC_GENERATION_HINT, _lora_array_schema, _recipe_enum
from .draw import ComfyuiDrawTool


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
