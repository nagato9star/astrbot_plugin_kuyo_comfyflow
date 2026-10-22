"""按模型家族和工作流参数生成图片。"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any
from astrbot.api import FunctionTool
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event import MessageChain
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from pydantic import ConfigDict
from pydantic import Field
from pydantic.dataclasses import dataclass
from ..workflows.parameters import fill_trigger_words
from ..generation.receipts import task_reference
from ..clients.comfyui import ComfyUIClient
from ..storage.images import save_image
from ..workflows.registry import ModelFamily
from ..workflows.registry import ModelFamilyRegistry
from ..workflows.registry import WorkflowProfileStore
from ..storage.recipes import RecipeStore
from ..workflows.slots import ANIMA_DROP_NODES
from ..workflows.slots import apply_slots
from ..workflows.slots import detect_slots
from ..workflows.slots import looks_like_anima
from ..workflows.slots import parse_lora
from ..workflows.slots import read_current_values
from ..workflows.slots import resolve_size
from ..resources.catalog import exact_matches
from ..resources.catalog import filter_family
from ..resources.catalog import metadata_for
from ..resources.catalog import selection_error
from ..workflows.builder import WorkflowBuilder
from .common import _ASYNC_GENERATION_HINT, _enqueue_generation, _has_lora_input, _lora_array_schema, _match_resource, _remember_image_path, _remember_prompt_id, _validate_generation_values, _wait_outputs
from .reference import _match_lora_resources


_DRAW_DESC = (
    "从文字生成新图片并发送；需要修改现有图片时使用 comfyui_edit。model_family、prompt 必填，提示词遵循家族 prompt_style。"
    "省略可选参数沿用工作流。可按画风、角色、服饰或效果需求主动用 comfyui_lookup 查询并选用 LoRA；"
    "分辨率选择器工作流可用 aspect_ratio 和 megapixels 覆盖比例与目标百万像素数；"
    "按用途、类别或标签查询资源时传 model_family；完整已知文件名可跨家族查找。使用前核对资源与当前家族兼容，并采用返回的文件名、推荐权重和已知触发词。"
    "复用配方用 comfyui_recipe_draw。"
    + _ASYNC_GENERATION_HINT
)


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
                    "description": "画师风格串。可根据用户期望的画风主动选择有依据的规范画师标签；仅在家族支持画师/tag 提示词时使用，未知名称先用 comfyui_lookup(type=artist) 核实；用户明确不指定画师时不填",
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

        return f"图片已发送。本地路径: {local_path}\nseed={seed}.{saved_note}\n" + task_reference({"prompt_id": pid})

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
