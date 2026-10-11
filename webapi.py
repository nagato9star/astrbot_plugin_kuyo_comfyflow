"""Workflow Studio API: independent workflow routing/profiles and static recipes."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import random
from pathlib import Path
from typing import Any

from astrbot.api import logger

from comfy_client import (
    ComfyUIClient,
    execution_error_message,
    image_media_type,
)
from image_cache import save_image
from resource_catalog import selection_error
from model_families import EditWorkflowRegistry, ModelFamilyRegistry, WorkflowProfileStore
from config_options import refresh_config_options
from recipe_store import RecipeStore, materialize_values, recipe_template
from generation_support import fill_trigger_words
from slot_mapping import (
    SLOT_BASIC,
    SLOT_HELP,
    SLOT_ROLES,
    apply_slots,
    parse_lora,
    detect_slots,
    is_ui_workflow,
    list_nodes,
    looks_like_anima,
    node_options_for_slot,
    normalize_workflow,
    read_current_values,
    resolve_size,
    source_image_slots,
    ANIMA_DROP_NODES,
)
from workflow_builder import WorkflowBuilder
from workflow_graph import validate_api_workflow, validate_ui_snapshot

PLUGIN_NAME = "astrbot_plugin_kuyo_comfyflow"

SAMPLER_NAMES = [
    "er_sde",
    "euler",
    "euler_ancestral",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
    "dpmpp_sde",
    "ddim",
    "uni_pc",
    "lcm",
]
SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "beta"]


def _json(data: dict, status: int = 200):
    try:
        from astrbot.api.web import json_response

        return json_response(data, status_code=status)
    except Exception:
        from quart import jsonify

        resp = jsonify(data)
        resp.status_code = status
        return resp


async def _body() -> dict:
    try:
        from astrbot.api.web import request as web_request

        payload = await web_request.json(default={})
        return payload if isinstance(payload, dict) else {}
    except Exception:
        from quart import request

        return await request.get_json(force=True, silent=True) or {}


def _query(key: str, default: str = "") -> str:
    try:
        from astrbot.api.web import request as web_request

        val = web_request.query.get(key, default)
        return str(val if val is not None else default)
    except Exception:
        from quart import request

        return str(request.args.get(key) or default)


def _mapped_node_id(spec: Any) -> str:
    """Return the first selected node for scalar or multi-node workflow slots."""
    if isinstance(spec, list):
        spec = spec[0] if spec else None
    if isinstance(spec, dict):
        return str(spec.get("node") or "")
    return str(spec or "")


class StudioApi:
    def __init__(
        self,
        client: ComfyUIClient,
        builder: WorkflowBuilder,
        store: RecipeStore,
        output_dir: Path,
        shared: dict,
        draw_tool: Any = None,
        recipe_draw_tool: Any = None,
        edit_tool: Any = None,
        families: ModelFamilyRegistry | None = None,
        profiles: WorkflowProfileStore | None = None,
        config_defaults: dict | None = None,
        plugin_config: Any = None,
        auto_trigger_words: bool = False,
        edit_workflows: EditWorkflowRegistry | None = None,
    ) -> None:
        self.client = client
        self.builder = builder
        self.store = store
        self.output_dir = output_dir
        self.shared = shared
        self.draw_tool = draw_tool
        self.recipe_draw_tool = recipe_draw_tool
        self.edit_tool = edit_tool
        self.families = families
        self.edit_workflows = edit_workflows
        self.profiles = profiles
        self.config_defaults = config_defaults or {}
        self.auto_trigger_words = auto_trigger_words
        # AstrBotConfig（或测试用的 dict）：供「设为默认配方」直接写配置
        self.plugin_config = plugin_config if isinstance(plugin_config, dict) else {}

    def _refresh_draw_schema(self) -> None:
        refresh_config_options(self.plugin_config, self.builder, self.store)
        if self.draw_tool is not None and hasattr(self.draw_tool, "refresh_schema"):
            self.draw_tool.refresh_schema()
        if self.recipe_draw_tool is not None and hasattr(self.recipe_draw_tool, "refresh_schema"):
            self.recipe_draw_tool.refresh_schema()
        if self.edit_tool is not None:
            self.edit_tool.refresh_schema()
            self.edit_tool.active = bool(
                self.edit_workflows.names()
                if self.edit_workflows is not None
                else (self.families and self.families.editable_names())
            )

    async def status(self) -> Any:
        connected = await self.client.ping(timeout=5.0)
        force = _query("refresh") == "1"
        resources, from_cache = {}, False
        system_stats = None
        if connected:
            try:
                resources, from_cache = await self.client.list_resources(force_refresh=force)
                system_stats = await self.client.get_system_stats()
            except Exception as e:
                logger.warning(f"[ComfyUIDirect] 状态接口取资源失败: {e}")
        return _json(
            {
                "ok": True,
                "connected": connected,
                "base_url": self.client.base_url,
                "from_cache": from_cache,
                "resources": resources,
                "system_stats": system_stats,
                "sampler_names": SAMPLER_NAMES,
                "schedulers": SCHEDULERS,
                "default_workflow": self.builder.default_workflow,
                "model_families": self.families.list() if self.families else [],
                "edit_workflows": self.edit_workflows.list() if self.edit_workflows else [],
                "slot_roles": [
                    {
                        "id": r,
                        "label": lab,
                        "help": SLOT_HELP.get(r, ""),
                        "basic": r in SLOT_BASIC,
                    }
                    for r, lab in SLOT_ROLES
                ],
            }
        )

    async def list_workflows(self) -> Any:
        return _json({"ok": True, "templates": self.builder.list_templates()})

    def _workflow_payload(self, name: str, wf: dict) -> dict:
        refresh_config_options(self.plugin_config, self.builder, self.store)
        detected = detect_slots(wf, getattr(self.client, "_object_info", None))
        profile = (
            self.profiles.effective(name, wf)
            if self.profiles is not None
            else {"slots": detected, "drop_nodes": []}
        )
        selected = profile.get("slots", detected)
        return {
            "ok": True,
            "name": name,
            "source": self.builder.source_of(self.builder.resolve_workflow_path(name)) or "custom",
            "workflow": wf,
            "ui_workflow": self.builder.load_ui_template(name),
            "nodes": list_nodes(wf),
            "detected_slots": detected,
            "profile_slots": selected,
            "profile_source": profile.get("source", "detected"),
            "drop_nodes": profile.get("drop_nodes") or [],
            "slot_options": {
                role: node_options_for_slot(wf, role, _mapped_node_id(selected.get(role)))
                for role, _ in SLOT_ROLES
            },
        }

    async def get_workflow(self) -> Any:
        name = _query("name").strip()
        if not name:
            return _json({"ok": False, "error": "缺少 name 参数"})
        try:
            wf = self.builder.load_template(name)
        except FileNotFoundError as e:
            return _json({"ok": False, "error": str(e)})
        return _json(self._workflow_payload(name, wf))

    async def get_node_definitions(self) -> Any:
        definitions = await self.client.get_object_info()
        if not isinstance(definitions, dict) or not definitions:
            return _json({"ok": False, "error": "无法从远端 ComfyUI 获取节点定义"}, 503)
        return _json({"ok": True, "definitions": definitions})

    async def import_workflow(self) -> Any:
        body = await _body()
        name = str(body.get("name") or "").strip()
        raw = body.get("workflow")
        if not name:
            return _json({"ok": False, "error": "需要 name"})
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as e:
                return _json({"ok": False, "error": f"JSON 解析失败: {e}"})
        if not isinstance(raw, dict):
            return _json({"ok": False, "error": "需要 workflow JSON"})
        object_info = None
        if is_ui_workflow(raw):
            object_info = await self.client.get_object_info()
            if not object_info:
                return _json(
                    {
                        "ok": False,
                        "error": "这是 ComfyUI 前端格式。请先连上 ComfyUI，或改用 Save (API Format)。",
                    }
                )
        try:
            wf = normalize_workflow(raw, object_info)
        except ValueError as e:
            return _json({"ok": False, "error": str(e)})
        try:
            self.builder.save_template(name, wf)
            if is_ui_workflow(raw):
                self.builder.save_ui_template(name, raw)
            else:
                self.builder.delete_ui_template(name)
        except ValueError as e:
            return _json({"ok": False, "error": str(e)})
        payload = self._workflow_payload(name, wf)
        if self.profiles is not None:
            self.profiles.ensure(name, wf)
        return _json(payload)

    async def save_graph(self) -> Any:
        body = await _body()
        name = str(body.get("name") or "").strip()
        try:
            wf = validate_api_workflow(body.get("workflow"))
            ui = validate_ui_snapshot(body.get("ui_workflow"), wf)
            self.builder.save_template(name, wf)
            self.builder.save_ui_template(name, ui)
        except ValueError as e:
            return _json({"ok": False, "error": str(e)}, 400)
        if self.profiles is not None:
            self.profiles.ensure(name, wf)
        self._refresh_draw_schema()
        return _json(self._workflow_payload(name, wf))

    async def run_graph(self) -> Any:
        body = await _body()
        name = str(body.get("name") or "").strip()
        try:
            wf = validate_api_workflow(body.get("workflow"))
            ui = validate_ui_snapshot(body.get("ui_workflow"), wf)
        except ValueError as e:
            return _json({"ok": False, "error": str(e)}, 400)
        prompt_id, error = await self.client.submit_prompt_detail(wf, ui_workflow=ui)
        if error or not prompt_id:
            return _json({"ok": False, "error": error or "ComfyUI 未返回任务编号"}, 400)
        pending = self.shared.setdefault("web_pending_runs", {})
        pending[prompt_id] = {
            "entry": "graph",
            "workflow": name,
            "prompt": "",
            "values": {},
            "recipe": {},
        }
        if len(pending) > 64:
            for old_id in list(pending)[:-64]:
                pending.pop(old_id, None)
        return _json({"ok": True, "prompt_id": prompt_id})

    async def import_from_history(self) -> Any:
        body = await _body()
        prompt_id = str(body.get("prompt_id") or "").strip()
        name = str(body.get("name") or "").strip()
        rows = await self.client.list_history(24)
        entry = next((r for r in rows if r.get("prompt_id") == prompt_id), None) if prompt_id else (rows[0] if rows else None)
        if not entry or not entry.get("workflow"):
            return _json({"ok": False, "error": "ComfyUI 历史里没有可导入的工作流"})
        name = name or f"history-{str(entry['prompt_id'])[:8]}"
        try:
            raw_workflow = entry["workflow"]
            object_info = await self.client.get_object_info() if is_ui_workflow(raw_workflow) else None
            if is_ui_workflow(raw_workflow) and not object_info:
                return _json({"ok": False, "error": "无法从远端获取节点定义以解析历史工作流"}, 503)
            wf = normalize_workflow(raw_workflow, object_info)
            self.builder.save_template(name, wf)
            if is_ui_workflow(raw_workflow):
                self.builder.save_ui_template(name, raw_workflow)
            else:
                self.builder.delete_ui_template(name)
        except ValueError as e:
            return _json({"ok": False, "error": str(e)})
        if self.profiles is not None:
            self.profiles.ensure(name, wf)
        return _json(self._workflow_payload(name, wf))

    async def comfy_history(self) -> Any:
        rows = await self.client.list_history(12)
        slim = [
            {
                "prompt_id": r.get("prompt_id"),
                "number": r.get("number"),
                "status": r.get("status"),
                "has_workflow": r.get("has_workflow"),
            }
            for r in rows
        ]
        return _json({"ok": True, "items": slim})

    async def delete_workflow(self) -> Any:
        body = await _body()
        name = str(body.get("name") or "").strip()
        if not name:
            return _json({"ok": False, "error": "缺少 name"})
        # 引用完整性：生图家族、编辑路由或旧配方仍引用时拒绝删除。
        refs = self.store.used_templates().get(name) or []
        family_refs = (
            self.families.workflow_references().get(name) or []
            if self.families is not None
            else []
        )
        refs.extend(f"模型家族:{family}" for family in family_refs)
        edit_refs = (
            self.edit_workflows.workflow_references().get(name) or []
            if self.edit_workflows is not None
            else []
        )
        refs.extend(f"编辑路由:{route}" for route in edit_refs)
        if refs:
            return _json(
                {
                    "ok": False,
                    "error": f"模板 {name} 仍被配置引用：{'、'.join(refs)}。请先解除或改绑这些引用。",
                    "used_by": refs,
                }
            )
        try:
            self.builder.delete_template(name)
        except ValueError as e:
            return _json({"ok": False, "error": str(e)})
        if self.profiles is not None:
            self.profiles.delete(name)
        refresh_config_options(self.plugin_config, self.builder, self.store)
        return _json({"ok": True, "name": name})

    async def detect(self) -> Any:
        body = await _body()
        name = str(body.get("name") or _query("name")).strip()
        if not name:
            return _json({"ok": False, "error": "缺少 name"})
        try:
            wf = self.builder.load_template(name)
        except FileNotFoundError as e:
            return _json({"ok": False, "error": str(e)})
        detected = detect_slots(wf, getattr(self.client, "_object_info", None))
        return _json(
            {
                "ok": True,
                "slots": detected,
                "values": read_current_values(wf, detected),
                "anima": looks_like_anima(wf),
                "slot_options": {
                    role: node_options_for_slot(wf, role, _mapped_node_id(detected.get(role)))
                    for role, _ in SLOT_ROLES
                },
            }
        )

    async def save_workflow_profile(self) -> Any:
        body = await _body()
        name = str(body.get("workflow") or body.get("name") or "").strip()
        if not name:
            return _json({"ok": False, "error": "缺少 workflow"})
        if self.profiles is None:
            return _json({"ok": False, "error": "工作流档案未初始化"})
        try:
            wf = self.builder.load_template(name)
        except FileNotFoundError as e:
            return _json({"ok": False, "error": str(e)})
        slots = body.get("slots")
        if not isinstance(slots, dict):
            return _json({"ok": False, "error": "slots 必须是对象"})
        try:
            if "source_images" in slots or "source_image" in slots:
                source_image_slots(wf, self.profiles._normalize_slots(slots))
            saved = self.profiles.save(
                name,
                slots,
                body.get("drop_nodes")
                or (ANIMA_DROP_NODES if looks_like_anima(wf) else []),
            )
        except ValueError as e:
            return _json({"ok": False, "error": str(e)})
        return _json({"ok": True, "profile": saved})

    async def save_workflow_binding(self) -> Any:
        body = await _body()
        mode = str(body.get("mode") or "").strip().lower()
        family_name = str(body.get("family") or "").strip()
        edit_route_name = str(body.get("edit_route") or "").strip()
        workflow = str(body.get("workflow") or "").strip()
        if mode not in {"generate", "edit"}:
            return _json({"ok": False, "error": "mode 仅支持 generate / edit"})
        independent_edit = mode == "edit" and self.edit_workflows is not None and "edit_route" in body
        family = self.families.get(family_name) if self.families is not None else None
        if mode == "generate" and family is None:
            return _json({"ok": False, "error": f"模型家族不存在: {family_name}"})
        if mode == "edit" and not independent_edit and family is None:
            return _json({"ok": False, "error": f"模型家族不存在: {family_name}"})
        if mode == "generate" and not workflow:
            return _json({"ok": False, "error": "生图工作流不能为空"})
        if independent_edit and not edit_route_name:
            return _json({"ok": False, "error": "编辑路由名不能为空"})
        workflow_data = None
        if workflow:
            try:
                workflow_data = self.builder.load_template(workflow)
            except (FileNotFoundError, ValueError) as e:
                return _json({"ok": False, "error": str(e)})

        warnings: list[str] = []
        if workflow_data is not None:
            profile = (
                self.profiles.effective(workflow, workflow_data)
                if self.profiles is not None else {"slots": detect_slots(workflow_data)}
            )
            slots = profile.get("slots") or {}
            prompt_node = str((slots.get("prompt") or {}).get("node") or "")
            if prompt_node not in workflow_data:
                warnings.append("主提示词节点尚未有效映射")
            if mode == "edit":
                image_node = _mapped_node_id(slots.get("source_images") or slots.get("source_image"))
                if (workflow_data.get(image_node) or {}).get("class_type") != "LoadImage":
                    warnings.append("编辑来源图片尚未映射到 LoadImage")

        if independent_edit:
            old_edit_workflows = copy.deepcopy(self.plugin_config.get("edit_workflows") or [])
            editing = copy.deepcopy(old_edit_workflows)
            row = next(
                (r for r in editing if isinstance(r, dict)
                 and str(r.get("name") or "").casefold() == edit_route_name.casefold()),
                None,
            )
            if row is None:
                row = {"__template_key": "edit_workflow", "name": edit_route_name}
                editing.append(row)
            row["workflow"] = workflow
            if "description" in body:
                row["description"] = str(body.get("description") or "").strip()
            self.plugin_config["edit_workflows"] = editing
            save = getattr(self.plugin_config, "save_config", None)
            try:
                if callable(save):
                    save()
            except (OSError, ValueError) as e:
                self.plugin_config["edit_workflows"] = old_edit_workflows
                return _json({"ok": False, "error": f"保存编辑路由失败: {e}"})

            self.edit_workflows.reconfigure(
                editing,
                self.plugin_config.get("edit_families") or [],
                self.plugin_config.get("model_families") or [],
            )
            self._refresh_draw_schema()
            return _json({
                "ok": True,
                "edit_workflows": self.edit_workflows.list(),
                "warnings": warnings,
            })

        old_gen = copy.deepcopy(self.plugin_config.get("model_families") or [])
        old_edit = copy.deepcopy(self.plugin_config.get("edit_families") or [])
        generation = copy.deepcopy(old_gen)
        editing = copy.deepcopy(old_edit)
        if mode == "generate":
            row = next(
                (r for r in generation if isinstance(r, dict)
                 and str(r.get("name") or r.get("family") or "").casefold() == family.name.casefold()),
                None,
            )
            if row is None:
                return _json({"ok": False, "error": "配置中找不到该生图家族，请在插件配置页添加"})
            row["workflow"] = workflow
        else:
            row = next(
                (r for r in editing if isinstance(r, dict)
                 and str(r.get("model_family") or r.get("family") or r.get("name") or "").casefold() == family.name.casefold()),
                None,
            )
            if row is None:
                editing.append({
                    "__template_key": "edit_family",
                    "model_family": family.name,
                    "workflow": workflow,
                })
            else:
                row["workflow"] = workflow

        self.plugin_config["model_families"] = generation
        self.plugin_config["edit_families"] = editing
        save = getattr(self.plugin_config, "save_config", None)
        try:
            if callable(save):
                save()
        except (OSError, ValueError) as e:
            self.plugin_config["model_families"] = old_gen
            self.plugin_config["edit_families"] = old_edit
            return _json({"ok": False, "error": f"保存家族绑定失败: {e}"})

        self.families.reconfigure(generation, self.builder.default_workflow, editing)
        self.builder.default_workflow = self.families.first().workflow
        if self.edit_workflows is not None:
            self.edit_workflows.reconfigure(
                self.plugin_config.get("edit_workflows") or [],
                editing,
                generation,
            )
        self._refresh_draw_schema()
        return _json({
            "ok": True,
            "model_families": self.families.list(),
            "edit_workflows": self.edit_workflows.list() if self.edit_workflows else [],
            "warnings": warnings,
        })

    async def list_recipes(self) -> Any:
        return _json(
            {
                "ok": True,
                "recipes": self.store.list(),
                "default_recipe": self._configured_default_recipe(),
            }
        )

    def _configured_default_recipe(self) -> str:
        return str(self.plugin_config.get("default_recipe") or "").strip()

    async def set_default_recipe(self) -> Any:
        body = await _body()
        name = str(body.get("name") or "").strip()
        if not name:
            return _json({"ok": False, "error": "缺少 name"})
        if self.store.get(name) is None:
            return _json({"ok": False, "error": f"配方不存在: {name}"})
        self.plugin_config["default_recipe"] = name
        save = getattr(self.plugin_config, "save_config", None)
        if callable(save):
            save()
        # 运行中的 store 立即生效，不等插件重载
        self.store.preferred_default = name
        self._refresh_draw_schema()
        return _json({"ok": True, "default_recipe": name})

    async def get_recipe(self) -> Any:
        name = _query("name").strip()
        recipe = self.store.get(name) if name else self.store.default()
        if recipe is None:
            return _json({"ok": False, "error": "配方不存在"})
        family = self.families.resolve_recipe(recipe) if self.families else None
        payload = dict(recipe)
        if family is not None:
            payload["family"] = family.name
        return _json(
            {
                "ok": True,
                "recipe": payload,
                "resolved_family": family.public() if family is not None else None,
            }
        )

    async def save_recipe(self) -> Any:
        body = await _body()
        family_name = str(body.get("family") or body.get("model_family") or "").strip()
        family = self.families.get(family_name) if self.families is not None else None
        if family is None:
            available = "、".join(self.families.names()) if self.families else "无"
            return _json(
                {
                    "ok": False,
                    "error": f"请选择已配置的模型家族。当前可用：{available}",
                }
            )
        body["family"] = family.name
        try:
            saved = self.store.save(body)
        except ValueError as e:
            return _json({"ok": False, "error": str(e)})
        self._refresh_draw_schema()
        return _json({"ok": True, "recipe": saved})

    async def delete_recipe(self) -> Any:
        body = await _body()
        name = str(body.get("name") or "").strip()
        rid = str(body.get("id") or "").strip()
        removed = self.store.delete(rid or name)
        if not removed:
            return _json({"ok": False, "error": "配方不存在"})
        self._refresh_draw_schema()
        return _json({"ok": True, "name": name, "removed": removed})

    async def generate(self) -> Any:
        body = await _body()
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            return _json({"ok": False, "error": "prompt 不能为空"})
        recipe_name = str(body.get("recipe") or "").strip()
        recipe = self.store.get(recipe_name) if recipe_name else self.store.default()
        if recipe is None:
            return _json({"ok": False, "error": "还没有配方，请先导入工作流并保存配方"})
        family = self.families.resolve_recipe(recipe) if self.families else None
        explicit_family = str(recipe.get("family") or "").strip()
        has_legacy_route = bool(
            recipe_template(recipe) and (recipe.get("slots") or {}).get("prompt")
        )
        if explicit_family and family is None and not has_legacy_route:
            return _json(
                {
                    "ok": False,
                    "error": f"配方引用的模型家族「{explicit_family}」未配置",
                }
            )
        tname = family.workflow if family is not None else recipe_template(recipe)
        if not tname:
            return _json({"ok": False, "error": "配方没有模型家族，请重新保存配方"})
        try:
            wf = self.builder.load_template(tname)
        except FileNotFoundError:
            return _json(
                {
                    "ok": False,
                    "error": f"模型家族使用的工作流「{tname}」不存在，请检查配置",
                }
            )
        if family is not None and self.profiles is not None:
            profile = self.profiles.effective(tname, wf)
            slots = profile.get("slots") or {}
            drop_nodes = list(profile.get("drop_nodes") or [])
        else:
            slots = recipe.get("slots") or detect_slots(wf)
            drop_nodes = list(recipe.get("drop_nodes") or [])
        if not slots.get("prompt"):
            return _json({"ok": False, "error": f"工作流「{tname}」未指定主提示词节点"})

        seed = body.get("seed")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            return _json({"ok": False, "error": "seed 必须是整数"})
        if not 0 <= seed <= 2**63 - 1:
            return _json({"ok": False, "error": "seed 超出允许范围（0 ~ 2^63-1）"})
        defaults = recipe.get("defaults") or {}
        loras = body.get("loras") if "loras" in body else body.get("lora")
        trigger = body.get("trigger_words")
        overrides = {
            "prompt": prompt,
            "seed": seed,
            "artist": body.get("artist"),
            "model": body.get("model"),
            "loras": loras,
            "width": body.get("width"),
            "height": body.get("height"),
            "steps": body.get("steps"),
            "cfg": body.get("cfg"),
            "sampler_name": body.get("sampler_name"),
            "scheduler": body.get("scheduler"),
            "denoise": body.get("denoise"),
            "trigger_words": trigger,
        }
        values = materialize_values(recipe, overrides)
        if trigger is not None:
            values["trigger_words"] = trigger
        elif "trigger_words" in defaults:
            values["trigger_words"] = defaults["trigger_words"]
        values["seed"] = seed
        # 与机器人 comfyui_generate 保持一致：配方没配的项补插件配置默认值（画师/画质/负向等）
        for key, val in self.config_defaults.items():
            if key not in values and val not in (None, "", 0, 0.0, []):
                values["negative" if key == "negative_prompt" else key] = val
        if family is not None and (values.get("model") or values.get("loras")):
            resources = await self.client.generation_resources()
            names = [str(item.get("name")) for item in parse_lora(values.get("loras")) if item.get("name")]
            if names:
                resources = await self.client.selected_lora_metadata(resources, names, require_trusted=self.auto_trigger_words and values.get("trigger_words") is None)
            values = fill_trigger_words(values, resources, self.auto_trigger_words and bool(slots.get("trigger_words")))
            error = selection_error(resources, values, family.name,
                                    getattr(self.client, "resource_family_rules", []))
            if error:
                return _json({"ok": False, "error": error}, 400)
        if body.get("size"):
            values["width"], values["height"] = resolve_size(
                int(values["width"]) if values.get("width") else None,
                int(values["height"]) if values.get("height") else None,
                str(body.get("size")),
                presets=recipe.get("size_presets"),
            )
        try:
            apply_slots(
                wf,
                slots,
                values,
                drop_nodes=drop_nodes,
            )
        except ValueError as e:
            return _json({"ok": False, "error": str(e)}, 400)
        pid, err = await self.client.submit_prompt_detail(wf)
        if err:
            return _json({"ok": False, "error": err})
        pending_runs = self.shared.setdefault("web_pending_runs", {})
        pending_runs[pid] = {
            "values": {
                k: values.get(k)
                for k in ("model", "loras", "width", "height", "steps", "cfg", "sampler_name", "scheduler", "denoise", "trigger_words", "seed")
            },
            "recipe": recipe,
            "family": family.name if family is not None else explicit_family,
            "workflow": tname,
            "slots": slots,
            "drop_nodes": drop_nodes,
            "prompt": prompt,
        }
        # 防止浏览器一直提交试跑导致内存中的任务元数据无限增长。
        if len(pending_runs) > 64:
            for old_pid in list(pending_runs)[:-64]:
                pending_runs.pop(old_pid, None)
        return _json({"ok": True, "prompt_id": pid, "wait_timeout": getattr(self.client, "timeout", 300)})

    async def generate_poll(self) -> Any:
        pid = _query("pid").strip()
        if not pid:
            return _json({"ok": False, "error": "缺少 pid"})
        cache = self.shared.setdefault("web_result_cache", {})
        locks = self.shared.setdefault("web_result_locks", {})
        holder = locks.setdefault(pid, {"lock": asyncio.Lock(), "users": 0})
        holder["users"] += 1
        try:
            async with holder["lock"]:
                saved = cache.get(pid)
                if not saved:
                    history = await asyncio.to_thread(self.store.history_get, pid) or {}
                    if history.get("local_path") and history.get("filename"):
                        local = Path(history["local_path"]).resolve()
                        if local.is_relative_to(self.output_dir.resolve()):
                            saved = {"path": str(local), "filename": history["filename"]}
                if saved:
                    try:
                        content = await asyncio.to_thread(Path(saved["path"]).read_bytes)
                        cache[pid] = saved
                        return _json(self._image_result(saved["filename"], content))
                    except OSError:
                        cache.pop(pid, None)
                result = await self._generate_result(pid)
                path = result.pop("_local_path", None)
                if path:
                    cache[pid] = {"path": path, "filename": result["filename"]}
                return _json(result)
        finally:
            holder["users"] -= 1
            for old_pid in list(cache)[:-64]:
                cache.pop(old_pid, None)
            for old_pid in list(locks):
                if len(locks) <= 128:
                    break
                if not locks[old_pid]["users"]:
                    locks.pop(old_pid, None)


    async def interrupt(self) -> Any:
        body = await _body()
        pid = str(body.get("prompt_id") or "").strip()
        if not pid:
            return _json({"ok": False, "error": "缺少 prompt_id"}, status=400)
        pending_runs = self.shared.get("web_pending_runs") or {}
        if pid not in pending_runs:
            return _json(
                {"ok": False, "error": "该任务不属于当前 WebUI 试跑或已经结束"},
                status=404,
            )
        ok = await self.client.interrupt(prompt_id=pid)
        return _json({"ok": ok, "prompt_id": pid})

    async def list_history(self) -> Any:
        return _json({"ok": True, "items": self.store.list_history(20)})

    async def recipe_from_history(self) -> Any:
        body = await _body()
        pid = str(body.get("prompt_id") or "").strip()
        name = str(body.get("name") or "").strip()
        if not pid or not name:
            return _json({"ok": False, "error": "需要 prompt_id 与 name"})
        try:
            saved = self.store.recipe_from_history(pid, name, str(body.get("description") or ""))
        except ValueError as e:
            return _json({"ok": False, "error": str(e)})
        self._refresh_draw_schema()
        return _json({"ok": True, "recipe": saved})

    async def _generate_result(self, pid: str) -> dict:
        if hasattr(self.client, "wait_for_history"):
            entry, _ = await self.client.wait_for_history(pid, timeout=min(5.0, self.client.timeout))
        else:
            entry = await self.client.get_history_entry(pid)
        if entry is None:
            progress = self.client.execution_progress(pid) if hasattr(self.client, "execution_progress") else {}
            return {"ok": True, "done": False, "progress": progress}
        st = entry.get("status") or {}
        if st.get("status_str") == "error":
            pending_runs = self.shared.get("web_pending_runs") or {}
            pending_runs.pop(pid, None)
            return {"ok": True, "done": True, "error": execution_error_message(st)}
        if st.get("completed") is False:
            return {"ok": True, "done": False}
        images = []
        for node_out in (entry.get("outputs") or {}).values():
            images.extend(node_out.get("images") or [])
        if not images:
            pending_runs = self.shared.get("web_pending_runs") or {}
            pending = pending_runs.pop(pid, {})
            if pending.get("entry") == "graph" and st.get("completed"):
                return {"ok": True, "done": True, "message": "执行完成，无图片输出"}
            return {"ok": True, "done": True, "error": "执行完成但无输出图片"}
        img = next((item for item in images if item.get("type", "output") == "output"), images[-1])
        content = await self.client.download_image(
            img["filename"], img.get("subfolder", ""), image_type=img.get("type", "output")
        )
        if not content:
            return {"ok": True, "done": True, "error": f"图片下载失败: {img['filename']}"}
        try:
            local_path = await asyncio.to_thread(save_image, self.output_dir, img["filename"], content)
        except (ValueError, OSError) as e:
            return {"ok": True, "done": True, "error": f"图片本地保存失败: {e}"}
        pending_runs = self.shared.get("web_pending_runs") or {}
        pending = pending_runs.get(pid) or {}
        recipe = pending.get("recipe") or {}
        values = pending.get("values") or {}
        if pending or self.store.history_get(pid) is None:
            self.store.save_history(
                {
                    "prompt_id": pid,
                    "entry": pending.get("entry") or "recipe",
                    "family": pending.get("family") or recipe.get("family") or "",
                    "recipe": recipe.get("name"),
                    "recipe_id": recipe.get("id"),
                    "workflow": pending.get("workflow") or recipe_template(recipe),
                    "slots": pending.get("slots") or recipe.get("slots") or {},
                    "drop_nodes": pending.get("drop_nodes") or recipe.get("drop_nodes") or [],
                    "prompt": pending.get("prompt") or "",
                    "values": {
                        k: v
                        for k, v in values.items()
                        if v is not None and (v != "" or k == "trigger_words") and (v != [] or k == "loras")
                    },
                    "filename": img["filename"],
                    "local_path": str(local_path),
                }
            )
        pending_runs.pop(pid, None)
        return {**self._image_result(img["filename"], content), "_local_path": str(local_path)}

    @staticmethod
    def _image_result(filename: str, content: bytes) -> dict:
        return {"ok": True, "done": True, "filename": filename,
                "data_url": f"data:{image_media_type(content, filename)};base64," + base64.b64encode(content).decode("ascii")}




def register_web_apis(
    context,
    client: ComfyUIClient,
    builder: WorkflowBuilder,
    store: RecipeStore | None = None,
    output_dir: Path | None = None,
    shared: dict | None = None,
    draw_tool: Any = None,
    recipe_draw_tool: Any = None,
    edit_tool: Any = None,
    families: ModelFamilyRegistry | None = None,
    profiles: WorkflowProfileStore | None = None,
    config_defaults: dict | None = None,
    plugin_config: Any = None,
    auto_trigger_words: bool = False,
    edit_workflows: EditWorkflowRegistry | None = None,
) -> None:
    if store is None or output_dir is None:
        logger.error("[ComfyUIDirect] WebUI 缺少 recipe store，跳过注册")
        return
    api = StudioApi(
        client,
        builder,
        store,
        output_dir,
        shared if shared is not None else {},
        draw_tool,
        recipe_draw_tool,
        edit_tool,
        families,
        profiles,
        config_defaults=config_defaults,
        plugin_config=plugin_config,
        auto_trigger_words=auto_trigger_words,
        edit_workflows=edit_workflows,
    )
    routes = [
        ("/status", api.status, ["GET"], "ComfyUI 状态"),
        ("/workflows", api.list_workflows, ["GET"], "列出工作流"),
        ("/workflow", api.get_workflow, ["GET"], "获取工作流"),
        ("/workflow/nodes", api.get_node_definitions, ["GET"], "获取远端节点定义"),
        ("/workflow/import", api.import_workflow, ["POST"], "导入工作流"),
        ("/workflow/save", api.save_graph, ["POST"], "保存节点画布"),
        ("/workflow/run", api.run_graph, ["POST"], "运行节点画布"),
        ("/workflow/import-history", api.import_from_history, ["POST"], "从 ComfyUI 历史导入"),
        ("/workflow/delete", api.delete_workflow, ["POST"], "删除工作流"),
        ("/workflow/detect", api.detect, ["POST"], "检测槽位"),
        ("/workflow/profile", api.save_workflow_profile, ["POST"], "保存工作流槽位档案"),
        ("/workflow/bind", api.save_workflow_binding, ["POST"], "绑定生图家族或编辑路由"),
        ("/comfy-history", api.comfy_history, ["GET"], "ComfyUI 历史"),
        ("/recipes", api.list_recipes, ["GET"], "列出配方"),
        ("/recipe", api.get_recipe, ["GET"], "获取配方"),
        ("/recipe/save", api.save_recipe, ["POST"], "保存配方"),
        ("/recipe/delete", api.delete_recipe, ["POST"], "删除配方"),
        ("/recipe/default", api.set_default_recipe, ["POST"], "设为默认配方"),
        ("/recipe/from-history", api.recipe_from_history, ["POST"], "历史存成配方"),
        ("/generate", api.generate, ["POST"], "按配方试跑"),
        ("/generate", api.generate_poll, ["GET"], "轮询试跑结果"),
        ("/generate/interrupt", api.interrupt, ["POST"], "中断试跑"),
        ("/history", api.list_history, ["GET"], "插件生成历史"),
    ]
    for path, handler, methods, desc in routes:
        context.register_web_api(f"/{PLUGIN_NAME}{path}", handler, methods, desc)
