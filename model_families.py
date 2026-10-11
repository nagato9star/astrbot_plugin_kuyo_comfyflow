"""模型家族配置与工作流槽位档案。

模型家族负责把 LLM 选择的家族名路由到工作流。节点槽位属于工作流档案，
配方只保存可复用的生成参数并引用模型家族。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from astrbot.api import logger

from recipe_store import model_family, recipe_template
from resource_catalog import canonical_family
from slot_mapping import (
    ANIMA_DROP_NODES,
    detect_slots,
    looks_like_anima,
    merge_slots,
    parse_node_option,
    slots_from_config,
)


@dataclass(frozen=True)
class ModelFamily:
    """一条可由 LLM 选择的模型家族路由。"""

    name: str
    workflow: str
    edit_workflow: str = ""
    prompt_style: str = "auto"
    description: str = ""

    def public(self) -> dict[str, str]:
        return asdict(self)


class ModelFamilyRegistry:
    """解析 ``model_families`` template_list 并提供稳定的查找接口。"""

    def __init__(self, raw: Any, default_workflow: str = "anima-v3", edit_raw: Any = None) -> None:
        self._items: dict[str, ModelFamily] = {}
        for row in self._rows(raw):
            name = str(row.get("name") or row.get("family") or "").strip()
            workflow = str(row.get("workflow") or "").strip()
            if not name or not workflow:
                logger.warning("[ComfyUIDirect] 忽略缺少家族名或工作流的模型家族配置")
                continue
            key = name.casefold()
            if key in self._items:
                logger.warning(f"[ComfyUIDirect] 模型家族名重复，保留第一条: {name}")
                continue
            style = str(row.get("prompt_style") or "auto").strip().casefold()
            if style not in {"auto", "danbooru", "natural"}:
                style = "auto"
            self._items[key] = ModelFamily(
                name=name,
                workflow=workflow,
                edit_workflow=str(row.get("edit_workflow") or "").strip(),
                prompt_style=style,
                description=str(row.get("description") or "").strip(),
            )

        if not self._items:
            workflow = str(default_workflow or "anima-v3").strip() or "anima-v3"
            guessed = canonical_family(model_family(workflow)) or "default"
            self._items[guessed.casefold()] = ModelFamily(
                name=guessed,
                workflow=workflow,
                prompt_style="danbooru" if guessed == "anima" else "auto",
                description="由旧版 default_workflow 自动兼容",
            )

        seen_edits: set[str] = set()
        for row in self._rows(edit_raw):
            name = str(row.get("model_family") or row.get("family") or row.get("name") or "").strip()
            if not name or "workflow" not in row:
                continue
            workflow = str(row.get("workflow") or "").strip()
            key = name.casefold()
            if key in seen_edits:
                logger.warning(f"[ComfyUIDirect] 编辑图家族重复，保留第一条: {name}")
                continue
            seen_edits.add(key)
            family = self._items.get(key)
            if family is None:
                logger.warning(f"[ComfyUIDirect] 编辑图家族「{name}」没有对应生图家族，已忽略")
                continue
            self._items[key] = replace(family, edit_workflow=workflow)

    @staticmethod
    def _rows(raw: Any) -> list[dict]:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError):
                return []
        if isinstance(raw, list):
            return [row for row in raw if isinstance(row, dict)]
        if isinstance(raw, dict):
            rows = []
            for name, value in raw.items():
                if isinstance(value, str):
                    rows.append({"name": name, "workflow": value})
                elif isinstance(value, dict):
                    rows.append({"name": name, **value})
            return rows
        return []

    def list(self) -> list[dict[str, str]]:
        return [item.public() for item in self._items.values()]

    def names(self) -> list[str]:
        return [item.name for item in self._items.values()]

    def editable_names(self) -> list[str]:
        return [item.name for item in self._items.values() if item.edit_workflow]

    def reconfigure(self, raw: Any, default_workflow: str, edit_raw: Any = None) -> None:
        """Refresh the same registry object held by Bot tools and Workflow Studio."""
        updated = ModelFamilyRegistry(raw, default_workflow, edit_raw)
        self._items = updated._items

    def first(self) -> ModelFamily:
        return next(iter(self._items.values()))

    def get(self, name: str) -> ModelFamily | None:
        return self._items.get(str(name or "").strip().casefold())

    def by_workflow(self, workflow: str) -> ModelFamily | None:
        wanted = str(workflow or "").strip().casefold()
        if not wanted:
            return None
        return next(
            (item for item in self._items.values() if item.workflow.casefold() == wanted),
            None,
        )

    def resolve_recipe(self, recipe: dict) -> ModelFamily | None:
        """把新旧配方解析到家族；旧配方可从底模或历史工作流迁移。"""
        explicit = str((recipe or {}).get("family") or "").strip()
        if explicit:
            return self.get(explicit)
        found = self.by_workflow(recipe_template(recipe or {}))
        if found:
            return found
        defaults = (recipe or {}).get("defaults") or {}
        guessed = canonical_family(model_family(str(defaults.get("model") or "")))
        if guessed:
            found = self.get(guessed)
            if found:
                return found
        return None

    def workflow_references(self) -> dict[str, list[str]]:
        refs: dict[str, list[str]] = {}
        for item in self._items.values():
            refs.setdefault(item.workflow, []).append(item.name)
            if item.edit_workflow:
                refs.setdefault(item.edit_workflow, []).append(f"{item.name}（编辑）")
        return refs

    def catalog(self) -> str:
        parts = []
        for item in self._items.values():
            detail = " ".join(item.description.split())[:100]
            parts.append(f"{item.name}（prompt_style={item.prompt_style}" + (f"；{detail}" if detail else "") + "）")
        return "；".join(parts)


@dataclass(frozen=True)
class EditWorkflow:
    """独立的图片编辑路由，不依赖生图模型家族。"""

    name: str
    workflow: str
    description: str = ""

    def public(self) -> dict[str, str]:
        return asdict(self)


class EditWorkflowRegistry:
    """解析独立编辑路由，并读取旧的家族绑定作为迁移兼容。"""

    def __init__(
        self,
        raw: Any = None,
        legacy_edit_raw: Any = None,
        legacy_families_raw: Any = None,
    ) -> None:
        self._items: dict[str, EditWorkflow] = {}
        # Older edit_families rows used the generation family name as route name.
        for row in ModelFamilyRegistry._rows(legacy_edit_raw):
            name = str(
                row.get("name") or row.get("model_family") or row.get("family") or ""
            ).strip()
            workflow = str(row.get("workflow") or "").strip()
            if name:
                self._items.setdefault(
                    name.casefold(),
                    EditWorkflow(name, workflow, str(row.get("description") or "").strip()),
                )
        # The original configuration put edit_workflow directly on model_families.
        for row in ModelFamilyRegistry._rows(legacy_families_raw):
            name = str(row.get("name") or row.get("family") or "").strip()
            workflow = str(row.get("edit_workflow") or "").strip()
            if name and workflow:
                self._items.setdefault(name.casefold(), EditWorkflow(name, workflow))
        # New independent edit_workflows always take precedence over legacy values.
        for row in ModelFamilyRegistry._rows(raw):
            name = str(row.get("name") or row.get("route") or "").strip()
            if not name:
                continue
            self._items[name.casefold()] = EditWorkflow(
                name=name,
                workflow=str(row.get("workflow") or "").strip(),
                description=str(row.get("description") or "").strip(),
            )

    def list(self) -> list[dict[str, str]]:
        return [item.public() for item in self._items.values()]

    def names(self) -> list[str]:
        return [item.name for item in self._items.values() if item.workflow]

    def get(self, name: str) -> EditWorkflow | None:
        return self._items.get(str(name or "").strip().casefold())

    def by_workflow(self, workflow: str) -> EditWorkflow | None:
        wanted = str(workflow or "").strip().casefold()
        if not wanted:
            return None
        return next(
            (item for item in self._items.values() if item.workflow.casefold() == wanted),
            None,
        )

    def set(self, name: str, workflow: str, description: str | None = None) -> EditWorkflow:
        route_name = str(name or "").strip()
        if not route_name:
            raise ValueError("编辑路由名不能为空")
        key = route_name.casefold()
        previous = self._items.get(key)
        route = EditWorkflow(
            name=previous.name if previous else route_name,
            workflow=str(workflow or "").strip(),
            description=(
                str(description).strip()
                if description is not None
                else (previous.description if previous else "")
            ),
        )
        self._items[key] = route
        return route

    def reconfigure(
        self,
        raw: Any = None,
        legacy_edit_raw: Any = None,
        legacy_families_raw: Any = None,
    ) -> None:
        updated = EditWorkflowRegistry(raw, legacy_edit_raw, legacy_families_raw)
        self._items = updated._items

    def workflow_references(self) -> dict[str, list[str]]:
        refs: dict[str, list[str]] = {}
        for item in self._items.values():
            if item.workflow:
                refs.setdefault(item.workflow, []).append(item.name)
        return refs


class WorkflowProfileStore:
    """持久化工作流的槽位映射，使多份配方共享同一份节点定义。"""

    def __init__(self, data_dir: Path, configured: Any = None) -> None:
        self.path = data_dir / "workflow_profiles.json"
        self._configured = self._parse_configured(configured)

    @staticmethod
    def _parse_configured(raw: Any) -> dict[str, dict[str, Any]]:
        """解析配置页按工作流填写的完整槽位映射。"""
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError):
                return {}
        if isinstance(raw, list):
            rows = [row for row in raw if isinstance(row, dict)]
        elif isinstance(raw, dict) and raw.get("workflow"):
            rows = [raw]
        elif isinstance(raw, dict):
            rows = [
                {"workflow": workflow, **slots}
                for workflow, slots in raw.items()
                if isinstance(slots, dict)
            ]
        else:
            rows = []

        configured: dict[str, dict[str, Any]] = {}
        for row in rows:
            workflow = str(row.get("workflow") or "").strip()
            if not workflow:
                continue
            key = workflow.casefold()
            if key in configured:
                logger.warning(f"[ComfyUIDirect] 工作流节点映射重复，保留第一条: {workflow}")
                continue
            configured[key] = slots_from_config(row)
        return configured

    def configured(self, workflow: str) -> dict[str, Any] | None:
        slots = self._configured.get(str(workflow or "").strip().casefold())
        return dict(slots) if slots is not None else None

    @staticmethod
    def _normalize_slots(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        out: dict[str, Any] = {}
        for role, spec in raw.items():
            if role == "source_images":
                if isinstance(spec, str):
                    specs = [item.strip() for item in spec.split(",") if item.strip()]
                elif isinstance(spec, list):
                    specs = spec
                elif isinstance(spec, dict):
                    specs = [spec]
                elif spec in (None, ""):
                    specs = []
                else:
                    raise ValueError("source_images 必须是有序节点列表")
                nodes = []
                seen = set()
                for item in specs:
                    if isinstance(item, str):
                        node = parse_node_option(item)
                        if node:
                            nodes.append({"node": node, "field": "image", "mode": "replace"})
                    elif isinstance(item, dict):
                        node = parse_node_option(item.get("node"))
                        if node:
                            normalized_item = dict(item)
                            normalized_item["node"] = node
                            nodes.append(normalized_item)
                    else:
                        raise ValueError("参考图映射必须包含 node 字段")
                    if not node:
                        raise ValueError("参考图映射存在空节点，请选择节点或删除该行")
                    if node in seen:
                        raise ValueError(f"参考图节点 {node} 重复映射，请为每个输入选择不同节点")
                    seen.add(node)
                out[str(role)] = nodes
                continue
            if isinstance(spec, str):
                node = parse_node_option(spec)
                if node:
                    out[str(role)] = {"node": node}
                continue
            if not isinstance(spec, dict):
                continue
            node = parse_node_option(spec.get("node"))
            if not node:
                continue
            normalized = dict(spec)
            normalized["node"] = node
            out[str(role)] = normalized
        return out

    def _read(self) -> dict[str, dict]:
        if not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"[ComfyUIDirect] 读取工作流槽位档案失败: {e}")
            return {}
        profiles = payload.get("profiles") if isinstance(payload, dict) else None
        if not isinstance(profiles, dict):
            profiles = payload if isinstance(payload, dict) else {}
        return {str(k): v for k, v in profiles.items() if isinstance(v, dict)}

    def _write(self, profiles: dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "profiles": profiles}
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def get(self, workflow: str) -> dict | None:
        data = self._read().get(str(workflow or "").strip())
        if not data:
            return None
        return {
            "workflow": str(workflow).strip(),
            "slots": self._normalize_slots(data.get("slots")),
            "drop_nodes": [str(x) for x in (data.get("drop_nodes") or []) if str(x)],
            "updated_at": data.get("updated_at"),
        }

    def save(
        self,
        workflow: str,
        slots: Any,
        drop_nodes: Any = None,
    ) -> dict:
        name = str(workflow or "").strip()
        if not name:
            raise ValueError("工作流名不能为空")
        normalized = self._normalize_slots(slots)
        profiles = self._read()
        profiles[name] = {
            "slots": normalized,
            "drop_nodes": [str(x) for x in (drop_nodes or []) if str(x)],
            "updated_at": int(time.time()),
        }
        self._write(profiles)
        return self.get(name) or {"workflow": name, "slots": normalized, "drop_nodes": []}

    def effective(self, workflow: str, wf: dict, object_info: dict | None = None) -> dict:
        """配置页映射优先，其次工作台档案，最后采用自动检测结果。"""
        saved = self.get(workflow)
        configured = self.configured(workflow)
        if configured is not None:
            slots = configured
            drop_nodes = (
                list(saved.get("drop_nodes") or [])
                if saved is not None
                else list(ANIMA_DROP_NODES) if looks_like_anima(wf) else []
            )
            source = "config"
        elif saved:
            slots = saved.get("slots") or {}
            drop_nodes = list(saved.get("drop_nodes") or [])
            source = "profile"
        else:
            slots = detect_slots(wf, object_info)
            drop_nodes = list(ANIMA_DROP_NODES) if looks_like_anima(wf) else []
            source = "detected"
        return {
            "workflow": workflow,
            "slots": slots,
            "drop_nodes": drop_nodes,
            "source": source,
        }

    def ensure(
        self,
        workflow: str,
        wf: dict,
        configured_slots: Any = None,
    ) -> dict:
        if self.configured(workflow) is not None:
            return self.effective(workflow, wf)
        current = self.get(workflow)
        if current:
            return current
        slots = merge_slots(detect_slots(wf), self._normalize_slots(configured_slots))
        drop_nodes = list(ANIMA_DROP_NODES) if looks_like_anima(wf) else []
        return self.save(workflow, slots, drop_nodes)

    def import_legacy_recipe(self, recipe: dict) -> bool:
        """首次升级时把旧配方里的槽位复制到工作流档案，原配方保持可回退。"""
        workflow = recipe_template(recipe or {})
        slots = (recipe or {}).get("slots") or {}
        if not workflow or not slots.get("prompt") or self.get(workflow):
            return False
        self.save(workflow, slots, (recipe or {}).get("drop_nodes") or [])
        return True

    def delete(self, workflow: str) -> None:
        name = str(workflow or "").strip()
        profiles = self._read()
        if name in profiles:
            profiles.pop(name, None)
            self._write(profiles)
