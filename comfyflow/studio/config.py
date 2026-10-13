"""Populate AstrBot's live plugin schema from local workflow and recipe data."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from ..workflows.slots import SLOT_ROLES, node_matches_slot, node_options_for_slot, parse_node_option


_WORKFLOW_TEMPLATE_PREFIX = "local_workflow_"


def _rows(value: Any) -> list[dict]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _select_options(meta: dict, available: list[str], selected: list[str], *, optional: bool = False) -> None:
    names = list(dict.fromkeys(name.strip() for name in available if isinstance(name, str) and name.strip()))
    if not names:
        meta.pop("options", None)
        meta.pop("labels", None)
        return
    choices = ([""] if optional else []) + names
    labels = (["不选择"] if optional else []) + names.copy()
    for value in selected:
        name = str(value or "").strip()
        if name and name not in choices:
            choices.append(name)
            labels.append(f"{name}（当前配置，未找到本地文件）")
    meta["options"] = choices
    meta["labels"] = labels


def refresh_config_options(config: Any, builder: Any, store: Any) -> None:
    """Mutate only schema presentation; never rewrite persisted config values."""
    schema = getattr(config, "schema", None)
    if not isinstance(schema, dict):
        return
    templates = builder.list_templates()
    workflows = sorted({str(row.get("name") or "") for row in templates if row.get("name")})
    generation = _rows(config.get("model_families"))
    editing = _rows(config.get("edit_families"))
    edit_workflows = _rows(config.get("edit_workflows"))
    mappings = _rows(config.get("workflow_node_mappings"))
    family_names = sorted({str(row.get("name") or "").strip() for row in generation if row.get("name")})

    family_items = schema["model_families"]["templates"]["family"]["items"]
    edit_items = schema["edit_families"]["templates"]["edit_family"]["items"]
    edit_workflow_items = schema["edit_workflows"]["templates"]["edit_workflow"]["items"]
    mapping_items = schema["workflow_node_mappings"]["templates"]["mapping"]["items"]
    _select_options(family_items["workflow"], workflows, [r.get("workflow") for r in generation])
    _select_options(edit_items["workflow"], workflows, [r.get("workflow") for r in editing], optional=True)
    _select_options(
        edit_workflow_items["workflow"],
        workflows,
        [r.get("workflow") for r in edit_workflows],
        optional=True,
    )
    _select_options(mapping_items["workflow"], workflows, [r.get("workflow") for r in mappings])

    mapping_templates = schema["workflow_node_mappings"]["templates"]
    for key in list(mapping_templates):
        if key.startswith(_WORKFLOW_TEMPLATE_PREFIX):
            mapping_templates.pop(key)
    for name in workflows:
        try:
            workflow = builder.load_template(name)
        except (OSError, ValueError):
            continue
        template_key = _WORKFLOW_TEMPLATE_PREFIX + hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
        items = {"workflow": {"type": "string", "default": name, "invisible": True}}
        for role, _label in SLOT_ROLES:
            if role not in mapping_items:
                continue
            item = copy.deepcopy(mapping_items[role])
            if role == "source_images":
                item["hint"] = str(item.get("hint") or "") + "；多个节点 ID 用逗号分隔，通常优先在 Workflow Studio 多选"
                items[role] = item
                continue
            selected = {
                parse_node_option(row.get(role)) for row in mappings
                if str(row.get("workflow") or "").strip() == name
            }
            candidate_ids = {
                str(nid) for nid, node in workflow.items()
                if node_matches_slot(node, role)
            }
            options = [""] + [
                label for label in node_options_for_slot(workflow, role)[1:]
                if parse_node_option(label) in candidate_ids | selected
            ]
            options.extend(f"{nid} — 节点不存在" for nid in sorted(selected) if nid and nid not in workflow)
            item["options"] = options
            item["labels"] = ["不映射", *options[1:]]
            item["hint"] = str(item.get("hint") or "") + "；无候选节点时可用通用手填条目或工作台选择"
            items[role] = item
        mapping_templates[template_key] = {
            "name": f"{name} 的节点映射",
            "hint": "节点来自本地工作流；留空槽位保持未映射",
            "items": items,
        }

    _select_options(edit_items["model_family"], family_names, [r.get("model_family") for r in editing])
    if "default_workflow" in schema:
        _select_options(schema["default_workflow"], workflows, [config.get("default_workflow")])
    if "default_recipe" in schema:
        _select_options(schema["default_recipe"], store.names(), [config.get("default_recipe")])
