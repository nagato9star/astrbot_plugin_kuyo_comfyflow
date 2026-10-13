"""Shared validation for workflows edited in the WebUI graph canvas."""

from __future__ import annotations

import re
from typing import Any


_EXECUTION_ID = re.compile(r"^\d+(?::\d+)*$")


def validate_api_workflow(workflow: Any) -> dict[str, dict]:
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("工作流为空或格式无效")
    if len(workflow) > 2000:
        raise ValueError("工作流节点过多")
    nodes: dict[str, dict] = {}
    for raw_id, node in workflow.items():
        node_id = str(raw_id)
        if not _EXECUTION_ID.fullmatch(node_id) or not isinstance(node, dict):
            raise ValueError(f"节点 ID 无效: {node_id}")
        if not isinstance(node.get("class_type"), str) or not node["class_type"]:
            raise ValueError(f"节点 {node_id} 缺少 class_type")
        if not isinstance(node.get("inputs"), dict):
            raise ValueError(f"节点 {node_id} 缺少 inputs")
        nodes[node_id] = node
    for node_id, node in nodes.items():
        for field, value in node["inputs"].items():
            if not isinstance(value, list) or len(value) != 2:
                continue
            source, output = value
            source_id = str(source)
            if not _EXECUTION_ID.fullmatch(source_id) or not isinstance(output, int):
                continue
            if source_id not in nodes or output < 0:
                raise ValueError(f"节点 {node_id} 的 {field} 连接无效: {value}")
    return nodes


def validate_ui_snapshot(ui: Any, workflow: dict[str, dict]) -> dict:
    if not isinstance(ui, dict) or not isinstance(ui.get("nodes"), list) or not isinstance(ui.get("links"), list):
        raise ValueError("画布 JSON 缺少 nodes 或 links")
    by_id: dict[str, dict] = {}
    for node in ui["nodes"]:
        if not isinstance(node, dict) or not str(node.get("id", "")).isdecimal():
            raise ValueError("画布中存在无效节点")
        node_id = str(node["id"])
        if node_id in by_id:
            raise ValueError(f"画布节点 ID 重复: {node_id}")
        by_id[node_id] = node
    for node_id, node in workflow.items():
        canvas_node = by_id.get(node_id)
        if ":" in node_id:
            root_id = node_id.split(":", 1)[0]
            if root_id not in by_id:
                raise ValueError(f"画布缺少子图实例 {root_id}")
        elif canvas_node is None or canvas_node.get("type") != node["class_type"]:
            raise ValueError(f"画布与执行工作流的节点 {node_id} 不一致")
        for field, value in node["inputs"].items():
            if not isinstance(value, list) or len(value) != 2 or not _EXECUTION_ID.fullmatch(str(value[0])) or not isinstance(value[1], int):
                continue
            source = by_id.get(str(value[0]))
            outputs = source.get("outputs") if source else None
            if isinstance(outputs, list) and value[1] >= len(outputs):
                raise ValueError(f"节点 {node_id} 的 {field} 引用了不存在的输出端口 {value[1]}")
    return ui
