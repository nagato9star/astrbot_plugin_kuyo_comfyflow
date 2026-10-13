"""API 格式工作流 → UI 格式快照转换器。

背景：插件通过 /prompt API 裸提交时，ComfyUI 保存的 PNG 只有 `prompt`
chunk（API 格式），没有 `workflow` chunk（UI 格式）。前端拖图加载时，
旧版前端/旧版 rgthree 扩展无法从 API 格式还原 Power Lora Loader 的
动态 lora 槽，导致用户在界面里看到 lora 为空。

修复：提交时同步携带 extra_data.extra_pnginfo.workflow（UI 快照），
PNG 即内嵌标准 workflow 元数据，任何版本前端拖图都能完整还原。

转换规则：
- API dict: {node_id: {class_type, inputs: {field: value}}}
- 连线值形如 [source_id, output_slot]，转成 UI links + 端口 link 引用
- widget 值按 object_info 定义顺序排入 widgets_values；
  已转成连线的 widget 仍保留 null 占位，带 control_after_generate 的
  INT 字段还需追加 control 值（"fixed"）
- rgthree Power Lora Loader 的动态 lora_N 字典值 → widgets_values
  = [{on, lora, strength}, ...]（与其前端 configure() 逻辑对齐）
"""

from __future__ import annotations

import json
import re
import logging

logger = logging.getLogger("[ComfyUIDirect]")

# 常见连线端口类型（API 值为 [node_id, slot] 二元组）
_LINK_RE = re.compile(r"lora_\d+$")
_DEFAULT_SIZE = [300.0, 130.0]
# XB_ToolBox legacy aliases render a seed control widget in the UI, but their
# /object_info INT spec omits control_after_generate. Keep that widget position.
_SEED_CONTROL_COMPAT = {
    ("XB_ROCmKSampler", "seed"),
    ("XB_ROCmKSamplerAdvanced", "noise_seed"),
}


def _num(v) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.lstrip("-").isdigit():
        return int(v)
    return None


def _ui_node_id_map(api: dict) -> dict[str, int]:
    """Assign numeric UI ids to API nodes, including subgraph ids like ``459:474``."""
    result: dict[str, int] = {}
    used: set[int] = set()
    for raw_id in api:
        key = str(raw_id)
        numeric_id = _num(raw_id)
        if numeric_id is not None and numeric_id >= 0 and numeric_id not in used:
            result[key] = numeric_id
            used.add(numeric_id)
    for node in api.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            if isinstance(value, list) and len(value) == 2 and _num(value[1]) is not None:
                referenced_id = _num(value[0])
                if referenced_id is not None and referenced_id >= 0:
                    used.add(referenced_id)

    next_id = max(used, default=0) + 1
    for raw_id in api:
        key = str(raw_id)
        if key in result:
            continue
        while next_id in used:
            next_id += 1
        result[key] = next_id
        used.add(next_id)
        next_id += 1
    return result


def _ui_link(value, node_ids: dict[str, int]) -> tuple[int, int] | None:
    """Map an API node link to numeric UI node and output ids."""
    if not isinstance(value, list) or len(value) != 2:
        return None
    source_id = node_ids.get(str(value[0]))
    if source_id is None:
        source_id = _num(value[0])
    output_id = _num(value[1])
    if source_id is None or output_id is None:
        return None
    return source_id, output_id


def _needs_control_after_generate(field_type, class_type: str = "", field: str = "") -> bool:
    if isinstance(field_type, list) and len(field_type) >= 2:
        cfg = field_type[1]
        if isinstance(cfg, dict) and "control_after_generate" in cfg:
            return True
    return (class_type, field) in _SEED_CONTROL_COMPAT


def _is_widget_spec(field_type) -> bool:
    if not isinstance(field_type, list) or not field_type:
        return False
    head = field_type[0]
    if isinstance(head, list):
        return True
    return head in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")


def api_to_ui(api: dict, object_info: dict | None = None) -> dict:
    """把 ComfyUI API 格式工作流转换为 UI 格式（extra_pnginfo.workflow 用）。"""
    nodes: list[dict] = []
    links: list[list] = []
    node_ids = _ui_node_id_map(api)
    link_id = 1
    order = 0

    for nid_s, node in api.items():
        nid = node_ids.get(str(nid_s))
        if nid is None or not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        info = (object_info or {}).get(class_type, {})
        input_def = info.get("input", {})
        out_types = info.get("output", [])
        out_names = info.get("output_name", [])

        in_defs = {}
        input_order = info.get("input_order") or {}
        ordered_fields: list[str] = []
        for group in ("required", "optional"):
            block = input_def.get(group) or {}
            declared = input_order.get(group) if isinstance(input_order, dict) else None
            names = list(declared) if isinstance(declared, list) else []
            names.extend(name for name in block if name not in names)
            for k in names:
                if k not in block:
                    continue
                v = block[k]
                in_defs[k] = v
                if k in (node.get("inputs") or {}):
                    ordered_fields.append(k)
        ordered_fields.extend(k for k in (node.get("inputs") or {}) if k not in ordered_fields)

        ui_inputs: list[dict] = []
        ui_outputs: list[dict] = []
        widgets_values: list = []
        rgthree_loras: list[dict] = []
        is_rgthree_pll = "Power Lora Loader" in class_type

        # 输出端口
        for i, t in enumerate(out_types):
            ui_outputs.append(
                {
                    "name": out_names[i] if i < len(out_names) else str(t),
                    "type": t,
                    "links": [],
                    "slot_index": i,
                }
            )

        for field in ordered_fields:
            value = node["inputs"][field]
            if is_rgthree_pll and _LINK_RE.match(field) and isinstance(value, dict):
                rgthree_loras.append(value)
                continue
            mapped_link = _ui_link(value, node_ids)
            if mapped_link is not None:
                src, slot = mapped_link
                ltype = in_defs.get(field, [None])[0] if field in in_defs else None
                if not isinstance(ltype, str):
                    ltype = "*"
                slot_idx = len(ui_inputs)
                ui_inputs.append({"name": field, "type": ltype, "link": link_id})
                links.append([link_id, src, slot, nid, slot_idx, ltype])
                link_id += 1
                field_type = in_defs.get(field)
                if _is_widget_spec(field_type):
                    widgets_values.append(None)
                    if _needs_control_after_generate(field_type, class_type, field):
                        widgets_values.append("fixed")
                continue
            if field in in_defs and _needs_control_after_generate(in_defs[field], class_type, field):
                widgets_values.extend([value, "fixed"])
            else:
                widgets_values.append(value)

        if is_rgthree_pll:
            # rgthree configure() 期望 widgets_values 为 lora 对象列表
            widgets_values = rgthree_loras

        pos = [(order % 5) * 340.0, (order // 5) * 300.0]
        nodes.append(
            {
                "id": nid,
                "type": class_type,
                "pos": pos,
                "size": list(_DEFAULT_SIZE),
                "flags": {},
                "order": order,
                "mode": 0,
                "inputs": ui_inputs,
                "outputs": ui_outputs,
                "properties": {"Node name for S&R": class_type},
                "widgets_values": widgets_values,
            }
        )
        order += 1

    # 回填源节点输出端口的 links 引用
    by_id = {n["id"]: n for n in nodes}
    for lk in links:
        _, src, slot, _, _, ltype = lk
        src_node = by_id.get(src)
        if src_node and slot < len(src_node["outputs"]):
            src_node["outputs"][slot].setdefault("links", []).append(lk[0])

    return {
        "last_node_id": max((n["id"] for n in nodes), default=0),
        "last_link_id": link_id - 1,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def build_extra_pnginfo(api: dict, object_info: dict | None = None) -> dict | None:
    """构造 /prompt 的 extra_data.extra_pnginfo；失败返回 None（裸提交兜底）。"""
    try:
        return {"workflow": api_to_ui(api, object_info)}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[ComfyUIDirect] UI 快照构造失败，跳过元数据: {e}")
        return None


def _selftest():  # pragma: no cover
    import sys

    api = json.load(open(sys.argv[1]))
    obj = None
    try:
        obj = json.load(open("/tmp/objinfo.json"))
    except Exception:
        pass
    ui = api_to_ui(api, obj)
    print(json.dumps(ui, ensure_ascii=False)[:1500])
    print("nodes:", len(ui["nodes"]), "links:", len(ui["links"]))


if __name__ == "__main__":
    _selftest()
