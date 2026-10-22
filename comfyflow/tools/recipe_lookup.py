"""只读配方目录，供基础工具模式选择已有方案。"""

import json

from ..resources.catalog import canonical_family, page_number
from ..storage.recipes import RecipeStore
from .common import _as_bool


def lookup_recipes(store: RecipeStore | None, query: str, kwargs: dict) -> str:
    if store is None:
        return "查询失败：配方存储尚未初始化。"
    family = canonical_family(str(kwargs.get("model_family") or "").strip())
    query = query.casefold()
    rows = [row for row in store.list()
            if (not family or canonical_family(row.get("family") or "") == family)
            and (not query or query in " ".join(str(row.get(key) or "") for key in
                 ("id", "name", "description", "family", "model", "loras")).casefold())]
    limit = max(1, page_number(kwargs.get("limit"), 5, 10))
    offset = page_number(kwargs.get("offset"))
    default = store.default()
    default_id = default.get("id") if default else None
    lines = [f"【已保存配方】共 {len(rows)} 项"]
    detailed = _as_bool(kwargs.get("detail"))
    for row in rows[offset:offset + limit]:
        marker = "（默认）" if row["id"] == default_id else ""
        lines.append(f"{row['name']}{marker} [家族={row['family'] or '未指定'}]")
        if row.get("description"):
            lines.append("用途：" + " ".join(row["description"].split())[:160])
        if detailed:
            recipe = store.get(row["id"])
            if recipe:
                values = {key: value for key, value in (recipe.get("defaults") or {}).items()
                          if key != "prompt"}
                lines.append("参数：" + json.dumps(values, ensure_ascii=False))
    if offset + limit < len(rows):
        lines.append(f"next_offset={offset + limit}")
    if not rows[offset:offset + limit]:
        lines.append("无匹配配方。可调整 query / model_family，或用 comfyui_draw 自由生成。")
    else:
        lines.append("选中后用 comfyui_recipe_draw(recipe=配方名, prompt=本次内容)；省略 recipe 使用默认配方。")
    return "\n".join(lines)
