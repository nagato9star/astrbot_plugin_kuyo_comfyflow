"""角色查询的来源选择、作品取样与系列检索。"""

from ..clients.external import DanbooruClient, GelbooruClient


async def lookup_booru(danbooru, gelbooru, kind, query, source="danbooru", *,
                      detail=False, sample_limit=30):
    """查询规范词；详细模式附带作品标签统计，并标明实际来源。"""
    try:
        sample_limit = min(max(int(sample_limit), 1), 50) if detail else 0
    except (TypeError, ValueError):
        sample_limit = 30 if detail else 0
    clients = {"danbooru": danbooru, "gelbooru": gelbooru}
    fallback = "gelbooru" if source == "danbooru" else "danbooru"
    data = None
    for used_source in (source, fallback):
        client = clients[used_source]
        if client is None:
            continue
        method = client.search_artist if kind == "artist" else client.search_character
        # Gelbooru 通过作品识别规范名，简短查询也需要取样。
        request_limit = 20 if used_source == "gelbooru" and not detail else sample_limit
        data = await method(query, request_limit)
        if data is not None:
            break
    if data is None:
        return f"未找到{'画师' if kind == 'artist' else '角色'}：{query}"

    name = data.get(kind) or query
    trigger = "@" + name if kind == "artist" else name
    lines = [f"【{'画师' if kind == 'artist' else '角色'} {trigger}】来源 {used_source}"]
    aliases = (data.get("aliases") or [])[:8]
    if aliases:
        lines.append("别名: " + ", ".join(aliases))
    lines.append("触发词: " + trigger)
    if detail:
        posts = data.get("posts") or []
        if used_source == "gelbooru":
            posts = GelbooruClient.normalize_posts(posts)
        tags = DanbooruClient.aggregate_tags(posts)
        lines.append("常用 tag（作品取样统计）:")
        lines.append("  " + ", ".join(f"{tag}({count})" for tag, count in tags)
                     if tags else "  (无样本)")
        if data.get("url"):
            lines.append("参考: " + data["url"])
    return "\n".join(lines)


async def lookup_animadex(client, kind, query, page=1):
    """保留 AnimaDex 的角色、画师与作品系列分页查询。"""
    if client is None:
        return "查询失败：AnimaDex 客户端未配置。"
    try:
        page = max(int(page), 1)
    except (TypeError, ValueError):
        page = 1
    method = {
        "character": client.search_characters,
        "artist": client.search_artists,
        "copyright": client.search_copyrights,
    }[kind]
    text = await method(query, page=page)
    if not text:
        return "查询失败：无法连接 AnimaDex 服务。"
    return f"【AnimaDex】{kind}: {query}\n{text.strip()}"
