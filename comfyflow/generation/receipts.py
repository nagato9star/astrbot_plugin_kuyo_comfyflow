"""区分用户可读的任务消息与仅用于管理的内部编号。"""

import re


_TASK_ID = re.compile(r"(?<![A-Za-z0-9_])C\d{6,}(?![A-Za-z0-9_])", re.IGNORECASE)
INTERNAL_ID_NOTICE = "编号仅为内部标记，不用向用户声明或复述。"


def _summary(value: str, limit: int) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[:limit] + "…"


def task_reference(job: dict, *, include_notice: bool = True) -> str:
    """为模型提供任务关联信息，不包含本地缓存路径。"""
    parts = [INTERNAL_ID_NOTICE] if include_notice else []
    if job.get("id"):
        parts.append(f"task_id={job['id']}")
    if job.get("prompt_id"):
        parts.append(f"prompt_id={job['prompt_id']}")
    if job.get("source_request"):
        parts.append("原请求：" + _summary(job["source_request"], 160))
    history = job.get("history") or {}
    for key, label in (("entry", "操作"), ("family", "模型家族"), ("recipe", "配方"),
                       ("workflow", "工作流")):
        if history.get(key):
            parts.append(f"{label}：{history[key]}")
    if history.get("prompt") and history["prompt"] != job.get("source_request"):
        parts.append("生成指令：" + _summary(history["prompt"], 240))
    return "\n".join(parts)


def hide_task_ids(text: str, identifiers=()) -> str:
    """只清除插件任务号及已知后端编号，保留种子、数量和节点编号。"""
    text = text.replace(INTERNAL_ID_NOTICE, "")
    text = _TASK_ID.sub("", text)
    for identifier in identifiers:
        if identifier:
            text = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(identifier) + r"(?![A-Za-z0-9_])", "", text)
    text = re.sub(r"[（(]\s*(?:(?:任务|队列)?(?:编号|ID)|prompt_id|task_id)?\s*[:：#=]?\s*[）)]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:任务|队列)?(?:编号|ID)\s*[:：#]\s*(?=[，。；\n]|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:prompt_id|task_id)\s*=\s*(?=[，。；\n]|$)", "", text)
    text = re.sub(r"[ \t]+([，。；])", r"\1", text)
    return text.strip()


def mark_task_reply(context, shared: dict, *, identifiers=()) -> None:
    """给本次工具调用所属事件标记编号策略，不改变其他会话的回复。"""
    event = getattr(getattr(context, "context", None), "event", None)
    if event is None or not hasattr(event, "set_extra"):
        return
    event.set_extra("comfyui_task_reply", True)
    queue = shared.get("generation_queue")
    identifiers = list(identifiers) + ([job.get("prompt_id", "") for job in queue.jobs.values()
                                      if job["umo"] == event.unified_msg_origin] if queue is not None else [])
    identifiers.append((shared.get("last_prompt_ids") or {}).get(event.unified_msg_origin, ""))
    event.set_extra("comfyui_backend_ids", identifiers)


def clean_task_result(event) -> None:
    """在进入其他发送前插件之前，清除任务回复中无须播报的编号。"""
    if not event.get_extra("comfyui_task_reply"):
        return
    from astrbot.api.message_components import Plain

    result = event.get_result()
    if result is None:
        return
    identifiers = event.get_extra("comfyui_backend_ids") or []
    reasoning = event.get_extra("_llm_reasoning_content")
    if reasoning:
        event.set_extra("_llm_reasoning_content", hide_task_ids(reasoning, identifiers))
    for component in result.chain:
        if isinstance(component, Plain):
            component.text = hide_task_ids(component.text, identifiers)
    result.chain[:] = [component for component in result.chain
                       if not isinstance(component, Plain) or component.text.strip()]
