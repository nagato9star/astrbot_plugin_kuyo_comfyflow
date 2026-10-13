"""按异步任务隔离后台回复期间的直接发送记录。"""

from contextlib import contextmanager
from contextvars import ContextVar

_current_event = ContextVar("comfyui_completion_delivery", default=None)
_through_event = ContextVar("comfyui_event_send", default=None)


@contextmanager
def through_completion_event(event):
    token = _through_event.set(event)
    try:
        yield
    finally:
        _through_event.reset(token)


@contextmanager
def capture_completion_sends(context, event):
    """兼容直接调用共享 Context 发送的分段插件，不拦截其他任务或会话。"""
    state = getattr(context, "_comfyui_send_capture", None)
    if state is None:
        original = context.send_message
        state = {"original": original, "users": 0}

        async def send_message(session, message_chain):
            current = _current_event.get()
            matches = current is not None and str(session) == current.unified_msg_origin
            if matches:
                from astrbot.api.message_components import Plain
                from .receipts import hide_task_ids

                for component in message_chain.chain:
                    if isinstance(component, Plain):
                        component.text = hide_task_ids(component.text, current.get_extra("comfyui_backend_ids") or [])
            try:
                found = await original(session, message_chain)
            except Exception as error:
                if matches:
                    current.send_error = error
                raise
            if matches:
                if found is False:
                    if _through_event.get() is not current:
                        # 部分平台只支持原事件发送；复用队列已有的原事件兜底。
                        await current.send(message_chain)
                        return True
                else:
                    current.record_delivery(message_chain)
            return found

        state["wrapper"] = send_message
        context._comfyui_send_capture = state
        context.send_message = send_message
    state["users"] += 1
    token = _current_event.set(event)
    try:
        yield
    finally:
        _current_event.reset(token)
        state["users"] -= 1
        if not state["users"]:
            if context.send_message is state["wrapper"]:
                context.send_message = state["original"]
            del context._comfyui_send_capture
