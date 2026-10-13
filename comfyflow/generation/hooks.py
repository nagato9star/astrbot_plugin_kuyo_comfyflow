"""为单次后台回复补齐 AstrBot 的 Agent 生命周期。"""

import copy

from astrbot.core.pipeline.context import call_event_hook
from astrbot.core.star.star_handler import EventType


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(getattr(part, "text", "") for part in content or [])


async def begin_completion(context, event, request) -> bool:
    """请求 hook 已处理参数后，创建与正常 Agent 相同的运行上下文。"""
    if not hasattr(EventType, "OnAgentBeginEvent"):
        return False
    from astrbot.core.agent.message import Message
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.astr_agent_context import AstrAgentContext

    messages = [Message.model_validate(item) for item in request.contexts
                if item.get("role") != "_checkpoint"]
    had_system_prompt = bool(request.system_prompt)
    if had_system_prompt:
        messages.insert(0, Message(role="system", content=request.system_prompt))
    messages.append(Message(role="user", content=request.prompt))
    run_context = ContextWrapper(
        context=AstrAgentContext(context=context, event=event), messages=messages
    )
    event.set_extra("comfyui_agent_context", run_context)
    original_messages = copy.deepcopy(messages)
    if await call_event_hook(event, EventType.OnAgentBeginEvent, run_context):
        return True
    if run_context.messages == original_messages:
        return False
    # 开始 hook 可能修改消息上下文，下一次模型调用必须使用修改后的内容。
    messages = list(run_context.messages)
    if not messages:
        event.stop_event()
        return True
    systems = []
    while messages and messages[0].role == "system":
        systems.append(_text(messages.pop(0).content))
    request.system_prompt = "\n\n".join(systems)
    if messages and messages[-1].role == "user":
        request.prompt = _text(messages.pop().content)
    original_history = original_messages[1 if had_system_prompt else 0:-1]
    if messages != original_history:
        # 只修改提示词的 hook 不应重新序列化历史；历史改动也保留未变的前缀。
        original_contexts = [item for item in request.contexts if item.get("role") != "_checkpoint"]
        request.contexts = [original_contexts[index] if index < len(original_history)
                            and message == original_history[index] else message.model_dump()
                            for index, message in enumerate(messages)]
    return False


async def finish_completion(event, response) -> None:
    """按正常 Agent 顺序触发响应与结束 hook，并保留事件结果的直接修改。"""
    result = event.get_result()
    previous = copy.deepcopy(result.chain) if result is not None else None
    previous_format = {field: getattr(result, field) for field in ("use_t2i_", "use_markdown_", "type")} if result is not None else {}
    if response.reasoning_content:
        event.set_extra("_llm_reasoning_content", response.reasoning_content)
    event.set_extra("llm_response", response)
    await call_event_hook(event, EventType.OnLLMResponseEvent, response)
    run_context = event.get_extra("comfyui_agent_context")
    if run_context is not None:
        from astrbot.core.agent.message import Message

        original_text = response.completion_text or ""
        message = Message(role="assistant", content=original_text)
        run_context.messages.append(message)
        await call_event_hook(event, EventType.OnAgentDoneEvent, run_context, response)
        if response.completion_text == original_text and _text(message.content) != original_text:
            response.completion_text = _text(message.content)
    # 只在插件没有直接替换、清空或修改结果时，同步响应对象的变化。
    if result is not None and event.get_result() is result:
        if result.chain == previous:
            if response.result_chain is not None:
                result.chain = copy.deepcopy(response.result_chain.chain)
            else:
                from astrbot.api.message_components import Plain

                result.chain = [Plain(response.completion_text or "")]
        if response.result_chain is not None:
            # 格式标记随响应同步，但保留插件直接修改的事件结果字段。
            for field, value in previous_format.items():
                if getattr(result, field) == value:
                    setattr(result, field, getattr(response.result_chain, field))
    if event.get_result() is not None:
        from astrbot.core.message.message_event_result import ResultContentType

        event.get_result().set_result_content_type(ResultContentType.LLM_RESULT)
