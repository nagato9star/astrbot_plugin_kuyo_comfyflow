"""使用 AstrBot 主 Agent 处理已交付图片的后台完成事件。"""

from dataclasses import fields


def native_api():
    """旧版 SDK 缺少主 Agent 构建接口时保留单次通知路径。"""
    try:
        from astrbot.core.astr_main_agent import MainAgentBuildConfig, build_main_agent
    except ImportError:
        return None
    return MainAgentBuildConfig, build_main_agent


async def wake_completion(context, event, request, provider_id: str):
    """返回是否支持原生唤醒及其响应，执行失败交由队列统一兜底。"""
    api = native_api()
    if api is None:
        return False, None

    from astrbot.api.provider import LLMResponse
    from astrbot.core.agent.tool import FunctionTool, ToolSet
    from astrbot.core.astr_agent_hooks import MainAgentHooks
    from astrbot.core.db.po import Conversation
    from astrbot.core.pipeline.context import call_event_hook
    from astrbot.core.star.star_handler import EventType

    from .hooks import finish_completion
    from .reply import prepare_completion_result, send_completion_reply

    class SendCompletionText(FunctionTool):
        async def call(self, run_context, messages: list):
            if event.get_extra("comfyui_completion_text_handled"):
                return "本次完成通知已经处理，无需再次发送。"
            if any(item.get("type") != "plain" for item in messages):
                return "图片已经由后台交付，本次通知仅支持文本。"
            text = "\n".join(item["text"] for item in messages).strip()
            if not text:
                return "请提供要发送的通知文本。"
            event.set_extra("comfyui_completion_text_handled", True)
            await send_completion_reply(
                context, event, LLMResponse(role="assistant", completion_text=text), from_llm=False
            )
            return "完成通知已处理，不要重复发送或再次生成图片。"

    began = False

    class CompletionHooks(MainAgentHooks):
        async def on_agent_begin(self, run_context):
            nonlocal began
            if began:
                return
            began = True
            await super().on_agent_begin(run_context)
            if event.is_stopped():
                runner.request_stop()

        async def on_agent_done(self, run_context, response):
            prepare_completion_result(event, response)
            await finish_completion(event, response, run_context=run_context)
            event.set_extra("comfyui_native_completion", True)

    config_type, build = api
    config = event.config
    settings = config.get("provider_settings") or {}
    runner_config = (config.get("agent_runner") or {}).get("config") or {}
    misc = runner_config.get("misc") or {}
    persona = runner_config.get("persona") or {}
    model = runner_config.get("model") or {}
    options = {
        "tool_call_timeout": misc.get("tool_call_timeout", 120),
        "streaming_response": False,
        "computer_use_runtime": "none",
        "add_cron_tools": False,
        "provider_settings": settings,
        "llm_safety_mode": persona.get("safety_mode", True),
        "safety_mode_strategy": persona.get("safety_mode_strategy", "system_prompt"),
        "timezone": config.get("timezone"),
        "fallback_provider_ids": model.get("fallback_provider_ids", []),
        "request_max_retries": model.get("request_max_retries", 5),
    }
    # 不同 SDK 版本的构建配置字段不同，只传当前版本支持的字段。
    supported = {field.name for field in fields(config_type)}
    build_config = config_type(**{key: value for key, value in options.items() if key in supported})
    if request.conversation is None:
        # 临时对话只供核心解析本会话人格，不创建或切换持久化对话。
        request.conversation = Conversation(
            platform_id=event.get_platform_id(), user_id=event.unified_msg_origin,
            cid="", history="[]", persona_id=None,
        )
    request.system_prompt = ""
    provider = context.get_provider_by_id(provider_id)
    if provider is None:
        raise RuntimeError(f"完成通知模型不可用：{provider_id}")
    result = await build(
        event=event, plugin_context=context, config=build_config,
        provider=provider, req=request, apply_reset=False,
    )
    if result is None:
        raise RuntimeError("无法构建完成通知主 Agent。")
    reset = result.reset_coro
    try:
        event.set_extra("provider_request", result.provider_request)
        if await call_event_hook(event, EventType.OnLLMRequestEvent, result.provider_request):
            return True, None
        # 仅允许向原会话发送一次文本，完成回调不开放生图、发图或缓存管理工具。
        result.provider_request.func_tool = ToolSet([SendCompletionText(
            name="send_message_to_user",
            description="向当前会话发送本次图片任务的完成通知文本。图片已交付，无需重复发送。",
            parameters={"type": "object", "properties": {"messages": {
                "type": "array", "items": {"type": "object", "properties": {
                    "type": {"type": "string", "enum": ["plain"]},
                    "text": {"type": "string"},
                }, "required": ["type", "text"]},
            }}, "required": ["messages"]},
        )])
        await reset
        reset = None
        runner = result.agent_runner
        selected = event.get_extra("selected_provider")
        if isinstance(selected, str) and selected and selected != provider_id:
            selected_provider = context.get_provider_by_id(selected)
            if selected_provider is None:
                raise RuntimeError(f"完成通知模型不可用：{selected}")
            runner.provider = selected_provider
        runner.agent_hooks = CompletionHooks()
        # 部分 SDK 直到 Provider 返回才检查停止信号，先让开始 hook 完成拦截。
        await runner.agent_hooks.on_agent_begin(runner.run_context)
        if event.is_stopped():
            return True, None
        async for _ in runner.step_until_done(3):
            if event.is_stopped():
                runner.request_stop()
        response = runner.get_final_llm_resp()
        if event.send_error:
            raise event.send_error
        state = getattr(runner, "state", getattr(runner, "_state", None))
        if getattr(state, "name", "") == "ERROR" or response is None or response.role != "assistant":
            raise RuntimeError("完成通知主 Agent 执行失败。")
        result_chain = event.get_result()
        if (not event.get_extra("comfyui_completion_text_handled") and not event.is_stopped()
                and result_chain is not None and not result_chain.get_plain_text().strip()):
            raise RuntimeError("完成通知主 Agent 未返回通知文本。")
        return True, response
    finally:
        if reset is not None:
            reset.close()
