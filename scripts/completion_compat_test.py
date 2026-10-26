"""使用真实 SDK 和可选的 OutputPro 验证后台回复及编号策略。"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

PLUGIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN))


async def run(root, outputpro):
    import astrbot.api  # noqa: F401
    from astrbot.api.event import AstrMessageEvent, MessageChain
    from astrbot.api.message_components import Image, Plain
    from astrbot.api.provider import LLMResponse
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
    from astrbot.core.platform.platform_metadata import PlatformMetadata
    from astrbot.core.star.star import StarMetadata, star_map
    from astrbot.core.star.star_handler import EventType, StarHandlerMetadata, star_handlers_registry

    import generation_queue_test as cases
    from comfyflow.generation.receipts import INTERNAL_ID_NOTICE, clean_task_result, hide_task_ids, task_reference
    from comfyflow.tools import ComfyuiJobTool
    from comfyflow.generation.reply import CompletionEvent
    from comfyflow.generation.transport import capture_completion_sends

    class NativeEvent(AstrMessageEvent):
        def __init__(self):
            message = AstrBotMessage()
            message.type = "GroupMessage"
            message.session_id = message.group_id = "A"
            message.message_id = "source-message"
            message.self_id = "bot"
            message.sender = MessageMember("user-A", "Alice")
            message.message = [Plain("draw a cat")]
            message.raw_message = {"native": True}
            super().__init__("draw a cat", message, PlatformMetadata("aiocqhttp", "test", "test"), "A")
            self.bot = object()
            self.set_extra("cleaner_state", "prepared")
            self.set_extra("_streaming_finished", True)
            self.stop_event()
            self.set_result(self.plain_result("original result"))

    class Context(cases.Context):
        def get_config(self, **kwargs):
            config = super().get_config(**kwargs)
            config["plugin_set"] = ["compat_formatter"]
            return config

        async def llm_generate(self, **kwargs):
            await super().llm_generate(**kwargs)
            if scenario == "outputpro":
                text = "[动作]第一段好了。第二段好了（任务编号：C000001）(旁白)&&happy&&"
            else:
                text = "[动作]图片好了（任务编号：C000001）(旁白)&&happy&&"
            text += INTERNAL_ID_NOTICE
            chain = MessageChain().message(text)
            if scenario in {"response_format", "agent_format", "event_chain_format", "event_format"}:
                chain.use_t2i_ = chain.use_markdown_ = scenario != "event_format"
                chain.type = "original-format"
            return LLMResponse(role="assistant", result_chain=chain)

        async def send_message(self, umo, chain):
            items = []
            for component in chain.chain:
                if isinstance(component, Image):
                    items.append(("image", await component.convert_to_file_path()))
                elif isinstance(component, Plain):
                    items.append(("text", component.text))
            self.sent.append((umo, items))
            return True

    real_cleaner = None
    if outputpro is not None:
        sys.path.insert(0, str(outputpro.parent))
        from astrbot_plugin_outputpro.main import OutputPlugin

        def defaults(schema):
            return {key: defaults(value["items"]) if value.get("type") == "object"
                    else value.get("default") for key, value in schema.items()}

        config = defaults(json.loads((outputpro / "_conf_schema.json").read_text(encoding="utf-8")))
        config["pipeline"]["steps"] = ["block(消息拦截)", "clean(文本清洗)", "replace(文本替换)", "split(分段回复)"]
        config["pipeline"]["llm_steps"] = ["block(消息拦截)", "clean(文本清洗)", "replace(文本替换)", "split(分段回复)"]
        config["block"].update(timeout=60, block_reread=False, block_words=[])
        config["clean"].update(text_threshold=1000, bracket=True, parenthesis=True, emotion_tag=True)
        config["clean"]["punctuation"] = ""
        config["replace"]["words"] = ["好了 完成"]
        config["split"].update(show_typing=False, per_char_delay=0, delay_scope_str="0~0",
                               max_length=1000, max_count=4, char_list=["。"], tail_punc=["。"])
        real_cleaner = OutputPlugin(Context(), config)

    module = "completion_compat_formatter"
    seen = []
    observed_formats = {}

    async def waiting(event):
        seen.append("waiting")
        queue.begin_turn(event)
        assert event.get_extra("cleaner_state") == "prepared" or scenario == "restored"
        if scenario == "waiting_stop":
            event.stop_event()

    async def request(event, req):
        seen.append("request")
        assert req is event.get_extra("provider_request")
        if scenario == "no_conversation":
            assert req.conversation is None and req.contexts == []
        else:
            assert req.contexts == json.loads(req.conversation.history)
        assert req.system_prompt == "friendly persona"
        if scenario == "selected_provider_hook":
            event.set_extra("selected_provider", "hook-provider")

    async def begin(event, run_context):
        seen.append("begin")
        assert run_context.context.event is event and run_context.messages
        assert event.get_result() is None
        if scenario == "agent_messages":
            run_context.messages[-1].content += "[begin修改]"
        elif scenario == "agent_history":
            run_context.messages[-2].content += "[历史修改]"
        elif scenario == "begin_stop":
            event.stop_event()

    async def response(event, reply):
        seen.append("response")
        assert event.get_result().is_llm_result()
        if scenario == "response_clear":
            event.clear_result()
        elif scenario == "response_replace":
            event.set_result(event.plain_result("直接替换后的回复"))
        elif scenario in {"response_format", "event_chain_format", "event_format"}:
            reply.result_chain.use_t2i_ = reply.result_chain.use_markdown_ = scenario == "event_format"
            reply.result_chain.type = "response-format"
            if scenario == "event_chain_format":
                event.get_result().chain = [Plain("直接修改组件后的回复")]
            elif scenario == "event_format":
                event.get_result().use_t2i_ = event.get_result().use_markdown_ = None
                event.get_result().type = "event-format"

    async def done(event, run_context, reply):
        seen.append("done")
        assert run_context.context.event is event
        assert run_context.messages[-1].role == "assistant"
        event.set_extra("comfyui_include_task_ids", True)
        if scenario in {"agent_clean", "agent_history", "native", "restored", "response_format",
                        "agent_format", "event_format", "event_chain_format",
                        "no_conversation", "selected_provider_hook"}:
            reply.completion_text = "Agent 结束阶段清理后的回复"
        elif scenario == "agent_messages":
            assert "[begin修改]" in context.requests[-1]["prompt"]
            run_context.messages[-1].content = "运行上下文中的回复已清理"
        elif scenario == "agent_clear":
            event.clear_result()
        elif scenario == "handled":
            await event.send(MessageChain().message("插件自行发送C000001完成回复（编号：pid-1）"))
            event.stop_event()
        elif scenario == "handled_context":
            await context.send_message(event.unified_msg_origin, MessageChain().message("插件直接Context发送C000001完成回复（编号：pid-1）"))
            event.stop_event()
        if scenario == "agent_format":
            reply.result_chain.use_t2i_ = reply.result_chain.use_markdown_ = False
            reply.result_chain.type = "agent-format"

    async def decorate(event):
        seen.append("decorate")
        if scenario in {"response_format", "agent_format", "event_chain_format", "event_format"}:
            result = event.get_result()
            observed_formats[scenario] = (result.use_t2i_, result.use_markdown_, result.type)
        if scenario == "native":
            assert isinstance(event, NativeEvent) and event.bot is origin.bot
            assert event.message_obj.raw_message == {"native": True}
            assert event.get_extra("_streaming_finished") is None
            assert event.get_extra("cleaner_state") == "prepared"
        if scenario.startswith("outputpro"):
            if scenario == "outputpro":
                assert event.get_extra("comfyui_source_message_timestamp") == origin.message_obj.timestamp
                assert time.time() - event.message_obj.timestamp < 60
            else:
                event.message_obj.timestamp = int(time.time()) - 120
            await real_cleaner.on_decorating_result(event)

    async def after(event):
        seen.append("after")

    handlers = [(EventType.OnWaitingLLMRequestEvent, waiting), (EventType.OnLLMRequestEvent, request),
                (EventType.OnAgentBeginEvent, begin), (EventType.OnLLMResponseEvent, response),
                (EventType.OnAgentDoneEvent, done), (EventType.OnDecoratingResultEvent, decorate),
                (EventType.OnAfterMessageSentEvent, after)]
    notification_logger = Mock()
    with (
        patch.object(cases.generation_queue, "logger", notification_logger),
        patch.object(star_handlers_registry, "_handlers", []),
        patch.object(star_handlers_registry, "star_handlers_map", copy.copy(star_handlers_registry.star_handlers_map)),
        patch.dict(star_map, {module: StarMetadata(name="compat_formatter")}),
        patch.object(cases.generation_queue, "MessageChain", MessageChain),
    ):
        for event_type, handler in handlers:
            star_handlers_registry.append(StarHandlerMetadata(
                event_type=event_type, handler_full_name=handler.__name__, handler_name=handler.__name__,
                handler_module_path=module, handler=handler, event_filters=[],
            ))
        scenarios = ["agent_clean", "agent_messages", "agent_history", "response_clear", "response_replace", "agent_clear",
                     "handled", "handled_context", "native", "restored", "waiting_stop", "begin_stop",
                     "response_format", "agent_format", "event_chain_format", "event_format",
                     "no_conversation", "selected_provider_hook"]
        if real_cleaner is not None:
            scenarios.extend(["outputpro", "outputpro_actual_timeout"])
        for scenario in scenarios:
            seen.clear()
            notification_logger.reset_mock()
            context, origin = Context(), NativeEvent()
            if scenario == "no_conversation":
                context.conversation_manager.current.pop(origin.unified_msg_origin)
            if scenario == "outputpro":
                origin.message_obj.timestamp = int(time.time()) - 600
            original_result = origin.get_result()
            queue, _, client, shared = cases.make_queue(root / scenario, context=context)
            old_messages = [{"role": "assistant", "content": None, "tool_calls": [
                {"id": "old-tool", "type": "function", "function": {"name": "comfyui_job", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "old-tool", "content": "任务C000099，pid-old"},
                {"role": "assistant", "content": "旧回复"}]
            context.conversation_manager.history["cid-A"].extend(old_messages)
            client.finished.set()
            if real_cleaner is not None:
                real_cleaner.cfg.context = context
            try:
                receipt = await queue.enqueue({}, origin, history={"prompt": "Manhattan Cafe at sunset", "recipe": "Anima"})
                assert "task_id=C000001" in receipt and INTERNAL_ID_NOTICE in receipt
                if scenario == "restored":
                    queue.events.clear()
                await cases.until(lambda: queue.jobs["C000001"]["delivery_state"] in {"sent", "failed"} and not queue.delivering, timeout=6)
                job = queue.jobs["C000001"]
                assert job["delivery_state"] == "sent", job.get("delivery_error")
                if scenario in {"response_format", "agent_format", "event_chain_format", "event_format"}:
                    expected = None if scenario == "event_format" else False
                    kind = {"agent_format": "agent-format", "event_format": "event-format"}.get(scenario, "response-format")
                    assert observed_formats[scenario] == (expected, expected, kind)
                texts = [text for _, items in context.sent for kind, text in items if kind == "text"]
                if scenario == "waiting_stop":
                    assert seen == ["waiting"] and not texts and not context.requests
                elif scenario == "begin_stop":
                    assert seen == ["waiting", "request", "begin"] and not texts and not context.requests
                else:
                    assert seen[:5] == ["waiting", "request", "begin", "response", "done"], seen
                assert not any("C000001" in text for text in texts)
                assert not any("pid-1" in text or INTERNAL_ID_NOTICE in text for text in texts)
                log_messages = [call.args[0] for call in notification_logger.info.call_args_list]
                assert any("C000001 图片发送结束" in item for item in log_messages)
                assert any("C000001 完成通知处理结束" in item for item in log_messages)
                if context.requests:
                    prompt = context.requests[-1]["prompt"]
                    assert "task_id=C000001" in prompt and "prompt_id=pid-1" in prompt
                    assert "生成指令：Manhattan Cafe at sunset" in prompt and "配方：Anima" in prompt
                    assert "原请求：draw a cat" in prompt
                    if scenario == "no_conversation":
                        assert context.requests[-1]["contexts"] == []
                    elif scenario == "agent_history":
                        assert context.requests[-1]["contexts"][-3:-1] == old_messages[:-1]
                        assert context.requests[-1]["contexts"][-1]["content"] == "旧回复[历史修改]"
                    else:
                        assert context.requests[-1]["contexts"][-3:] == old_messages
                    assert any("C000001 触发完成通知 LLM" in item for item in log_messages)
                    assert any("C000001 完成通知 LLM 返回" in item for item in log_messages)
                    if scenario == "selected_provider_hook":
                        assert context.requests[-1]["chat_provider_id"] == "hook-provider"
                else:
                    assert any("C000001 完成通知 LLM 跳过" in item for item in log_messages)
                    assert not any("触发完成通知 LLM" in item for item in log_messages)
                assert not queue.turns
                assert origin.is_stopped() and origin.get_result() is original_result
                assert origin.get_extra("_streaming_finished") is True
                if scenario in {"response_clear", "agent_clear", "waiting_stop", "begin_stop"}:
                    assert not texts and "decorate" not in seen and not context.conversation_manager.pairs
                elif scenario == "outputpro_actual_timeout":
                    assert not texts and not context.conversation_manager.pairs
                elif scenario == "response_replace":
                    assert texts == ["直接替换后的回复"]
                elif scenario == "event_chain_format":
                    assert texts == ["直接修改组件后的回复"]
                elif scenario == "agent_messages":
                    assert texts == ["运行上下文中的回复已清理"]
                elif scenario == "handled":
                    assert texts == ["插件自行发送完成回复"]
                elif scenario == "handled_context":
                    assert texts == ["插件直接Context发送完成回复"]
                elif scenario == "outputpro":
                    assert texts == ["第一段完成", "第二段完成"], texts
                    assert "after" in seen
                else:
                    assert texts == ["Agent 结束阶段清理后的回复"], texts
                if scenario == "no_conversation":
                    assert not context.conversation_manager.pairs
                elif texts:
                    assert context.conversation_manager.pairs[-1][2]["content"] == "\n".join(texts)
                job_tool = ComfyuiJobTool(client=client, shared=shared)
                wrapped = type("Wrapper", (), {"context": type("Inner", (), {"event": origin})()})()
                hidden = await job_tool.call(wrapped, action="status")
                shown = await job_tool.call(wrapped, action="status", include_ids=True)
                assert "task_id=C000001" in hidden and "prompt_id=pid-1" in hidden and INTERNAL_ID_NOTICE in hidden
                assert hidden == shown
                assert "include_ids" not in job_tool.parameters["properties"]
                assert "C000001" in queue.describe(job) and "pid-1" in queue.describe(job)
                origin.set_result(origin.plain_result("已加入队列C000001，种子2026101301，节点509\n" + INTERNAL_ID_NOTICE))
                origin.set_extra("comfyui_include_task_ids", True)
                clean_task_result(origin)
                assert origin.get_result().get_plain_text() == "已加入队列，种子2026101301，节点509"
                print(scenario, "OK")
            finally:
                await queue.close()
    assert hide_task_ids("任务C000001已完成", []) == "任务已完成"
    assert hide_task_ids("图片已完成（prompt_id=pid-1）", ["pid-1"]) == "图片已完成"
    long_job = {"id": "C000001", "source_request": "原请求" * 200,
                "history": {"prompt": "long prompt " * 500}}
    original = copy.deepcopy(long_job)
    summary = task_reference(long_job)
    assert len(summary) < 500 and "task_id=C000001" in summary
    assert "…" in summary and long_job == original
    duplicate = task_reference({"id": "C000002", "source_request": "cat", "history": {"prompt": "cat"}})
    assert duplicate.count("cat") == 1
    shared_context = Context()
    original_send = shared_context.send_message
    events = []
    for suffix in ("A", "B"):
        event = CompletionEvent(
            {"id": "C000001", "umo": f"test:GroupMessage:{suffix}"},
            "completed", shared_context.get_config(), lambda chain: None,
        )
        event.set_extra("comfyui_include_task_ids", True)
        events.append(event)
    ready = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()

    async def concurrent(index):
        event = events[index]
        with capture_completion_sends(shared_context, event):
            ready[index].set()
            await release.wait()
            await shared_context.send_message(event.unified_msg_origin, MessageChain().message(f"{index} C000001"))

    tasks = [asyncio.create_task(concurrent(index)) for index in range(2)]
    await asyncio.gather(*(gate.wait() for gate in ready))
    await shared_context.send_message("test:GroupMessage:A", MessageChain().message("普通会话C000999"))
    release.set()
    await asyncio.gather(*tasks)
    assert events[0].sent_texts == ["0"] and events[1].sent_texts == ["1"]
    assert shared_context.sent[0][1] == [("text", "普通会话C000999")]
    assert shared_context.send_message == original_send
    assert not hasattr(shared_context, "_comfyui_send_capture")
    print("parallel completion sends / unrelated normal sends / Context restoration OK")
    print("native events / full lifecycle / real cleanup / model task association / silent chat IDs / notification logs / history fidelity OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputpro", type=Path)
    args = parser.parse_args()
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            import astrbot.api  # noqa: F401
            # 旧 SDK 路径的完整 hook 兼容性单独保留。
            with patch("comfyflow.generation.agent.native_api", return_value=None):
                asyncio.run(run(Path(directory), args.outputpro))
        finally:
            os.chdir(cwd)
