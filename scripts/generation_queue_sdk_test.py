"""Run with the real AstrBot SDK: lifecycle, tool schemas and shared session lock."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

PLUGIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN))


async def completion_pipeline(root, cases):
    from astrbot.api.event import MessageChain
    from astrbot.api.provider import LLMResponse
    from astrbot.core.agent.message import TextPart
    from astrbot.core.star.star import StarMetadata, star_map
    from astrbot.core.star.star_handler import (
        EventType,
        StarHandlerMetadata,
        star_handlers_registry,
    )

    calls = []
    events = []
    module_path = "queue_completion_test"

    class Context(cases.Context):
        def __init__(self):
            super().__init__()
            self.reply_metadata = []

        def get_config(self, **kwargs):
            config = super().get_config(**kwargs)
            config["plugin_set"] = ["completion_formatter"]
            settings = config["platform_settings"]["segmented_reply"]
            settings.update(
                enable=True,
                only_llm_result=True,
                split_mode="regex",
                regex=r"[^\n]+",
                content_cleanup_rule=r"\[cleanup\]",
                interval="0,0",
                interval_method="random",
            )
            return config

        async def llm_generate(self, **kwargs):
            await super().llm_generate(**kwargs)
            self.response = LLMResponse(
                role="assistant",
                result_chain=MessageChain().message(
                    "**第一段**\\n\n**第二段**[cleanup]"
                ),
            )
            self.response.result_chain.use_markdown_ = True
            self.response.result_chain.use_t2i_ = False
            self.response.result_chain.type = "completion-test"
            return self.response

        async def send_message(self, umo, chain):
            if hasattr(chain, "use_markdown_"):
                self.reply_metadata.append((chain.use_markdown_, chain.type))
            return await super().send_message(umo, chain)

    async def request_hook(event, request):
        calls.append("request")
        assert request.session_id == event.unified_msg_origin
        assert request.conversation.cid == "cid-A"
        temporary.write_text("formatter temporary file", encoding="utf-8")
        event.track_temporary_local_file(str(temporary))
        if scenario == "request-stop":
            event.stop_event()
            return
        request.system_prompt += "\nformatter request hook"
        request.extra_user_content_parts.append(
            TextPart(text="formatter extra content")
        )
        request.model = "formatter-model"
        request.audio_urls = ["formatter-audio.wav"]
        event.set_extra("formatter_initialized", True)

    async def response_hook(event, response):
        calls.append("response")
        assert event.get_extra("formatter_initialized")
        if scenario == "stop":
            event.stop_event()
        elif scenario == "handled-response":
            await event.send(MessageChain().message("插件处理后的回复"))
            event.stop_event()
        response.completion_text = response.completion_text.replace("\\n", "\n")

    async def decorate_hook(event):
        calls.append("decorate")
        events.append(event)
        assert event.get_result().is_llm_result()
        assert event.get_sender_id() == "user-A"
        assert event.get_sender_name() == "Alice"
        assert event.get_group_id() == "A"
        assert event.get_self_id() == "bot"
        assert event.get_extra("comfyui_completion_task_id") == "C000001"
        for component in event.get_result().chain:
            component.text = component.text.replace("**", "")
        if scenario == "clear":
            event.clear_result()
        elif scenario == "decorate-stop":
            event.stop_event()
        elif scenario == "handled-decorate":
            await event.send(MessageChain().message("插件处理后的回复"))
            event.clear_result()

    async def after_hook(event):
        calls.append("after")

    async def excluded_hook(*args):
        raise AssertionError("a disabled plugin must not receive completion hooks")

    handlers = []
    for kind, handler in (
        (EventType.OnLLMRequestEvent, request_hook),
        (EventType.OnLLMResponseEvent, response_hook),
        (EventType.OnDecoratingResultEvent, decorate_hook),
        (EventType.OnAfterMessageSentEvent, after_hook),
        (EventType.OnDecoratingResultEvent, excluded_hook),
    ):
        path = "queue_disabled_test" if handler is excluded_hook else module_path
        handlers.append(
            StarHandlerMetadata(
                event_type=kind,
                handler_full_name=handler.__name__,
                handler_name=handler.__name__,
                handler_module_path=path,
                handler=handler,
                event_filters=[],
            )
        )
    registry_map = copy.copy(star_handlers_registry.star_handlers_map)
    with (
        patch.object(star_handlers_registry, "_handlers", []),
        patch.object(star_handlers_registry, "star_handlers_map", registry_map),
        patch.dict(
            star_map,
            {
                module_path: StarMetadata(name="completion_formatter"),
                "queue_disabled_test": StarMetadata(name="disabled_formatter"),
            },
        ),
    ):
        for handler in handlers:
            star_handlers_registry.append(handler)
        for scenario in (
            "normal",
            "recovered",
            "fallback",
            "disabled",
            "transport",
            "stop",
            "request-stop",
            "clear",
            "decorate-stop",
            "handled-response",
            "handled-decorate",
        ):
            calls.clear()
            events.clear()
            context = Context()
            origin = cases.event()
            origin.get_sender_name = lambda: "Alice"
            origin.get_group_id = lambda: "A"
            origin.get_self_id = lambda: "bot"
            origin._extras = {"_streaming_finished": True}
            origin._force_stopped = True
            origin._result = "original result"
            queue, _, client, _ = cases.make_queue(root / scenario, context=context)
            await queue.enqueue({}, origin)
            temporary = root / scenario / "format.tmp"
            text_attempts = []
            if scenario == "recovered":
                await cases.until(lambda: queue.jobs["C000001"]["state"] == "running")
                await queue.close()
                queue, _, _, _ = cases.make_queue(
                    root / scenario, context=context, client=client
                )
                await queue.start()
            elif scenario == "fallback":
                context.llm_error = True
            elif scenario == "disabled":
                queue.notify_llm = False
            elif scenario == "transport":
                # Fail only on reply text to verify RespondStage's swallowed error
                # is still recorded in the durable job's delivery state.
                send = context.send_message

                async def fail_text(umo, chain):
                    if not hasattr(chain, "items"):
                        text_attempts.append(chain)
                        raise RuntimeError("text transport offline")
                    return await send(umo, chain)

                context.send_message = fail_text
            client.finished.set()
            await cases.until(
                lambda: queue.jobs["C000001"]["delivery_state"] in {"sent", "failed"}
            )
            texts = [
                text
                for _, items in context.sent
                for kind, text in items
                if kind == "text"
            ]
            if scenario in {"normal", "recovered"}:
                await cases.until(lambda: len(context.conversation_manager.pairs) == 1)
                assert calls == ["request", "response", "decorate", "after"], calls
                assert texts == ["第一段", "第二段"], texts
                assert context.reply_metadata == [(True, "completion-test")] * 2
                assert "**" in context.response.completion_text, (
                    "decoration must not mutate the provider response chain"
                )
                assert "formatter request hook" in context.requests[0]["system_prompt"]
                assert context.requests[0]["model"] == "formatter-model"
                assert context.requests[0]["audio_urls"] == ["formatter-audio.wav"]
                assert (
                    context.requests[0]["extra_user_content_parts"][0].text
                    == "formatter extra content"
                )
                assert len(context.conversation_manager.pairs) == 1
                assert (
                    context.conversation_manager.pairs[0][2]["content"]
                    == "第一段\n第二段"
                )
            elif scenario in {"fallback", "disabled"}:
                assert calls == (["request"] if scenario == "fallback" else []) + [
                    "decorate",
                    "after",
                ]
                assert texts and "C000001" not in texts[0]
                assert all("本地路径" not in text for text in texts)
            elif scenario == "transport":
                assert queue.jobs["C000001"]["delivery_state"] == "failed"
                assert (
                    "text transport offline" in queue.jobs["C000001"]["delivery_error"]
                )
                assert not context.conversation_manager.pairs
                assert len(text_attempts) == 1, (
                    "transport failures must not trigger more platform sends"
                )
            elif scenario == "stop":
                assert calls == ["request", "response"] and not texts
                assert not context.conversation_manager.pairs
            elif scenario == "request-stop":
                assert calls == ["request"] and not context.requests and not texts
                assert not context.conversation_manager.pairs
            elif scenario in {"clear", "decorate-stop"}:
                assert calls == ["request", "response", "decorate"] and not texts
                assert not context.conversation_manager.pairs
            elif scenario in {"handled-response", "handled-decorate"}:
                await cases.until(lambda: len(context.conversation_manager.pairs) == 1)
                assert calls == ["request", "response"] + (
                    ["decorate"] if scenario == "handled-decorate" else []
                )
                assert texts == ["插件处理后的回复"]
                assert context.conversation_manager.pairs[0][2]["content"] == texts[0]
            assert not temporary.exists(), (
                "formatter temporary files must be cleaned up"
            )
            assert origin._extras == {"_streaming_finished": True}
            assert origin._force_stopped and origin._result == "original result"
            assert all(event is not origin for event in events)
            await queue.close()
    print(
        "real SDK completion hooks / plugin filtering / segmentation / cleanup / fresh events / recovery / fallback / transport / stop OK"
    )


async def run(root):
    import astrbot.api  # noqa: F401 - load the real SDK before the test fixtures
    import generation_queue_test as cases
    from astrbot.core.utils.session_lock import session_lock_manager

    # Run the full queue regressions against real SDK result processing as well.
    await cases.main()
    await completion_pipeline(root / "completion", cases)

    queue, context, client, _ = cases.make_queue(root / "locked")
    client.finished.set()
    try:
        async with session_lock_manager.acquire_lock(cases.event().unified_msg_origin):
            await queue.enqueue({}, cases.event())
            await cases.until(lambda: queue.jobs["C000001"]["state"] == "completed")
            await asyncio.sleep(0.04)
            assert not context.sent and not context.requests
        await cases.until(lambda: queue.jobs["C000001"]["delivery_state"] == "sent")
    finally:
        await queue.close()

    spec = importlib.util.spec_from_file_location(
        "queue_sdk_plugin", PLUGIN / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class PluginContext(cases.Context):
        def add_llm_tools(self, *tools):
            self.tools = tools

        def register_web_api(self, *args):
            pass

    context = PluginContext()
    with (
        patch.object(module.StarTools, "get_data_dir", return_value=root / "plugin"),
        patch.object(module.ComfyUIClient, "start_events"),
        patch.object(module.ComfyUIClient, "start_warmup"),
    ):
        plugin = module.ComfyUIDirectPlugin(context, {"image_cache_auto_clean": False})
        try:
            await plugin.initialize()
            assert (
                plugin._draw_tool.shared["generation_queue"] is plugin._generation_queue
            )
            job_tool = next(
                tool for tool in context.tools if tool.name == "comfyui_job"
            )
            assert job_tool.active and "task_id" in job_tool.parameters["properties"]
            from image_cache import save_image

            pending = save_image(
                plugin._output_dir, "pending.png", b"awaiting notification"
            )
            latest = save_image(
                plugin._output_dir, "latest.png", b"latest editing input"
            )
            stale = save_image(plugin._output_dir, "stale.png", b"unreferenced output")
            old = time.time() - 10 * 86400
            for path in (pending, latest, stale):
                os.utime(path, (old, old))
            plugin._generation_queue.jobs["pinned-test"] = {
                "delivery_state": "pending",
                "paths": [str(pending)],
            }
            plugin._generation_queue.shared["last_image_paths"] = {"test": str(latest)}
            cleaned = await plugin._manage_image_cache("expired")
            assert cleaned["removed"] == 1 and cleaned["protected"] == 2
            assert pending.is_file() and latest.is_file() and not stale.exists()
            assert (await plugin._manage_image_cache("status"))["protected"] == 2
            cleared = await plugin._manage_image_cache("clear")
            assert cleared["removed"] == 1 and pending.is_file() and not latest.exists()
            print(
                "real SDK automatic retention / explicit clear / pending-delivery cache protection OK"
            )
        finally:
            await plugin.terminate()
        assert plugin._generation_queue.closed and not plugin._generation_queue.tasks
    print(
        "real AstrBot SDK tool schemas / plugin initialization / session lock / termination OK"
    )


if __name__ == "__main__":
    previous_directory = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)  # Keep SDK initialization files out of the checkout.
            asyncio.run(run(Path(directory)))
        finally:
            os.chdir(previous_directory)
