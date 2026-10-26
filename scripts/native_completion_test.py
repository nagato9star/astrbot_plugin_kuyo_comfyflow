"""真实 SDK 主 Agent 唤醒、通知交付与缓存保持回归，不访问网络。"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PLUGIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN))


async def run(root):
    import astrbot.api  # noqa: F401
    from astrbot.api.event import AstrMessageEvent, MessageChain
    from astrbot.api.message_components import Image, Plain
    from astrbot.api.provider import LLMResponse
    from astrbot.core.agent.tool import FunctionTool, ToolSet
    from astrbot.core.db.po import Conversation
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
    from astrbot.core.platform.platform_metadata import PlatformMetadata
    from astrbot.core.provider.provider import Provider
    from astrbot.core.provider.entities import ProviderMeta
    from astrbot.core.star.star import StarMetadata, star_map
    from astrbot.core.star.star_handler import EventType, StarHandlerMetadata, star_handlers_registry

    import generation_queue_test as cases
    from comfyflow.generation.notices import SUCCESS_NOTICES
    from comfyflow.generation.recovery import reuse_images

    class NativeProvider(Provider):
        def __init__(self, context):
            super().__init__({"id": "original-provider", "type": "native-test", "max_context_tokens": 128000,
                              "modalities": ["text", "tool_use"], "model": "native-test"}, {})
            self.context = context
            self.set_model("native-test")

        def meta(self):
            return ProviderMeta(id="original-provider", model="native-test", type="native-test")

        def get_current_key(self):
            return ""

        def set_key(self, key):
            pass

        async def get_models(self):
            return ["native-test"]

        async def text_chat(self, **kwargs):
            context = self.context
            context.requests.append({key: copy.deepcopy(value) for key, value in kwargs.items()
                                     if key not in {"func_tool", "tools", "abort_signal"}})
            tools = kwargs.get("func_tool") or kwargs.get("tools")
            if scenario == "text-only-model":
                assert tools is None
            else:
                assert tools.names() == ["send_message_to_user"], tools.names()
            if scenario == "timeout" or (scenario == "tool-timeout" and len(context.requests) > 1):
                await asyncio.Event().wait()
            if scenario == "blank":
                return LLMResponse(role="assistant", completion_text="")
            if scenario == "provider-error":
                return LLMResponse(role="err", completion_text="provider offline")
            if scenario in {"tool", "tool-clear", "tool-transport", "tool-timeout", "tool-repeat"} and len(context.requests) == 1:
                repeats = 2 if scenario == "tool-repeat" else 1
                return LLMResponse(role="assistant", tools_call_args=[{
                                       "messages": [{"type": "plain", "text": "工具通知[cleanup]。"}],
                                   }] * repeats, tools_call_ids=[f"notice-call-{index}" for index in range(repeats)],
                                   tools_call_name=["send_message_to_user"] * repeats)
            return LLMResponse(role="assistant", completion_text="第一段[cleanup]。第二段。")

    class Manager:
        def get_full_tool_set(self):
            return ToolSet([FunctionTool(name="comfyui_draw", description="trap", parameters={}),
                            FunctionTool(name="comfyui_free_memory", description="trap", parameters={})])

        def get_builtin_tool(self, tool):
            return tool()

    class Context(cases.Context):
        def __init__(self):
            super().__init__()
            self.provider = NativeProvider(self)
            self.subagent_orchestrator = None
            self.kb_manager = object()
            self.saved_history = []
            original_get = self.conversation_manager.get_conversation

            async def get_conversation(umo, cid):
                item = await original_get(umo, cid)
                return Conversation(platform_id="test", user_id=umo, cid=cid, history=item.history,
                                    persona_id=item.persona_id) if item else None

            self.conversation_manager.get_conversation = get_conversation

        def get_config(self, **kwargs):
            config = super().get_config(**kwargs)
            config["plugin_set"] = ["native_formatter"]
            config["provider_settings"]["datetime_system_prompt"] = False
            config["provider_settings"]["request_max_retries"] = 1
            config["agent_runner"] = {"config": {"persona": {"safety_mode": False},
                                                   "model": {"request_max_retries": 1}}}
            return config

        async def resolve_selected_persona(self, **kwargs):
            return "friendly", {"name": "friendly", "prompt": "friendly persona", "tools": None,
                                "skills": [], "_begin_dialogs_processed": []}, None, False

        def get_llm_tool_manager(self):
            return Manager()

        def get_provider_by_id(self, provider_id):
            assert provider_id in {"original-provider", "hook-provider"}
            return self.provider

        async def llm_generate(self, **kwargs):
            raise AssertionError("原生唤醒不能调用旧的 llm_generate 路径")

        async def send_message(self, umo, chain):
            if scenario in {"transport", "tool-transport"} and any(isinstance(c, Plain) for c in chain.chain):
                raise RuntimeError("text transport offline")
            items = [("image", await c.convert_to_file_path()) if isinstance(c, Image)
                     else ("text", c.text) for c in chain.chain]
            self.sent.append((umo, items))
            return True

    seen = []
    systems = []
    before_notify = {}

    async def request(event, req):
        seen.append("request")
        before_notify.update({path: (hashlib.sha256(Path(path).read_bytes()).hexdigest(), Path(path).stat().st_mtime_ns)
                              for path in queue.jobs["C000001"]["paths"]})
        assert "friendly persona" in req.system_prompt
        assert "task_id=C000001" not in req.system_prompt and "pid-1" not in req.system_prompt
        systems.append(req.system_prompt)
        if scenario == "request-stop":
            event.stop_event()
        if scenario == "hook-provider":
            event.set_extra("selected_provider", "hook-provider")

    async def begin(event, run_context):
        seen.append("begin")
        assert event.get_result() is None
        if scenario == "begin-stop":
            event.stop_event()

    async def response(event, reply):
        seen.append("response")
        assert event.get_result().is_llm_result()
        if scenario == "clear":
            event.clear_result()
        elif scenario == "replace":
            event.set_result(event.plain_result("替换后的通知"))

    async def done(event, run_context, reply):
        seen.append("done")
        if scenario == "context-send":
            await context.send_message(event.unified_msg_origin, MessageChain().message("插件自行发送"))
            event.stop_event()

    async def decorate(event):
        seen.append("decorate")
        if scenario == "tool-clear":
            event.clear_result()
            return
        for component in event.get_result().chain:
            if isinstance(component, Plain):
                component.text = component.text.replace("[cleanup]", "")

    handlers = [(EventType.OnLLMRequestEvent, request), (EventType.OnAgentBeginEvent, begin),
                (EventType.OnLLMResponseEvent, response), (EventType.OnAgentDoneEvent, done),
                (EventType.OnDecoratingResultEvent, decorate)]
    with (patch.object(cases.generation_queue, "MessageChain", MessageChain),
          patch.object(star_handlers_registry, "_handlers", []),
          patch.object(star_handlers_registry, "star_handlers_map", copy.copy(star_handlers_registry.star_handlers_map)),
          patch.dict(star_map, {"native_formatter": StarMetadata(name="native_formatter")})):
        for kind, handler in handlers:
            star_handlers_registry.append(StarHandlerMetadata(
                event_type=kind, handler_full_name=handler.__name__, handler_name=handler.__name__,
                handler_module_path="native_formatter", handler=handler, event_filters=[],
            ))
        for scenario in ("normal", "tool", "tool-clear", "clear", "replace", "context-send", "no-conversation",
                         "restored", "branch", "request-stop", "begin-stop", "provider-error", "timeout",
                         "transport", "tool-transport", "hook-provider", "prefix-cache", "blank",
                         "tool-timeout", "tool-repeat", "text-only-model"):
            seen.clear()
            before_notify.clear()
            context = Context()
            if scenario == "text-only-model":
                context.provider.provider_config["modalities"] = ["text"]
            message = AstrBotMessage()
            message.type = "GroupMessage"
            message.session_id = message.group_id = "A"
            message.self_id = "bot"
            message.message_id = "source"
            message.sender = MessageMember("user-A", "Alice")
            message.message = [Plain("draw a cat")]
            message.raw_message = {}
            origin = AstrMessageEvent("draw a cat", message, PlatformMetadata("test", "test", "test"), "A")
            if scenario == "no-conversation":
                context.conversation_manager.current.pop(origin.unified_msg_origin)
            queue, _, client, _ = cases.make_queue(root / scenario, context=context)
            try:
                if scenario in {"timeout", "tool-timeout"}:
                    queue.llm_timeout = 0.8
                await queue.enqueue({"node": {"class_type": "Demo", "inputs": {"seed": 42}}}, origin,
                                    history={"prompt": "cat"})
                await cases.until(lambda: bool(client.submissions))
                submitted = copy.deepcopy(client.submissions)
                if scenario == "restored":
                    queue.events.clear()
                if scenario == "branch":
                    context.conversation_manager.current[origin.unified_msg_origin] = "new-cid"
                client.finished.set()
                await cases.until(lambda: queue.jobs["C000001"]["delivery_state"] in {"sent", "failed"}
                                  and not queue.delivering, timeout=12)
                job = queue.jobs["C000001"]
                texts = [text for _, items in context.sent for kind, text in items if kind == "text"]
                assert len(client.submissions) == 1 and client.submissions == submitted
                assert sum(kind == "image" for _, items in context.sent for kind, _ in items) == 2
                before = {path: (hashlib.sha256(Path(path).read_bytes()).hexdigest(), Path(path).stat().st_mtime_ns)
                          for path in job["paths"]}
                if before_notify:
                    assert before_notify == before
                assert reuse_images(job["paths"])
                assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == state[0]
                           and Path(path).stat().st_mtime_ns >= state[1] for path, state in before.items())
                if scenario in {"transport", "tool-transport"}:
                    assert job["delivery_state"] == "failed" and "text transport offline" in job["delivery_error"], job
                else:
                    assert job["delivery_state"] == "sent", job
                if scenario in {"provider-error", "timeout", "branch", "blank"}:
                    assert len(texts) == 1 and texts[0] in SUCCESS_NOTICES, (scenario, texts, job)
                elif scenario in {"clear", "tool-clear", "request-stop", "begin-stop", "transport", "tool-transport"}:
                    assert not texts, (scenario, texts)
                elif scenario == "replace":
                    assert texts == ["替换后的通知"]
                elif scenario == "context-send":
                    assert texts == ["插件自行发送"]
                elif scenario in {"tool", "tool-timeout", "tool-repeat"}:
                    assert texts == ["工具通知。"], texts
                else:
                    assert texts == ["第一段。第二段。"], (scenario, texts, job)
                if scenario in {"no-conversation", "branch"}:
                    assert not context.conversation_manager.pairs
                if texts and context.conversation_manager.pairs:
                    assert context.conversation_manager.pairs[-1][2]["content"] == "\n".join(texts)
                if scenario in {"request-stop", "begin-stop", "branch"}:
                    assert not context.requests
                if scenario == "prefix-cache":
                    first_system = systems[-1]
                    await queue.enqueue(submitted[0], origin, history={"prompt": "different request"})
                    await cases.until(lambda: queue.jobs["C000002"]["delivery_state"] == "sent" and not queue.delivering)
                    assert systems[-1] == first_system
                    assert client.submissions[0] == client.submissions[1]
                print(scenario, "OK")
            finally:
                await queue.close()
    print("real native main agent / single delivery / cache bytes and timestamps / stable system prefix OK")


async def main(root):
    try:
        await run(root)
    finally:
        from astrbot.core import db_helper
        await db_helper.engine.dispose()


if __name__ == "__main__":
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            asyncio.run(main(Path(directory)))
        finally:
            os.chdir(previous)
