"""Route background completion replies through AstrBot's reply stages."""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from time import time
from typing import TYPE_CHECKING

from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import Plain, Record
from astrbot.core.message.message_event_result import ResultContentType
from astrbot.core.pipeline.context import PipelineContext
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.platform_metadata import PlatformMetadata
from .hooks import finish_completion
from .transport import through_completion_event
from .receipts import clean_task_result, hide_task_ids

if TYPE_CHECKING:
    from astrbot.api.provider import LLMResponse
    from astrbot.api.star import Context


class CompletionEvent(AstrMessageEvent):
    """A fresh event that delivers to the job's original session."""

    _native_classes: dict[type, type] = {}

    def __new__(cls, job, summary, config, send, original=None):
        if isinstance(original, AstrMessageEvent):
            native = type(original)
            combined = native if issubclass(native, cls) else cls._native_classes.get(native)
            if combined is None:
                combined = type(f"ComfyCompletion{native.__name__}", (cls, native), {})
                cls._native_classes[native] = combined
            event = object.__new__(combined)
            event.__dict__.update(original.__dict__)
            return event
        return super().__new__(cls)

    def __init__(
        self,
        job: dict,
        summary: str,
        config: dict,
        send: Callable[[MessageChain], Awaitable[None]],
        original: AstrMessageEvent | None = None,
    ) -> None:
        session = MessageSession.from_str(job["umo"])
        original = original if isinstance(original, AstrMessageEvent) else None
        message = copy.copy(original.message_obj) if original is not None else AstrBotMessage()
        message.type = session.message_type
        message.session_id = session.session_id
        message.message_id = job.get("message_id", "")
        message.self_id = job.get("self_id", "")
        message.group_id = job.get("group_id", "")
        message.sender = MessageMember(
            job.get("sender_id", ""), job.get("sender_name", "")
        )
        message.message_str = summary
        message.message = [Plain(summary)]
        # 新完成事件从后台回复开始计时，不把 GPU 排队时间算作 LLM 超时。
        message.timestamp = time()
        raw_message = original.message_obj.raw_message if original is not None else None
        # SDK 消息携带 HTTP 客户端，浅拷贝保留客户端引用；普通载荷保持独立。
        message.raw_message = copy.deepcopy(raw_message) if isinstance(raw_message, (dict, list)) else copy.copy(raw_message)
        platform = copy.copy(original.platform_meta) if original is not None else PlatformMetadata(
            job.get("platform_name", session.platform_name), "", session.platform_name
        )
        AstrMessageEvent.__init__(
            self,
            summary,
            message,
            platform,
            session.session_id,
        )
        self.unified_msg_origin = job["umo"]
        self.role = job.get("sender_role", "member")
        self.is_wake = True
        self.is_at_or_wake_command = True
        if original is not None:
            self._extras.update({k: v for k, v in original._extras.items()
                                 if k not in {"provider_request", "llm_response", "_streaming_finished", "_llm_reasoning_content"}
                                 and not k.startswith("comfyui_")})
        enabled = config.get("plugin_set", ["*"])
        self.plugins_name = None if enabled == ["*"] else enabled
        self.set_extra("comfyui_completion_task_id", job["id"])
        self.set_extra("comfyui_source_message_timestamp", job.get("message_timestamp"))
        self.set_extra("comfyui_task_reply", True)
        self.set_extra("comfyui_backend_ids", [job.get("prompt_id", "")])
        self.config = config
        self._send_reply = send
        self.send_error: Exception | None = None
        self.sent_texts: list[str] = []
        self._recorded_chains: dict[int, int] = {}

    async def send(self, message: MessageChain) -> None:
        if self.send_error:
            raise self.send_error
        try:
            for component in message.chain:
                if isinstance(component, Plain):
                    component.text = hide_task_ids(component.text, self.get_extra("comfyui_backend_ids") or [])
            previous = self._recorded_chains.get(id(message), 0)
            with through_completion_event(self):
                await self._send_reply(message)
            if self._recorded_chains.get(id(message), 0) == previous:
                self.record_delivery(message)
        except Exception as error:
            # RespondStage logs transport failures; retain them for durable job state.
            self.send_error = error
            raise

    def record_delivery(self, message: MessageChain) -> None:
        """事件发送和直接 Context 发送统一记录，每次实际发送只记一次。"""
        self._has_send_oper = True
        self._recorded_chains[id(message)] = self._recorded_chains.get(id(message), 0) + 1
        text = message.get_plain_text().strip()
        if not text:
            text = "\n".join(component.text for component in message.chain
                             if isinstance(component, Record) and component.text).strip()
        if text:
            self.sent_texts.append(text)


async def send_completion_reply(
    context: Context,
    event: CompletionEvent,
    response: LLMResponse | None,
    *,
    from_llm: bool,
) -> str:
    """Apply reply processing and return the text actually delivered to the user."""
    try:
        if response is not None and not event.is_stopped():
            result = event.make_result()
            if response.result_chain:
                result.chain = copy.deepcopy(response.result_chain.chain)
                result.use_t2i_ = response.result_chain.use_t2i_
                result.use_markdown_ = response.result_chain.use_markdown_
                result.type = response.result_chain.type
            else:
                result.message(response.completion_text or "")
            # Fallback replies also need LLM-only cleanup and segmentation.
            event.set_result(
                result.set_result_content_type(ResultContentType.LLM_RESULT)
            )
            if from_llm:
                await finish_completion(event, response)
            if event.is_stopped():
                if event.send_error:
                    raise event.send_error
                return "\n".join(event.sent_texts)
            clean_task_result(event)
            config = event.config
            if not event.message_obj.message_id:
                # Restored jobs may predate persisted source-message metadata.
                config = {
                    **config,
                    "platform_settings": {
                        **config["platform_settings"],
                        "reply_with_quote": False,
                    },
                }
            pipeline = PipelineContext(
                astrbot_config=config,
                plugin_manager=SimpleNamespace(context=context),
                astrbot_config_id="",
            )
            decorate = ResultDecorateStage()
            respond = RespondStage()
            await decorate.initialize(pipeline)
            await respond.initialize(pipeline)
            async for _ in decorate.process(event):
                pass
            if not event.is_stopped():
                await respond.process(event)
        if event.send_error:
            raise event.send_error
        return "\n".join(event.sent_texts)
    finally:
        event.cleanup_temporary_local_files()
