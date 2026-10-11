"""Route background completion replies through AstrBot's reply stages."""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING

from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import Plain, Record
from astrbot.core.message.message_event_result import ResultContentType
from astrbot.core.pipeline.context import PipelineContext, call_event_hook
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.star.star_handler import EventType

if TYPE_CHECKING:
    from astrbot.api.provider import LLMResponse
    from astrbot.api.star import Context


class CompletionEvent(AstrMessageEvent):
    """A fresh event that delivers to the job's original session."""

    def __init__(
        self,
        job: dict,
        summary: str,
        config: dict,
        send: Callable[[MessageChain], Awaitable[None]],
    ) -> None:
        session = MessageSession.from_str(job["umo"])
        message = AstrBotMessage()
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
        message.raw_message = None
        super().__init__(
            summary,
            message,
            PlatformMetadata(
                job.get("platform_name", session.platform_name),
                "",
                session.platform_name,
            ),
            session.session_id,
        )
        self.unified_msg_origin = job["umo"]
        self.role = job.get("sender_role", "member")
        self.is_wake = True
        enabled = config.get("plugin_set", ["*"])
        self.plugins_name = None if enabled == ["*"] else enabled
        self.set_extra("comfyui_completion_task_id", job["id"])
        self.config = config
        self._send_reply = send
        self.send_error: Exception | None = None
        self.sent_texts: list[str] = []

    async def send(self, message: MessageChain) -> None:
        if self.send_error:
            raise self.send_error
        try:
            await self._send_reply(message)
            self._has_send_oper = True
            text = message.get_plain_text().strip()
            if not text:
                # TTS-only delivery still carries its spoken text in Record.text.
                text = "\n".join(
                    component.text
                    for component in message.chain
                    if isinstance(component, Record) and component.text
                ).strip()
            if text:
                self.sent_texts.append(text)
        except Exception as error:
            # RespondStage logs transport failures; retain them for durable job state.
            self.send_error = error
            raise


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
            if from_llm:
                if response.reasoning_content:
                    event.set_extra(
                        "_llm_reasoning_content", response.reasoning_content
                    )
                await call_event_hook(event, EventType.OnLLMResponseEvent, response)
            if not event.is_stopped():
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
