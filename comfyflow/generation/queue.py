"""Durable plugin jobs, independent ComfyUI execution and conversation delivery."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.event import MessageChain

from ..clients.comfyui import execution_error_message
from ..storage.images import save_image
from .receipts import INTERNAL_ID_NOTICE, hide_task_ids, task_reference

if TYPE_CHECKING:
    from astrbot.api.provider import LLMResponse
    from .reply import CompletionEvent

ACTIVE = {"queued", "submitting", "running", "downloading"}
TERMINAL = {"completed", "failed", "cancelled", "uncertain"}
LABELS = {
    "queued": "排队中",
    "submitting": "提交中",
    "running": "生成中",
    "downloading": "回收图片中",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
    "uncertain": "提交结果待确认",
}


class GenerationQueue:
    def __init__(
        self,
        context,
        client,
        store,
        output_dir: Path,
        data_dir: Path,
        shared: dict,
        *,
        max_pending: int = 32,
        concurrency: int = 1,
        notify_llm: bool = True,
        llm_timeout: float = 60,
    ) -> None:
        self.context, self.client, self.store = context, client, store
        self.output_dir = output_dir
        self.path = data_dir / "generation_jobs.json"
        self.shared = shared
        self.max_pending = max(1, max_pending)
        self.concurrency = max(1, min(4, concurrency))
        self.notify_llm, self.llm_timeout = notify_llm, max(1, llm_timeout)
        self.jobs: dict[str, dict] = {}
        self.sequence = 0
        self.pending: asyncio.Queue[str] = asyncio.Queue()
        self.mutation_lock = asyncio.Lock()
        self.delivery_slots = asyncio.Semaphore(2)
        self.tasks: set[asyncio.Task] = set()
        self.delivering: set[str] = set()
        self.events: dict[str, Any] = {}
        self.turns: dict[int, asyncio.Event] = {}
        self.session_locks: dict[str, asyncio.Lock] = {}
        self.started = False
        self.closed = False
        self.on_schema_change = None
        if self.path.exists():
            # Refuse to overwrite corrupt state: accepted tasks must remain recoverable.
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(
                payload.get("jobs"), dict
            ):
                raise ValueError(
                    "生图任务文件格式无效，请保留并检查 generation_jobs.json"
                )
            self.sequence = int(payload["sequence"])
            self.jobs = payload["jobs"]

    def _spawn(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)

        def finished(task):
            self.tasks.discard(task)
            if not task.cancelled() and task.exception():
                logger.error(f"[ComfyUIDirect] 队列后台异常: {task.exception()}")

        task.add_done_callback(finished)

    def begin_turn(self, event) -> None:
        key = id(event)
        if key not in self.turns:
            ready = self.turns[key] = asyncio.Event()
            task = asyncio.current_task()
            if task:

                def finished(_):
                    ready.set()
                    if self.turns.get(key) is ready:
                        self.turns.pop(key, None)

                task.add_done_callback(finished)

    def end_turn(self, event) -> None:
        ready = self.turns.pop(id(event), None)
        if ready:
            ready.set()

    async def _persist(self) -> None:
        # Called under mutation_lock. One atomic replacement for the complete registry.
        payload = json.dumps(
            {"sequence": self.sequence, "jobs": self.jobs}, ensure_ascii=False, indent=2
        )
        task = asyncio.create_task(asyncio.to_thread(self._write_state, payload))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task  # Complete publication before unloading or another mutation.
            raise

    def _write_state(self, payload: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.path)

    async def _update(self, job: dict, **changes) -> None:
        async with self.mutation_lock:
            job.update(changes, updated_at=time.time())
            await self._persist()

    async def start(self) -> None:
        async with self.mutation_lock:
            if self.started or self.closed:
                return
            for job in self.jobs.values():
                self.shared.setdefault("last_job_ids", {})[job["umo"]] = job["id"]
                if job.get("prompt_id"):
                    self.shared.setdefault("last_prompt_ids", {})[job["umo"]] = job[
                        "prompt_id"
                    ]
                if (
                    job.get("paths")
                    and job.get("images_sent")
                    and Path(job["paths"][0]).is_file()
                ):
                    self.shared.setdefault("last_image_paths", {})[job["umo"]] = job[
                        "paths"
                    ][0]
                if job["state"] == "submitting" and not job.get("prompt_id"):
                    job.update(
                        state="uncertain",
                        error="插件在提交期间退出，任务可能已被 ComfyUI 接收；请检查服务端队列，避免重复生成。",
                    )
                if job.get("delivery_state") == "sending":
                    job.update(
                        delivery_state="unknown",
                        delivery_error="发送期间插件退出，送达结果待确认；为避免重复发送，未自动重发。",
                    )
                if job["state"] in ACTIVE:
                    self.pending.put_nowait(job["id"])
                elif job.get("delivery_state", "pending") == "pending":
                    self._schedule_delivery(job)
            await self._persist()
            self.started = True
            for _ in range(self.concurrency):
                self._spawn(self._worker())

    async def enqueue(
        self,
        workflow: dict,
        event,
        *,
        history: dict | None = None,
        saved_recipe: dict | None = None,
    ) -> str:
        if self.closed:
            return "加入队列失败：插件正在卸载。"
        await self.start()
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        if not umo:
            return "加入队列失败：无法确定原会话。"
        manager = self.context.conversation_manager
        cid = await manager.get_curr_conversation_id(umo)
        async with self.mutation_lock:
            if self.closed:
                return "加入队列失败：插件正在卸载。"
            if (
                sum(
                    job["state"] in ACTIVE
                    or job.get("delivery_state") in {"pending", "sending"}
                    for job in self.jobs.values()
                )
                >= self.max_pending
            ):
                return f"加入队列失败：生成队列已满（{self.max_pending} 个任务），请稍后再试。"
            self.sequence += 1
            identifier = f"C{self.sequence:06d}"
            job = {
                "id": identifier,
                "state": "queued",
                "umo": umo,
                "cid": cid,
                "workflow": copy.deepcopy(workflow),
                "history": copy.deepcopy(history or {}),
                "saved_recipe": copy.deepcopy(saved_recipe),
                "prompt_id": "",
                "created_at": time.time(),
                "updated_at": time.time(),
                "paths": [],
                "delivery_state": "pending",
                "images_sent": 0,
                "sender_id": str(event.get_sender_id())
                if hasattr(event, "get_sender_id")
                else "",
                "sender_name": event.get_sender_name()
                if hasattr(event, "get_sender_name")
                else "",
                "sender_role": getattr(event, "role", "member"),
                "self_id": event.get_self_id() if hasattr(event, "get_self_id") else "",
                "group_id": event.get_group_id()
                if hasattr(event, "get_group_id")
                else "",
                "message_timestamp": getattr(getattr(event, "message_obj", None), "timestamp", None),
                "source_request": str(getattr(event, "message_str", "") or ""),
                "message_id": str(
                    getattr(getattr(event, "message_obj", None), "message_id", "")
                ),
                "platform_name": str(event.get_platform_name())
                if hasattr(event, "get_platform_name")
                else umo.split(":", 1)[0],
            }
            retired = [
                key
                for key, item in self.jobs.items()
                if item["state"] in TERMINAL
                and item.get("delivery_state") in {"sent", "failed", "unknown"}
            ]
            for key in retired[:-199]:
                self.jobs.pop(key)
            self.jobs[identifier] = job
            try:
                await self._persist()
            except BaseException:
                self.jobs.pop(identifier, None)
                raise
            self.events[identifier] = (event, self.turns.get(id(event)))
            self.shared.setdefault("last_job_ids", {})[umo] = identifier
            self.pending.put_nowait(identifier)
        return "已加入生成队列，完成后自动发送图片并通知。可以继续聊天，无需轮询。\n" + task_reference(job)

    def find(self, identifier: str, umo: str) -> dict | None:
        if not identifier:
            identifier = self.shared.get("last_job_ids", {}).get(umo, "")
            if not identifier:
                identifier = next(
                    (
                        key
                        for key, job in reversed(self.jobs.items())
                        if job["umo"] == umo
                    ),
                    "",
                )
        job = self.jobs.get(identifier.upper())
        if job is None:
            job = next(
                (
                    job
                    for job in self.jobs.values()
                    if job.get("prompt_id") == identifier
                ),
                None,
            )
        return job if job and job["umo"] == umo else None

    def describe(self, job: dict, *, include_ids: bool = True, include_notice: bool = True) -> str:
        result = f"任务：{LABELS.get(job['state'], job['state'])}"
        if include_ids:
            result += "\n" + task_reference(job, include_notice=include_notice)
        if job.get("paths"):
            result += f"\n输出图片：{len(job['paths'])} 张"
        for key in ("error", "note", "delivery_error", "history_error"):
            if job.get(key):
                result += "\n" + job[key]
        if job["state"] in TERMINAL:
            result += "\n通知状态：" + job.get("delivery_state", "pending")
        return result if include_ids else hide_task_ids(result, [job.get("prompt_id", "")])

    async def action(self, action: str, identifier: str, umo: str, *, include_ids: bool = False) -> str:
        if action == "queue":
            jobs = [
                job
                for job in self.jobs.values()
                if job["umo"] == umo
                and (
                    job["state"] in ACTIVE
                    or job.get("delivery_state") in {"pending", "sending"}
                )
            ]
            return "本会话生成队列：\n" + (
                "\n".join(self.describe(job, include_ids=include_ids, include_notice=False) for job in jobs) or "暂无任务。"
            ) + ("\n" + INTERNAL_ID_NOTICE if include_ids and jobs else "")
        job = self.find(identifier, umo)
        if job is None:
            return "查询失败：当前会话没有该任务。"
        if action == "cancel":
            async with self.mutation_lock:
                if job["state"] == "queued":
                    job.update(state="cancelled", error="任务在提交前已取消。")
                    await self._persist()
                elif job["state"] == "submitting":
                    return "任务正在提交，暂时无法确认取消；请稍后查询。"
                elif job["state"] == "downloading":
                    return "图片已生成，正在回收结果；无需中断 GPU 任务。"
                elif job["state"] not in ACTIVE:
                    return self.describe(job, include_ids=include_ids)
                else:
                    # Persist intent; worker interprets the backend interruption as cancellation.
                    job["cancel_requested"] = True
                    await self._persist()
            if job["state"] == "cancelled":
                self._schedule_delivery(job)
            else:
                queue = await self.client.get_queue()
                if queue is None:
                    await self._update(job, cancel_requested=False)
                    return "取消请求失败：无法确认 ComfyUI 队列，任务仍在跟踪中。"
                pending = [
                    str(item[1])
                    for item in queue.get("queue_pending", [])
                    if isinstance(item, (tuple, list)) and len(item) > 1
                ]
                running = [
                    str(item[1])
                    for item in queue.get("queue_running", [])
                    if isinstance(item, (tuple, list)) and len(item) > 1
                ]
                if job["prompt_id"] in pending:
                    if not await self.client.delete_queue_items([job["prompt_id"]]):
                        await self._update(job, cancel_requested=False)
                        return "移除排队任务失败，任务仍在跟踪中。"
                    await self._update(
                        job,
                        state="cancelled",
                        error="任务已从 ComfyUI 待执行队列移除。",
                    )
                    self._schedule_delivery(job)
                elif job["prompt_id"] in running:
                    if not await self.client.interrupt(prompt_id=job["prompt_id"]):
                        await self._update(job, cancel_requested=False)
                        return "取消请求失败，任务仍在跟踪中。"
                else:
                    await self._update(job, cancel_requested=False)
                    return "任务已离开 ComfyUI 队列，后台正在确认结果；未发出中断请求。"
            return f"已请求取消任务 {job['id']}。" if include_ids else "已请求取消任务。"
        # Queue jobs never hold another LLM tool invocation open.
        return self.describe(job, include_ids=include_ids)

    async def _worker(self) -> None:
        while True:
            identifier = await self.pending.get()
            job = self.jobs[identifier]
            try:
                if job["state"] in ACTIVE:
                    await self._generate(job)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(f"[ComfyUIDirect] 任务 {identifier} 失败: {error}")
                await self._update(
                    job,
                    state="uncertain"
                    if job["state"] == "submitting" and not job.get("prompt_id")
                    else "failed",
                    error=str(error),
                )
            finally:
                self.pending.task_done()
            if job["state"] in TERMINAL:
                self._schedule_delivery(job)

    async def _generate(self, job: dict) -> None:
        if not job.get("prompt_id"):
            async with self.mutation_lock:
                if job["state"] != "queued":
                    return
                job.update(state="submitting", updated_at=time.time())
                await self._persist()
            pid, error = await self.client.submit_prompt_detail(job["workflow"])
            if not pid or error:
                await self._update(
                    job,
                    state="uncertain" if error and "可能" in error else "failed",
                    error=error or "ComfyUI 未返回任务编号。",
                )
                return
            await self._update(job, prompt_id=pid, state="running")
            self.shared.setdefault("last_prompt_ids", {})[job["umo"]] = pid
        pid = job["prompt_id"]
        if job["state"] != "downloading":
            while True:
                entry, error = await self.client.wait_for_history(
                    pid, timeout=min(30, self.client.timeout)
                )
                if job["state"] == "cancelled":
                    return
                if entry is not None:
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        await self._update(
                            job,
                            state="cancelled"
                            if job.get("cancel_requested")
                            else "failed",
                            error=execution_error_message(status, "详见 ComfyUI 日志"),
                        )
                        return
                    images = [
                        image
                        for output in entry.get("outputs", {}).values()
                        for image in output.get("images", [])
                    ]
                    finals = [
                        image for image in images if image.get("type") == "output"
                    ]
                    if not images:
                        await self._update(
                            job, state="failed", error="工作流完成，但没有输出图片。"
                        )
                        return
                    await self._update(
                        job, state="downloading", images=finals or images
                    )
                    break
                # A wait timeout does not mean the GPU job failed. Continue with its original ID.
                if error and not error.startswith("等待超时"):
                    raise RuntimeError(error)
                await asyncio.sleep(1)
        for image in job["images"][len(job["paths"]) :]:
            filename = str(image.get("filename") or "")
            if not filename:
                raise ValueError("输出图片缺少文件名。")
            content = None
            for attempt in range(3):
                content = await self.client.download_image(
                    filename,
                    image.get("subfolder", ""),
                    image_type=image.get("type", "output"),
                )
                if content:
                    break
                if attempt < 2:
                    await asyncio.sleep(attempt + 1)
            if not content:
                raise RuntimeError(f"图片已生成但下载失败（{filename}）。")
            path = await asyncio.to_thread(
                save_image, self.output_dir, filename, content
            )
            await self._update(job, paths=[*job["paths"], str(path)])
        history = {
            **job["history"],
            "prompt_id": pid,
            "task_id": job["id"],
            "filename": job["images"][0]["filename"],
            "local_path": job["paths"][0],
            "local_paths": job["paths"],
        }
        await asyncio.to_thread(self.store.save_history, history)
        if job.get("saved_recipe"):
            try:
                await asyncio.to_thread(self.store.save, job["saved_recipe"])
                if callable(self.on_schema_change):
                    self.on_schema_change()
                await self._update(
                    job, note=f"已保存快捷配方「{job['saved_recipe']['name']}」。"
                )
            except (OSError, ValueError) as error:
                await self._update(job, note=f"图片已完成，保存配方失败：{error}")
        await self._update(job, state="completed", workflow={})

    def _schedule_delivery(self, job: dict) -> None:
        if job["id"] not in self.delivering and job.get("delivery_state") == "pending":
            self.delivering.add(job["id"])
            self._spawn(self._deliver(job))

    @asynccontextmanager
    async def _session_guard(self, umo: str):
        try:
            from astrbot.core.utils.session_lock import session_lock_manager
        except ImportError:
            # Older SDKs can still serialize the plugin's own completion notices.
            async with self.session_locks.setdefault(umo, asyncio.Lock()):
                yield
        else:
            async with session_lock_manager.acquire_lock(umo):
                yield

    async def _send(self, job: dict, chain: MessageChain) -> None:
        found = await self.context.send_message(job["umo"], chain)
        if found is False:
            original = self.events.get(job["id"])
            if original is None:
                raise RuntimeError("无法找到原会话的平台。")
            await original[0].send(chain)

    async def _reply(
        self, job: dict, summary: str, event: CompletionEvent
    ) -> LLMResponse | None:
        from astrbot.api.provider import ProviderRequest
        from .hooks import begin_completion
        from astrbot.core.pipeline.context import call_event_hook
        from astrbot.core.star.star_handler import EventType

        if not self.notify_llm:
            logger.info(f"[ComfyUIDirect] 任务 {job['id']} 完成通知 LLM 跳过：配置关闭。")
            return None
        manager = self.context.conversation_manager
        cid = job.get("cid")
        if not cid or await manager.get_curr_conversation_id(job["umo"]) != cid:
            # A new/deleted conversation must not inherit an old job's history.
            logger.info(f"[ComfyUIDirect] 任务 {job['id']} 完成通知 LLM 跳过：原对话已切换或不存在。")
            return None
        conversation = await manager.get_conversation(job["umo"], cid)
        if conversation is None:
            logger.info(f"[ComfyUIDirect] 任务 {job['id']} 完成通知 LLM 跳过：原对话已删除。")
            return None
        config = event.config
        settings = config.get("provider_settings") or {}
        persona_manager = self.context.persona_manager
        _, persona, _, _ = await persona_manager.resolve_selected_persona(
            umo=job["umo"],
            conversation_persona_id=conversation.persona_id,
            platform_name=job.get("platform_name", job["umo"].split(":", 1)[0]),
            provider_settings=settings,
        )
        system = (persona or {}).get("prompt", "")
        # 临时通知指令放在末尾事件消息，保留人格与历史消息的稳定前缀。
        prompt = "[ComfyUI 后台任务事件]\n" + summary + "\n请结合原请求自然、简短地回应；图片发送状态以本事件为准，无需再次生图或发送图片。"
        request = ProviderRequest(
            prompt=prompt,
            session_id=job["umo"],
            contexts=json.loads(conversation.history),
            system_prompt=system,
            conversation=conversation,
        )
        event.set_extra("provider_request", request)
        waiting = getattr(EventType, "OnWaitingLLMRequestEvent", None)
        if waiting is not None and await call_event_hook(event, waiting):
            logger.info(f"[ComfyUIDirect] 任务 {job['id']} 完成通知 LLM 跳过：等待请求 hook 拦截。")
            return None
        if await call_event_hook(event, EventType.OnLLMRequestEvent, request):
            logger.info(f"[ComfyUIDirect] 任务 {job['id']} 完成通知 LLM 跳过：请求 hook 拦截。")
            return None
        if await begin_completion(self.context, event, request):
            logger.info(f"[ComfyUIDirect] 任务 {job['id']} 完成通知 LLM 跳过：Agent 开始 hook 拦截。")
            return None
        options = {}
        if request.model:
            options["model"] = request.model
        if request.extra_user_content_parts:
            options["extra_user_content_parts"] = request.extra_user_content_parts
        provider_id = await self.context.get_current_chat_provider_id(job["umo"])
        logger.info(f"[ComfyUIDirect] 任务 {job['id']} 触发完成通知 LLM：prompt_id={job.get('prompt_id', '')} "
                    f"会话={job['umo']} 对话={cid} provider={provider_id} 图片={job.get('images_sent', 0)} 张。")
        started = time.monotonic()
        response = await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=request.prompt,
            image_urls=request.image_urls,
            audio_urls=request.audio_urls,
            contexts=request.contexts,
            system_prompt=request.system_prompt,
            **options,
        )
        text = str(response.completion_text or "").strip()
        logger.info(f"[ComfyUIDirect] 任务 {job['id']} 完成通知 LLM 返回：耗时={time.monotonic() - started:.2f}s "
                    f"回复={len(text)} 字符。")
        if not text:
            raise RuntimeError("LLM 未返回通知文本。")
        return response

    async def _deliver(self, job: dict) -> None:
        event = None
        try:
            from astrbot.api.provider import LLMResponse
            from .reply import CompletionEvent, send_completion_reply
            from .transport import capture_completion_sends

            original = self.events.get(job["id"])
            if original and original[1]:
                await original[1].wait()
            async with self.delivery_slots, self._session_guard(job["umo"]):
                await self._update(job, delivery_state="sending")
                for path in job["paths"][job["images_sent"] :]:
                    await self._send(job, MessageChain().file_image(path))
                    await self._update(job, images_sent=job["images_sent"] + 1)
                if job["paths"]:
                    self.shared.setdefault("last_image_paths", {})[job["umo"]] = job[
                        "paths"
                    ][0]
                logger.info(f"[ComfyUIDirect] 任务 {job['id']} 图片发送结束：{job['images_sent']}/{len(job['paths'])} 张，"
                            f"生成状态={job['state']}。")
                notice = f"生图任务{LABELS[job['state']]}。"
                if job["paths"]:
                    notice += f"图片已发送，共 {len(job['paths'])} 张。"
                if job.get("error"):
                    notice += "\n" + job["error"]
                if job.get("note"):
                    notice += "\n" + job["note"]
                notice = hide_task_ids(notice, [job.get("prompt_id", "")])
                summary = notice + "\n" + task_reference(job)
                event = CompletionEvent(
                    job,
                    summary,
                    self.context.get_config(umo=job["umo"]),
                    lambda chain: self._send(job, chain),
                    original=original[0] if original else None,
                )
                event.set_extra("comfyui_backend_ids", [item.get("prompt_id", "")
                                                       for item in self.jobs.values() if item["umo"] == job["umo"]])
                with capture_completion_sends(self.context, event):
                    try:
                        response = await asyncio.wait_for(
                            self._reply(job, summary, event), self.llm_timeout
                        )
                    except Exception as error:
                        detail = str(error) or type(error).__name__
                        logger.warning(
                            f"[ComfyUIDirect] 任务 {job['id']} LLM 通知失败，使用任务结果: {detail}"
                        )
                        response = None
                        job["llm_error"] = detail
                    from_llm = response is not None
                    if response is None and not event.is_stopped():
                        response = LLMResponse(role="assistant", completion_text=notice)
                    text = await send_completion_reply(
                        self.context, event, response, from_llm=from_llm
                    )
                outcome = "文本已发送" if text else "回复已被插件处理或清空"
                logger.info(f"[ComfyUIDirect] 任务 {job['id']} 完成通知处理结束：{outcome}，"
                            f"来源={'LLM' if from_llm else '任务结果'}，送达文本={len(text)} 字符。")
                if text and job.get("cid"):
                    job["history_pair"] = [
                        {
                            "role": "user",
                            "content": "[ComfyUI 后台任务事件]\n" + summary,
                        },
                        {"role": "assistant", "content": text},
                    ]
                await self._update(job, delivery_state="sent")
                if job.get("history_pair"):
                    manager = self.context.conversation_manager
                    if await manager.get_curr_conversation_id(job["umo"]) == job["cid"]:
                        try:
                            await manager.add_message_pair(
                                job["cid"], *job["history_pair"]
                            )
                        except Exception as error:
                            await self._update(job, history_error=str(error))
                            logger.warning(
                                f"[ComfyUIDirect] 任务 {job['id']} 通知已发送，历史写入失败: {error}"
                            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(f"[ComfyUIDirect] 任务 {job['id']} 通知失败: {error}")
            await self._update(job, delivery_state="failed", delivery_error=str(error))
        finally:
            if event is not None:
                self.end_turn(event)
            self.delivering.discard(job["id"])
            self.events.pop(job["id"], None)

    async def close(self) -> None:
        self.closed = True
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.events.clear()
        self.turns.clear()
