"""生成任务状态、取消与输出取回。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from astrbot.api import FunctionTool
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from pydantic import ConfigDict
from pydantic import Field
from pydantic.dataclasses import dataclass
from ..generation.queue import ACTIVE
from ..generation.recovery import recover_outputs
from ..generation.receipts import INTERNAL_ID_NOTICE
from ..generation.receipts import hide_task_ids
from ..generation.receipts import mark_task_reply
from ..generation.receipts import task_reference
from ..clients.comfyui import ComfyUIClient
from ..clients.comfyui import execution_error_message
from .common import _event_scope, _last_prompt_id


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiJobTool(FunctionTool[AstrAgentContext]):
    """按 prompt_id 查询任务状态 / 等待 / 取消（等价 MCP job）。"""

    name: str = "comfyui_job"
    description: str = (
        "按任务编号或 prompt_id 查询生成任务状态或取消任务。"
        "action=status：查询状态与输出；action=wait：返回队列任务当前状态；"
        "action=cancel：中断任务；action=queue：查看本会话的生成队列；"
        "action=fetch：复用或补下载输出图片。fetch 也支持显式 prompt_id 查询后端历史输出。"
        "不传编号时查询本会话最近的任务。编号仅为内部标记，不用向用户声明或复述。插件队列任务自动通知完成，wait 也立即返回状态，无需反复查询。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "wait", "cancel", "queue", "fetch"],
                    "description": "操作：status=查状态（默认），wait=查当前状态，cancel=取消，queue=本会话队列，fetch=取回图片",
                },
                "prompt_id": {
                    "type": "string",
                    "description": "插件任务编号或 ComfyUI prompt_id，可选；不传用本会话最近任务",
                },
                "task_id": {
                    "type": "string",
                    "description": "插件任务编号；与 prompt_id 二选一",
                },
            },
        }
    )
    client: ComfyUIClient | None = None
    shared: dict = Field(default_factory=dict)

    output_dir: Path | None = None

    async def _fetch(self, context, identifier, kwargs):
        if self.output_dir is None:
            return "下载失败：插件输出目录未配置。"
        queue = self.shared.get("generation_queue")
        job = queue.find(identifier, _event_scope(context)) if queue is not None else None
        if job is not None:
            if job["state"] in ACTIVE:
                return queue.describe(job, include_ids=False) + "\n后台仍在处理，完成后可取回图片。"
            pid = job.get("prompt_id")
            images, paths = job.get("images") or [], job.get("paths") or []
        else:
            known_task = queue is not None and (identifier.upper() in queue.jobs or any(
                item.get("prompt_id") == identifier for item in queue.jobs.values()))
            if known_task or kwargs.get("task_id") or (queue is not None and not kwargs.get("prompt_id")):
                return "下载失败：当前会话没有该任务。"
            pid = str(kwargs.get("prompt_id") or "").strip() or _last_prompt_id(self.shared, context)
            images, paths = [], []
        if not pid:
            return "下载失败：任务尚未提交或没有可用的 prompt_id。"
        cache = self.shared.setdefault("fetched_outputs", {})
        previous = cache.get(pid)
        if previous:
            images, paths = previous["images"], previous["paths"]
        images, paths, error = await recover_outputs(
            self.client, self.output_dir, pid, images=images, paths=paths
        )
        cache[pid] = {"images": images, "paths": paths}
        if error:
            return "补下载失败：" + hide_task_ids(error, [pid])
        if job is not None and paths != job.get("paths"):
            await queue._update(job, paths=paths)
        return "输出图片本地路径：\n" + "\n".join(paths)

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        identifier = str(kwargs.get("task_id") or kwargs.get("prompt_id") or "").strip()
        action = str(kwargs.get("action") or "status").strip().lower()
        mark_task_reply(context, self.shared, identifiers=[identifier])
        result = await self._call(context, **kwargs)
        if INTERNAL_ID_NOTICE in result:
            return result
        queue = self.shared.get("generation_queue")
        if action == "queue":
            reference = INTERNAL_ID_NOTICE
        elif queue is not None:
            job = queue.find(identifier, _event_scope(context))
            if job is None and kwargs.get("prompt_id") and action == "fetch":
                job = {"prompt_id": identifier}
            reference = task_reference(job or {})
        else:
            reference = task_reference({"prompt_id": identifier or _last_prompt_id(self.shared, context)})
        return result + "\n" + reference

    async def _call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        action = str(kwargs.get("action") or "status").strip().lower()
        queue = self.shared.get("generation_queue")
        identifier = str(kwargs.get("task_id") or kwargs.get("prompt_id") or "").strip()
        if action not in {"status", "wait", "cancel", "queue", "fetch"}:
            return "操作无效：支持 status / wait / cancel / queue / fetch。"
        if action == "fetch":
            return await self._fetch(context, identifier, kwargs)
        if queue is not None:
            return await queue.action(action, identifier, _event_scope(context), include_ids=action == "queue")
        pid = str(kwargs.get("prompt_id") or "").strip() or _last_prompt_id(
            self.shared, context
        )

        if action == "queue":
            queue = await self.client.get_queue()
            if queue is None:
                return "查询失败：无法连接 ComfyUI。"
            running = [str(item[1]) for item in queue.get("queue_running") or [] if isinstance(item, (list, tuple)) and len(item) > 1]
            pending = [str(item[1]) for item in queue.get("queue_pending") or [] if isinstance(item, (list, tuple)) and len(item) > 1]
            return f"运行中: {len(running)} | 待执行: {len(pending)}"

        if not pid:
            return "查询失败：没有可用的 prompt_id（先运行一次生成，或显式传入 prompt_id）。"

        if action == "cancel":
            ok = await self.client.interrupt(prompt_id=pid)
            return "已请求取消任务。" if ok else "取消失败：无法连接 ComfyUI。"

        if action == "wait":
            while True:
                entry, wait_error = await self.client.wait_for_history(pid)
                if wait_error:
                    return hide_task_ids(wait_error, [pid])
                if entry is not None:
                    st = entry.get("status") or {}
                    if st.get("status_str") == "error":
                        return "任务失败：" + hide_task_ids(execution_error_message(st, "详见 ComfyUI 日志"), [pid])
                    outputs = entry.get("outputs", {})
                    n = sum(len(o.get("images", [])) for o in outputs.values())
                    return f"任务已完成，输出图片 {n} 张。"
                await asyncio.sleep(2)
            return f"任务等待超时（{int(self.client.timeout)}s）。"

        entry = await self.client.get_history_entry(pid)
        if entry is None:
            # 可能还在排队/运行中
            queue = await self.client.get_queue()
            if queue is not None:
                running = [str(item[1]) for item in queue.get("queue_running") or [] if isinstance(item, (list, tuple)) and len(item) > 1]
                pending = [str(item[1]) for item in queue.get("queue_pending") or [] if isinstance(item, (list, tuple)) and len(item) > 1]
                if pid in running:
                    return "任务正在运行中。"
                if pid in pending:
                    return "任务在待执行队列中。"
            return "任务状态未知（可能已过期或不存在）。"
        st = entry.get("status") or {}
        if st.get("status_str") == "error":
            return "任务失败：" + hide_task_ids(execution_error_message(st, "详见 ComfyUI 日志"), [pid])
        outputs = entry.get("outputs", {})
        images = []
        for node_out in outputs.values():
            images.extend(node_out.get("images", []))
        if not images:
            return "任务已完成，无图片输出。"
        f = images[0]
        return f"任务已完成。图片: {f.get('filename')}（subfolder: {f.get('subfolder', '')}）"
