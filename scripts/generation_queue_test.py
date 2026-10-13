"""Offline end-to-end queue regressions with blocked GPU and LLM responses."""

from __future__ import annotations

import asyncio
import copy
import errno
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import smoke_test  # noqa: F401 - shared AstrBot stubs
from astrbot.api.event import AstrMessageEvent

# Queue scheduling remains testable without the SDK; the SDK script exercises
# the real completion event, hook registry, result decoration and delivery.
if not hasattr(AstrMessageEvent, "make_result"):
    from contextlib import nullcontext

    # 隔离测试只验证调度；直接发送捕获由真实 SDK 集成测试覆盖。
    sys.modules['comfyflow.generation.transport'] = types.SimpleNamespace(
        capture_completion_sends=lambda context, event: nullcontext()
    )

    class Entity(types.SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(
                **{
                    "image_urls": [],
                    "audio_urls": [],
                    "extra_user_content_parts": [],
                    "model": None,
                    **kwargs,
                }
            )

    class CompletionEvent:
        def __init__(self, job, summary, config, send, original=None):
            self.send = send
            self.config = config

        def set_extra(self, *args):
            pass

        def is_stopped(self):
            return False

    async def call_event_hook(*args):
        return False

    async def send_completion_reply(context, event, response, **kwargs):
        await event.send(Chain().message(response.completion_text))
        return response.completion_text

    sys.modules["astrbot.api.provider"] = types.SimpleNamespace(
        LLMResponse=Entity, ProviderRequest=Entity
    )
    sys.modules["astrbot.core.pipeline.context"] = types.SimpleNamespace(
        call_event_hook=call_event_hook
    )
    sys.modules["astrbot.core.star.star_handler"] = types.SimpleNamespace(
        EventType=types.SimpleNamespace(OnLLMRequestEvent="request")
    )
    sys.modules['comfyflow.generation.reply'] = types.SimpleNamespace(
        CompletionEvent=CompletionEvent, send_completion_reply=send_completion_reply
    )

from astrbot.api.provider import LLMResponse  # noqa: E402
from comfyflow.generation import queue as generation_queue
from comfyflow.clients.comfyui import ComfyUIClient
from comfyflow.generation.queue import GenerationQueue
from comfyflow.workflows.registry import (
    EditWorkflowRegistry,
    ModelFamilyRegistry,
    WorkflowProfileStore,
)
from comfyflow.storage.recipes import RecipeStore
from comfyflow.workflows.slots import detect_slots
from comfyflow.tools import (
    ComfyuiDrawTool,
    ComfyuiEditTool,
    ComfyuiJobTool,
    ComfyuiRecipeDrawTool,
    ComfyuiValidateWorkflowTool,
)
from comfyflow.workflows.builder import WorkflowBuilder

PLUGIN = Path(__file__).resolve().parents[1]
PNG = b"\x89PNG\r\n\x1a\nqueue-result"
QUEUES = []


class Chain:
    def __init__(self):
        self.items = []

    def file_image(self, path):
        self.items.append(("image", path))
        return self

    def message(self, text):
        self.items.append(("text", text))
        return self


generation_queue.MessageChain = Chain


class Conversations:
    def __init__(self):
        self.current = {"test:GroupMessage:A": "cid-A", "test:GroupMessage:B": "cid-B"}
        self.history = {
            "cid-A": [{"role": "user", "content": "draw a cat"}],
            "cid-B": [],
        }
        self.pairs = []

    async def get_curr_conversation_id(self, umo):
        return self.current.get(umo)

    async def get_conversation(self, umo, cid):
        if cid not in self.history:
            return None
        return types.SimpleNamespace(
            cid=cid, history=json.dumps(self.history[cid]), persona_id="friendly"
        )

    async def add_message_pair(self, cid, user, assistant):
        self.pairs.append((cid, user, assistant))
        self.history[cid].extend([user, assistant])


if hasattr(AstrMessageEvent, "make_result"):
    from astrbot.api.star import Context as BaseContext
else:
    BaseContext = object


class Context(BaseContext):
    def __init__(self):
        self.conversation_manager = Conversations()
        self.persona_manager = self
        self.sent, self.requests = [], []
        self.llm_error = False
        self.llm_gate = None
        self.send_error = False

    def get_config(self, **kwargs):
        if not hasattr(AstrMessageEvent, "make_result"):
            return {"provider_settings": {"default_personality": "friendly"}}
        from astrbot.core.config.default import DEFAULT_CONFIG

        config = copy.deepcopy(DEFAULT_CONFIG)
        config["provider_settings"]["default_personality"] = "friendly"
        config["provider_tts_settings"]["enable"] = False
        config["t2i"] = False
        config["platform_settings"]["reply_with_mention"] = False
        config["platform_settings"]["reply_with_quote"] = False
        config["platform_settings"]["segmented_reply"]["enable"] = False
        return config

    def get_using_tts_provider(self, umo=None):
        return None

    async def get_using_tts_provider_async(self, umo=None):
        return None

    async def resolve_selected_persona(self, **kwargs):
        return "friendly", {"prompt": "friendly persona"}, None, False

    async def get_current_chat_provider_id(self, umo):
        return "original-provider"

    async def llm_generate(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        if self.llm_gate:
            await self.llm_gate.wait()
        if self.llm_error:
            raise RuntimeError("provider offline")
        return LLMResponse(role="assistant", completion_text="Your image is ready!")

    async def send_message(self, umo, chain):
        if self.send_error:
            raise RuntimeError("platform offline")
        if hasattr(chain, "items"):
            items = chain.items
        else:
            items = [("text", comp.text) for comp in chain.chain]
        self.sent.append((umo, items))
        return True


class Client(ComfyUIClient):
    def __init__(self):
        self.timeout = 0.04
        self.submissions, self.downloads = [], []
        self.finished = asyncio.Event()
        self.fail = False
        self.uncertain = False
        self.pending_backend = False
        self.cancelled = []
        self.waits = 0
        self.uploads = []
        self.download_error = False

    async def submit_prompt_detail(self, workflow):
        self.submissions.append(copy.deepcopy(workflow))
        if self.uncertain:
            return None, "提交超时，任务可能已在排队"
        return f"pid-{len(self.submissions)}", None

    async def wait_for_history(self, pid, timeout=None):
        self.waits += 1
        try:
            await asyncio.wait_for(self.finished.wait(), timeout)
        except TimeoutError:
            return None, "等待超时，任务可能仍在执行"
        if self.fail:
            return {
                "status": {
                    "status_str": "error",
                    "messages": [
                        ["execution_error", {"exception_message": "GPU failure"}]
                    ],
                }
            }, None
        return {
            "status": {"status_str": "success"},
            "outputs": {
                "preview": {"images": [{"filename": "preview.webp", "type": "temp"}]},
                "final": {
                    "images": [
                        {"filename": "one.png", "type": "output", "subfolder": "batch"},
                        {"filename": "two.png", "type": "output", "subfolder": "batch"},
                    ]
                },
            },
        }, None

    async def download_image(self, filename, subfolder="", image_type="output"):
        self.downloads.append((filename, subfolder, image_type))
        if self.download_error:
            return None
        return PNG + filename.encode()

    async def generation_resources(self):
        return {}

    async def upload_image(self, filename, content):
        self.uploads.append((filename, content))
        return "uploaded.png", None

    async def get_queue(self):
        return {
            "queue_pending": [[1, "pid-1"]] if self.pending_backend else [],
            "queue_running": [],
        }

    async def delete_queue_items(self, identifiers):
        self.cancelled.extend(identifiers)
        return True

    async def interrupt(self, prompt_id=None):
        raise AssertionError("must not interrupt an unrelated running task")


def event(umo="test:GroupMessage:A", images=None):
    return types.SimpleNamespace(
        unified_msg_origin=umo,
        message_obj=types.SimpleNamespace(message=images or []),
        get_sender_id=lambda: "user-A",
    )


async def until(predicate, timeout=4):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


def make_queue(root, context=None, client=None, **kwargs):
    context, client = context or Context(), client or Client()
    shared = {}
    queue = GenerationQueue(
        context, client, RecipeStore(root), root / "output", root, shared, **kwargs
    )
    shared["generation_queue"] = queue
    QUEUES.append(queue)
    return queue, context, client, shared


async def async_receipt_and_delivery(root):
    queue, context, client, shared = make_queue(root, max_pending=2)
    origin = event()
    queue.begin_turn(origin)
    receipt = await queue.enqueue(
        {"1": {"class_type": "Demo", "inputs": {}}},
        origin,
        history={"prompt": "cat", "values": {"seed": 42}},
    )
    assert "已加入生成队列" in receipt and "task_id=C000001" in receipt
    assert "编号仅为内部标记，不用向用户声明或复述。" in receipt
    assert "生成指令：cat" in receipt
    assert shared["last_job_ids"][origin.unified_msg_origin] == "C000001"
    assert not context.sent and not context.requests
    await until(lambda: len(client.submissions) == 1)
    await queue.enqueue({}, origin)
    assert "队列已满" in await queue.enqueue({}, origin)
    assert "没有该任务" in await queue.action(
        "status", "C000001", "test:GroupMessage:B"
    )
    assert "没有该任务" in await queue.action("cancel", "pid-1", "test:GroupMessage:B")
    assert "已请求取消" in await queue.action(
        "cancel", "C000002", origin.unified_msg_origin
    )
    client.finished.set()
    await until(lambda: queue.jobs["C000001"]["state"] == "completed")
    assert not context.sent, (
        "delivery must wait until original turn has sent its receipt"
    )
    context.conversation_manager.history["cid-A"].append(
        {"role": "assistant", "content": receipt}
    )
    queue.end_turn(origin)
    await until(
        lambda: all(job["delivery_state"] == "sent" for job in queue.jobs.values())
    )
    assert len(client.submissions) == 1 and len(client.downloads) == 2
    assert all(item[1:] == ("batch", "output") for item in client.downloads)
    assert sum(kind == "image" for _, items in context.sent for kind, _ in items) == 2
    completion_request = next(
        request for request in context.requests if "已完成" in request["prompt"]
    )
    assert completion_request["system_prompt"] == "friendly persona"
    assert any(
        message["content"] == receipt for message in completion_request["contexts"]
    )
    assert completion_request["chat_provider_id"] == "original-provider"
    assert "已完成" in completion_request["prompt"]
    assert "task_id=C000001" in completion_request["prompt"]
    assert "prompt_id=pid-1" in completion_request["prompt"]
    assert "生成指令：cat" in completion_request["prompt"]
    assert "编号仅为内部标记，不用向用户声明或复述。" in completion_request["prompt"]
    history = queue.store.history_get("pid-1")
    assert history["values"]["seed"] == 42 and len(history["local_paths"]) == 2
    assert (
        shared["last_image_paths"][origin.unified_msg_origin] == history["local_path"]
    )
    sent = len(context.sent)
    queue._schedule_delivery(queue.jobs["C000001"])
    await queue.close()
    recovered, _, _, recovered_shared = make_queue(root, context=context, client=client)
    await recovered.start()
    assert (
        recovered_shared["last_image_paths"][origin.unified_msg_origin]
        == history["local_path"]
    )
    await asyncio.sleep(0.05)
    assert len(context.sent) == sent and len(client.submissions) == 1
    await recovered.close()


async def restart_and_ambiguity(root):
    queue, context, client, _ = make_queue(root)
    await queue.enqueue({}, event())
    await until(lambda: queue.jobs["C000001"]["state"] == "running")
    await queue.close()
    recovered, _, _, _ = make_queue(root, context=context, client=client)
    await recovered.start()
    client.finished.set()
    await until(lambda: recovered.jobs["C000001"]["delivery_state"] == "sent")
    assert len(client.submissions) == 1
    await recovered.close()
    state = json.loads((root / "generation_jobs.json").read_text(encoding="utf-8"))
    job = state["jobs"]["C000001"]
    job.update(
        state="submitting",
        prompt_id="",
        delivery_state="pending",
        paths=[],
        images_sent=0,
    )
    (root / "generation_jobs.json").write_text(json.dumps(state), encoding="utf-8")
    uncertain, _, _, _ = make_queue(root, context=context, client=client)
    await uncertain.start()
    await until(lambda: uncertain.jobs["C000001"]["delivery_state"] == "sent")
    assert (
        uncertain.jobs["C000001"]["state"] == "uncertain"
        and len(client.submissions) == 1
    )
    await uncertain.close()
    state["jobs"]["C000001"].update(state="completed", delivery_state="sending")
    (root / "generation_jobs.json").write_text(json.dumps(state), encoding="utf-8")
    sent = len(context.sent)
    unknown, _, _, _ = make_queue(root, context=context, client=client)
    await unknown.start()
    assert unknown.jobs["C000001"]["delivery_state"] == "unknown"
    await asyncio.sleep(0.02)
    assert len(context.sent) == sent
    await unknown.close()


async def failures_and_cancellation(root):
    for scenario in (
        "llm",
        "llm-timeout",
        "gpu",
        "download",
        "platform",
        "branch",
        "cancel",
    ):
        queue, context, client, _ = make_queue(root / scenario)
        await queue.enqueue({}, event())
        await until(lambda: len(client.submissions) == 1)
        if scenario == "llm":
            context.llm_error = True
        elif scenario == "llm-timeout":
            context.llm_gate = asyncio.Event()
            queue.llm_timeout = 0.04
        elif scenario == "gpu":
            client.fail = True
        elif scenario == "download":
            client.download_error = True
        elif scenario == "platform":
            context.send_error = True
        elif scenario == "branch":
            context.conversation_manager.current[event().unified_msg_origin] = "new-cid"
        elif scenario == "cancel":
            client.pending_backend = True
            await until(lambda: queue.jobs["C000001"]["state"] == "running")
            await queue.action("cancel", "C000001", event().unified_msg_origin)
            assert client.cancelled == ["pid-1"]
        client.finished.set()
        await until(
            lambda: queue.jobs["C000001"]["delivery_state"] in {"sent", "failed"}
        )
        job = queue.jobs["C000001"]
        if scenario in {"llm", "llm-timeout"}:
            assert job["state"] == "completed" and "llm_error" in job
            assert "生图任务已完成" in context.sent[-1][1][0][1]
            assert "C000001" not in context.sent[-1][1][0][1]
            assert "本地路径" not in context.sent[-1][1][0][1]
            if scenario == "llm-timeout":
                assert job["llm_error"] == "TimeoutError"
        elif scenario == "gpu":
            assert job["state"] == "failed" and not client.downloads
        elif scenario == "download":
            assert (
                job["state"] == "failed"
                and len(client.downloads) == 3
                and not job["paths"]
            )
        elif scenario == "platform":
            assert job["state"] == "completed" and job["delivery_state"] == "failed"
        elif scenario == "branch":
            assert not context.requests and not context.conversation_manager.pairs
        elif scenario == "cancel":
            assert job["state"] == "cancelled" and not client.downloads
        await queue.close()
    queue, context, client, _ = make_queue(root / "lost-response")
    client.uncertain = True
    await queue.enqueue({}, event())
    await until(lambda: queue.jobs["C000001"]["delivery_state"] == "sent")
    assert (
        queue.jobs["C000001"]["state"] == "uncertain" and len(client.submissions) == 1
    )
    await queue.close()


async def slow_notice_and_wait_timeout(root):
    queue, context, client, _ = make_queue(root)
    context.llm_gate = asyncio.Event()
    await queue.enqueue({}, event())
    await until(lambda: client.waits >= 1)
    await asyncio.sleep(0.08)
    assert queue.jobs["C000001"]["state"] == "running"
    client.finished.set()
    await until(lambda: len(context.requests) == 1)
    await queue.enqueue({}, event())
    await until(lambda: queue.jobs["C000002"]["state"] == "completed")
    assert len(client.submissions) == 2, (
        "slow completion LLM must not hold the GPU worker"
    )
    context.llm_gate.set()
    await until(
        lambda: all(job["delivery_state"] == "sent" for job in queue.jobs.values())
    )
    await queue.close()


async def entrypoints(root):
    queue, context, client, shared = make_queue(root)
    workflow = json.loads(
        (PLUGIN / "scripts/fixtures/mini_workflow.json").read_text(encoding="utf-8")
    )
    builder = WorkflowBuilder(
        PLUGIN, default_workflow="mini", custom_dir=root / "workflows"
    )
    builder.save_template("mini", workflow)
    families = ModelFamilyRegistry([{"name": "demo", "workflow": "mini"}])
    profiles = WorkflowProfileStore(root)
    profiles.save("mini", detect_slots(workflow))
    queue.store.save({"name": "saved", "family": "demo", "defaults": {"steps": 12}})
    origin = event()
    wrapped = types.SimpleNamespace(context=types.SimpleNamespace(event=origin))
    draw = ComfyuiDrawTool(
        client=client,
        builder=builder,
        store=queue.store,
        output_dir=root / "output",
        shared=shared,
        families=families,
        profiles=profiles,
    )
    recipe = ComfyuiRecipeDrawTool(draw_tool=draw, store=queue.store, families=families)
    assert "已加入生成队列" in await draw.call(
        wrapped, model_family="demo", prompt="cat", save_as="new recipe"
    )
    assert "已加入生成队列" in await recipe.call(wrapped, prompt="dog", recipe="saved")
    builder.save_template("legacy-mini", workflow)
    queue.store.save({
        "name": "legacy saved", "workflow": "legacy-mini",
        "slots": detect_slots(workflow), "defaults": {"steps": 9},
    })
    assert "已加入生成队列" in await recipe.call(
        wrapped, prompt="legacy recipe", recipe="legacy saved"
    )
    edit_workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
        "2": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {"prompt": "old", "image": ["1", 0]},
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {"images": ["2", 0], "filename_prefix": "edit"},
        },
    }
    builder.save_template("edit", edit_workflow)
    routes = EditWorkflowRegistry(raw=[{"name": "edit", "workflow": "edit"}])
    profiles.save(
        "edit",
        {
            "prompt": {"node": "2", "field": "prompt"},
            "source_image": {"node": "1", "field": "image"},
        },
    )
    source = root / "source.png"
    source.write_bytes(PNG)
    from astrbot.api.message_components import Image

    origin.message_obj.message = [Image(str(source))]
    edit = ComfyuiEditTool(
        client=client,
        builder=builder,
        store=queue.store,
        output_dir=root / "output",
        shared=shared,
        edit_workflows=routes,
        families=families,
        profiles=profiles,
    )
    receipt = await edit.call(wrapped, edit_workflow="edit", prompt="change sky")
    assert "已加入生成队列" in receipt, receipt
    assert client.uploads and not context.sent
    job_tool = ComfyuiJobTool(client=client, shared=shared)
    assert "C000004" in await job_tool.call(wrapped, action="status", include_ids=True)
    assert "C000002" in await job_tool.call(wrapped, action="status", task_id="C000002", include_ids=True)
    assert queue.find("", origin.unified_msg_origin)["id"] == "C000004"
    assert queue.find("C000002", origin.unified_msg_origin)["id"] == "C000002"
    client.finished.set()
    await until(
        lambda: all(job["delivery_state"] == "sent" for job in queue.jobs.values())
    )
    assert len(client.submissions) == 4
    assert len(context.requests) == 4
    for job in queue.jobs.values():
        request = next(item for item in context.requests if f"task_id={job['id']}" in item["prompt"])
        assert f"prompt_id={job['prompt_id']}" in request["prompt"]
        assert "生成指令：" + job["history"]["prompt"] in request["prompt"]
    assert all(
        workflow["6"]["inputs"]["filename_prefix"]
        == item["6"]["inputs"]["filename_prefix"]
        for item in client.submissions[:3]
    )
    assert client.submissions[3]["3"]["inputs"]["filename_prefix"] == "edit"
    assert queue.store.get("new recipe") is not None
    assert client.submissions[1]["2"]["inputs"]["text"] == "dog"
    assert client.submissions[3]["1"]["inputs"]["image"] == "uploaded.png"
    assert client.submissions[2]["2"]["inputs"]["text"] == "legacy recipe"
    assert any(row["workflow"] == "legacy-mini" for row in queue.store.list_history())
    assert any(row["entry"] == "edit" for row in queue.store.list_history())
    await queue.close()


async def workflow_loading(root):
    root.mkdir(parents=True, exist_ok=True)
    workflow = {"1": {"class_type": "Demo", "inputs": {"text": "x" * 4096}}}
    raw = json.dumps(workflow)

    class ValidationClient(Client):
        async def get_object_info(self, force_refresh=False):
            return {"Demo": {"input": {"required": {"text": ["STRING"]}}}}

    validate = ComfyuiValidateWorkflowTool(client=ValidationClient())
    with patch.object(
        Path, "is_file", side_effect=OSError(errno.ENAMETOOLONG, "File name too long")
    ) as probe:
        assert ComfyuiValidateWorkflowTool._load_wf("\n " + raw + " \n") == workflow
        assert "校验结果: valid" in await validate.call(None, workflow=raw)
        probe.assert_not_called()
        error = await validate.call(None, workflow="x" * 4096)
        assert error.startswith("校验失败：读取工作流文件失败"), error

    path = root / "workflow.json"
    path.write_text(raw, encoding="utf-8")
    assert ComfyuiValidateWorkflowTool._load_wf(str(path)) == workflow
    assert ComfyuiValidateWorkflowTool._load_wf("  ") is None
    for payload in ("[]", "null", "123", "{malformed"):
        path.write_text(payload, encoding="utf-8")
        for value in (payload, str(path)):
            try:
                ComfyuiValidateWorkflowTool._load_wf(value)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid workflow accepted: {payload}")
    print(
        "inline/file workflow parsing / Linux ENAMETOOLONG / validation tool regressions OK"
    )


async def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        try:
            await workflow_loading(root / "loading")
            await async_receipt_and_delivery(root / "delivery")
            await restart_and_ambiguity(root / "recovery")
            await failures_and_cancellation(root / "failures")
            await slow_notice_and_wait_timeout(root / "slow-notice")
            await entrypoints(root / "entrypoints")
        finally:
            await asyncio.gather(*(queue.close() for queue in QUEUES))
    print(
        "generation queue async receipts / delivery / recovery / cancellation / LLM fallback / all entrypoints OK"
    )


if __name__ == "__main__":
    asyncio.run(main())
