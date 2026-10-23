"""Regressions for submission, deadlines, metadata preparation and result delivery."""

from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import time
import types
from collections import OrderedDict
from pathlib import Path

import httpx
from aiohttp import web

import smoke_test  # noqa: F401 - the shared AstrBot test environment
from comfyflow.studio import api as webapi
from comfyflow.clients.comfyui import ComfyUIClient
from comfyflow.clients.events import ExecutionEvents
from comfyflow.workflows.parameters import fill_trigger_words
from comfyflow.workflows.registry import ModelFamilyRegistry, WorkflowProfileStore
from comfyflow.storage.recipes import RecipeStore
from comfyflow.workflows.slots import detect_slots
from comfyflow.tools import ComfyuiDrawTool, ComfyuiRecipeDrawTool
from comfyflow.workflows.builder import WorkflowBuilder

PLUGIN = Path(__file__).resolve().parents[1]
PNG = b"\x89PNG\r\n\x1a\noriginal-bytes"


class QuietClient(ComfyUIClient):
    def _meta_debug(self, _message):
        pass


async def submission_and_timeout():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/object_info":
            return httpx.Response(200, json={"Test": {"input": {}, "output": []}})
        if request.url.path == "/history/pid":
            assert request.extensions["timeout"]["read"] == 0.3
            assert request.extensions["timeout"]["connect"] is not None
            return httpx.Response(200, json={})
        if request.url.path != "/prompt":
            return httpx.Response(404)
        if calls.count("/prompt") == 1:
            raise httpx.ConnectError("first connect failed", request=request)
        return httpx.Response(200, json={"prompt_id": "accepted"})

    client = QuietClient("test", 8188, request_timeout=0.3)
    client._client = httpx.AsyncClient(
        base_url=client.base_url, transport=httpx.MockTransport(handler)
    )
    await client.generation_resources()
    assert client._memory_cache is not None
    assert await client.submit_prompt_detail(
        {"1": {"class_type": "Test", "inputs": {}}}
    ) == ("accepted", None)
    await client.get_history_entry("pid")
    assert calls.count("/object_info") == 1
    await client.close()

    class BlockedWarmup(QuietClient):
        async def get_object_info(self, force_refresh=False):
            await asyncio.Event().wait()

    closing = BlockedWarmup("test", 8188)
    closing.start_warmup()
    await asyncio.sleep(0)
    await closing.close()
    assert closing._warmup_task.done()
    pid, error = await closing.submit_prompt_detail({}, ui_workflow={})
    assert pid is None and "已关闭" in error

    calls.clear()

    def uncertain(request):
        calls.append(request.url.path)
        raise httpx.ReadTimeout("response lost", request=request)

    client = QuietClient("test", 8188)
    client._client = httpx.AsyncClient(
        base_url=client.base_url, transport=httpx.MockTransport(uncertain)
    )
    pid, error = await client.submit_prompt_detail(
        {}, ui_workflow={"nodes": [], "links": []}
    )
    assert pid is None and "可能" in error and calls == ["/prompt"]
    await client.close()

    class StalledClient(QuietClient):
        async def get_history_entry(self, pid):
            await asyncio.sleep(10)

    client = StalledClient("test", 8188, timeout=0.08)
    start = time.monotonic()
    entry, error = await client.wait_for_history("still-running")
    assert entry is None and "still-running" in error and "排队或执行" in error
    assert time.monotonic() - start < 0.5
    await client.close()


async def bounded_history_requests():
    for events_enabled in (False, True):
        for failure in ("blocked", "retry"):
            calls = []
            cancelled = []

            async def handler(request):
                calls.append(request.url.path)
                if len(calls) == 1:
                    if failure == "retry":
                        raise httpx.ReadTimeout("history response lost", request=request)
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cancelled.append(True)
                        raise
                return httpx.Response(200, json={
                    "pid": {"status": {"status_str": "success"}, "outputs": {}}
                })

            client = QuietClient("test", 8188, timeout=0.8, request_timeout=0.1,
                                 events_enabled=events_enabled)
            client._client = httpx.AsyncClient(
                base_url=client.base_url, transport=httpx.MockTransport(handler)
            )
            finished = None
            if events_enabled:
                # 直接投递真实事件分发器，隔离网络抖动与 WebSocket 连接建立时间。
                events = ExecutionEvents.__new__(ExecutionEvents)
                events.states = OrderedDict()
                events.signals = {}
                events.task = asyncio.create_task(asyncio.Event().wait())
                client._events = events

                async def finish():
                    await asyncio.sleep(0.02)
                    events.handle({"type": "execution_success", "data": {"prompt_id": "pid"}})

                finished = asyncio.create_task(finish())
            try:
                started = time.monotonic()
                entry, error = await client.wait_for_history("pid")
                assert entry is not None and error is None, (events_enabled, failure, error)
                assert len(calls) == 2 and time.monotonic() - started < 0.7
                assert bool(cancelled) == (failure == "blocked")
                if events_enabled:
                    assert "pid" not in events.signals
            finally:
                if finished:
                    await finished
                await client.close()

    active = set()

    class StalledClient(QuietClient):
        async def get_history_entry(self, pid):
            active.add(pid)
            try:
                await asyncio.Event().wait()
            finally:
                active.discard(pid)

    client = StalledClient("127.0.0.1", 1, timeout=0.25,
                           request_timeout=0.1, events_enabled=True)
    try:
        started = time.monotonic()
        entry, error = await client.wait_for_history("deadline")
        assert entry is None and "等待超时" in error
        assert time.monotonic() - started < 0.6 and not active
        assert "deadline" not in client._events.signals
        waiting = asyncio.create_task(client.wait_for_history("cancelled", timeout=10))
        while "cancelled" not in active:
            await asyncio.sleep(0)
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        assert waiting.cancelled() and not active
        assert "cancelled" not in client._events.signals
    finally:
        await client.close()


async def event_wakeup_and_fallback():
    ready = asyncio.Event()
    sockets = []
    histories = {}
    reads = []

    async def socket_handler(request):
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        sockets.append((request.query["clientId"], socket))
        ready.set()
        async for _message in socket:
            pass
        return socket

    async def history_handler(request):
        pid = request.match_info["pid"]
        reads.append(pid)
        if pid == "blocked-error":
            await asyncio.sleep(2)
        return web.json_response({pid: histories[pid]} if pid in histories else {})

    async def submit_handler(request):
        body = await request.json()
        assert body["client_id"] == sockets[0][0]
        return web.json_response({"prompt_id": "event-job"})

    app = web.Application()
    app.router.add_get("/ws", socket_handler)
    app.router.add_get("/history/{pid}", history_handler)
    app.router.add_post("/prompt", submit_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = QuietClient("127.0.0.1", port, timeout=3, events_enabled=True)
    client.start_events()
    try:
        await asyncio.wait_for(ready.wait(), 2)
        assert (await client.submit_prompt_detail({}, ui_workflow={}))[0] == "event-job"

        async def finish():
            await asyncio.sleep(0.1)
            histories["event-job"] = {
                "status": {"status_str": "success", "completed": True},
                "outputs": {},
            }
            await sockets[0][1].send_json(
                {"type": "execution_success", "data": {"prompt_id": "event-job"}}
            )

        finished = asyncio.create_task(finish())
        start = time.monotonic()
        entry, error = await client.wait_for_history("event-job")
        assert entry and error is None and time.monotonic() - start < 1
        await finished
        assert reads.count("event-job") == 2

        async def delayed_history():
            await asyncio.sleep(0.02)
            await sockets[0][1].send_json(
                {"type": "execution_success", "data": {"prompt_id": "delayed-history"}}
            )
            await asyncio.sleep(0.35)
            histories["delayed-history"] = {"status": {"status_str": "success"}, "outputs": {}}

        finished = asyncio.create_task(delayed_history())
        started = time.monotonic()
        entry, error = await client.wait_for_history("delayed-history")
        assert entry is not None and error is None and time.monotonic() - started < 1
        assert 3 <= reads.count("delayed-history") <= 6
        await finished
        await sockets[0][1].send_json(
            {
                "type": "execution_error",
                "data": {"prompt_id": "error-job", "exception_message": "node failed"},
            }
        )
        await asyncio.sleep(0.02)
        entry, error = await client.wait_for_history("error-job")
        assert error is None and entry["status"]["status_str"] == "error"
        assert "error-job" not in reads

        blocked = asyncio.create_task(client.wait_for_history("blocked-error"))
        await asyncio.sleep(0.05)
        await sockets[0][1].send_json(
            {
                "type": "execution_error",
                "data": {
                    "prompt_id": "blocked-error",
                    "exception_message": "error while HTTP is stalled",
                },
            }
        )
        entry, error = await asyncio.wait_for(blocked, 0.5)
        assert error is None and entry["status"]["status_str"] == "error"

        await sockets[0][1].close()

        async def finish_offline():
            await asyncio.sleep(0.1)
            histories["http-fallback"] = {
                "status": {"status_str": "success"},
                "outputs": {},
            }

        finished = asyncio.create_task(finish_offline())
        entry, error = await client.wait_for_history("http-fallback")
        assert entry and error is None and reads.count("http-fallback") >= 2
        await finished
    finally:
        await client.close()
        assert client._events.task.done()
        await runner.cleanup()


async def quick_catalog(root):
    gate = asyncio.Event()

    class CatalogClient(QuietClient):
        def __init__(self):
            super().__init__("test", 8188, cache_file=root / "models.json")
            self.object_reads = 0
            self.enriched = []

        async def get_object_info(self, force_refresh=False):
            self.object_reads += 1
            return {
                "LoraLoader": {
                    "input": {
                        "required": {
                            "lora_name": [[f"lora_{i}.safetensors" for i in range(50)]]
                        }
                    }
                }
            }

        async def get_lora_manager_catalog(self):
            await gate.wait()
            return {"lora_0.safetensors": {"trainedWords": ["catalog_word"]}}

        async def get_embeddings(self):
            return []

        async def _fetch_lora_trigger_words(self, names, **kwargs):
            self.enriched.append(list(names))
            return {
                name: {
                    "trigger_words": ["known_trigger"],
                    "trusted_trigger_words": ["known_trigger"],
                    "source": "lora_manager",
                }
                for name in names
            }

    client = CatalogClient()
    first = await client.generation_resources()
    second = await asyncio.wait_for(client.generation_resources(), 0.2)
    assert client.object_reads == 1 and len(second["lora_name"]) == 50
    await client.selected_lora_metadata(first, ["lora_49.safetensors"])
    assert client.enriched == [["lora_49.safetensors"]]
    first["lora_name"].clear()
    assert len((await client.generation_resources())["lora_name"]) == 50
    gate.set()
    await client._metadata_task
    assert (await client.generation_resources())["lora_meta"]["lora_49.safetensors"][
        "trigger_words"
    ] == ["known_trigger"]
    assert (await client.generation_resources())["lora_meta"]["lora_0.safetensors"][
        "trusted_trigger_words"
    ] == ["catalog_word"]
    await client.close()


async def idempotent_result(root):
    class ResultClient:
        downloads = 0

        async def get_history_entry(self, pid):
            return {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "1": {"images": [{"filename": "image.png", "type": "output"}]}
                },
            }

        async def download_image(self, *args, **kwargs):
            self.downloads += 1
            await asyncio.sleep(0.02)
            return PNG

    class CountingStore(RecipeStore):
        writes = 0

        def save_history(self, entry):
            self.writes += 1
            return super().save_history(entry)

    store = CountingStore(root)
    client = ResultClient()
    api = webapi.StudioApi(
        client,
        None,
        store,
        root / "output",
        {
            "web_pending_runs": {
                "pid": {
                    "prompt": "actual prompt",
                    "recipe": {"name": "actual recipe"},
                    "values": {"seed": 123},
                }
            }
        },
    )
    original_query, original_json = webapi._query, webapi._json
    webapi._query = lambda key: "pid"
    webapi._json = lambda body, *args, **kwargs: body
    try:
        results = await asyncio.gather(*(api.generate_poll() for _ in range(3)))
        assert results[0] == results[1] == results[2]
        assert client.downloads == store.writes == 1
        history = store.history_get("pid")
        assert history["prompt"] == "actual prompt" and history["values"]["seed"] == 123
        api.shared["web_result_cache"].clear()
        await api.generate_poll()
        assert (
            client.downloads == store.writes == 1
            and store.history_get("pid")["prompt"] == "actual prompt"
        )
    finally:
        webapi._query, webapi._json = original_query, original_json


async def trigger_and_snapshot(root):
    class DrawClient(QuietClient):
        def __init__(self):
            super().__init__("test", 8188, timeout=1)
            self.snapshots = 0
            self.submitted = []

        async def generation_resources(self):
            self.snapshots += 1
            return {
                "unet_name": ["base.safetensors"],
                "lora_name": ["style.safetensors"],
                "lora_meta": {
                    "style.safetensors": {
                        "model_family": "demo",
                        "trigger_words": ["known_trigger"],
                        "trusted_trigger_words": ["known_trigger"],
                        "source": "lora_manager",
                    }
                },
                "model_meta": {"base.safetensors": {"model_family": "demo"}},
            }

        async def submit_prompt_detail(self, workflow):
            self.submitted.append(copy.deepcopy(workflow))
            return f"pid-{len(self.submitted)}", None

        async def get_history_entry(self, pid):
            return {
                "status": {"status_str": "success"},
                "outputs": {
                    "6": {"images": [{"filename": "out.png", "type": "output"}]}
                },
            }

        async def download_image(self, *args, **kwargs):
            return PNG

    builder = WorkflowBuilder(
        PLUGIN, default_workflow="mini", custom_dir=root / "workflows"
    )
    workflow = json.loads(
        (PLUGIN / "scripts/fixtures/mini_workflow.json").read_text(encoding="utf-8")
    )
    builder.save_template("mini", workflow)
    slots = detect_slots(workflow)
    slots["trigger_words"] = {"node": "2", "field": "text", "mode": "append"}
    profiles = WorkflowProfileStore(root)
    profiles.save("mini", slots)
    families = ModelFamilyRegistry([{"name": "demo", "workflow": "mini"}])
    store = RecipeStore(root)
    client = DrawClient()

    async def send(chain):
        pass

    context = types.SimpleNamespace(
        context=types.SimpleNamespace(
            event=types.SimpleNamespace(unified_msg_origin="test", send=send)
        )
    )
    draw = ComfyuiDrawTool(
        client=client,
        builder=builder,
        profiles=profiles,
        families=families,
        store=store,
        output_dir=root / "output",
    )
    await draw.call(
        context,
        model_family="demo",
        prompt="visible prompt",
        model="base",
        lora="style",
    )
    assert (
        client.snapshots == 1
        and client.submitted[-1]["2"]["inputs"]["text"] == "visible prompt"
    )
    draw.auto_trigger_words = True
    await draw.call(context, model_family="demo", prompt="visible prompt", lora="style")
    assert (
        client.submitted[-1]["2"]["inputs"]["text"] == "visible prompt, known_trigger"
    )
    await draw.call(
        context,
        model_family="demo",
        prompt="visible prompt",
        lora="style",
        trigger_words="",
        save_as="blank",
    )
    assert client.submitted[-1]["2"]["inputs"]["text"] == "visible prompt"
    assert store.get("blank")["defaults"]["trigger_words"] == ""
    store.save(
        {
            "name": "automatic",
            "family": "demo",
            "defaults": {"loras": [{"name": "style.safetensors"}]},
        }
    )
    recipe = ComfyuiRecipeDrawTool(draw_tool=draw, store=store, families=families)
    await recipe.call(context, prompt="recipe prompt", recipe="automatic")
    assert client.submitted[-1]["2"]["inputs"]["text"] == "recipe prompt, known_trigger"
    await recipe.call(context, prompt="blank recipe", recipe="blank")
    assert client.submitted[-1]["2"]["inputs"]["text"] == "blank recipe"

    api = webapi.StudioApi(
        client,
        builder,
        store,
        root / "output",
        {},
        families=families,
        profiles=profiles,
        auto_trigger_words=True,
    )
    original_body, original_json = webapi._body, webapi._json

    async def body():
        return {"recipe": "automatic", "prompt": "web prompt"}

    webapi._body, webapi._json = body, lambda data, *args, **kwargs: data
    try:
        result = await api.generate()
        assert (
            result["ok"]
            and client.submitted[-1]["2"]["inputs"]["text"]
            == "web prompt, known_trigger"
        )
    finally:
        webapi._body, webapi._json = original_body, original_json
    await client.close()
    inferred = {"loras": [{"name": "guessed"}]}
    assert "trigger_words" not in fill_trigger_words(
        inferred,
        {
            "lora_meta": {
                "guessed": {"source": "tag_frequency", "trigger_words": ["wrong"]}
            }
        },
        True,
    )
    verified = fill_trigger_words(
        {"loras": [{"name": "mixed"}]},
        {
            "lora_meta": {
                "mixed": {
                    "source": "civitai",
                    "trigger_words": ["real", "guessed"],
                    "trusted_trigger_words": ["real"],
                }
            }
        },
        True,
    )
    assert verified["trigger_words"] == "real"


async def main():
    await submission_and_timeout()
    await bounded_history_requests()
    await event_wakeup_and_fallback()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        await quick_catalog(root / "catalog")
        await idempotent_result(root / "result")
        await trigger_and_snapshot(root / "draw")
    print(
        "generation pipeline retry / timeout / WebSocket / fallback / idempotency / catalog / entrypoint regressions OK"
    )


if __name__ == "__main__":
    asyncio.run(main())
