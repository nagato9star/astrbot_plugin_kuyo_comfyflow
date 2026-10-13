"""Opt-in GPU queue check against a supplied ComfyUI; notifications use a local sink."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import secrets
import sys
import time
import types
from pathlib import Path

PLUGIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN))


class LocalConversation:
    async def get_curr_conversation_id(self, umo):
        return "live-test"

    async def add_message_pair(self, *args):
        pass


class LocalSink:
    """No external chat messages or model calls: record actual queue deliveries locally."""

    def __init__(self):
        self.conversation_manager = LocalConversation()
        self.messages = []

    def get_config(self, **kwargs):
        from astrbot.core.config.default import DEFAULT_CONFIG

        config = copy.deepcopy(DEFAULT_CONFIG)
        config["platform_settings"]["reply_with_mention"] = False
        config["platform_settings"]["reply_with_quote"] = False
        config["platform_settings"]["segmented_reply"]["enable"] = False
        config["provider_tts_settings"]["enable"] = False
        config["t2i"] = False
        return config

    def get_using_tts_provider(self, umo=None):
        return None

    async def get_using_tts_provider_async(self, umo=None):
        return None

    async def send_message(self, umo, chain):
        from astrbot.api.message_components import Image, Plain

        items = []
        for component in chain.chain:
            if isinstance(component, Image):
                path = await component.convert_to_file_path()
                assert Path(path).is_file()
                items.append({"type": "image", "path": str(path)})
            elif isinstance(component, Plain):
                items.append({"type": "text", "text": component.text})
        self.messages.append({"umo": umo, "items": items, "at": time.time()})
        return True


async def run(args):
    # Import the real SDK only after entering the separate artifact directory.
    from comfyflow.clients.comfyui import ComfyUIClient
    from comfyflow.generation.queue import GenerationQueue
    from comfyflow.workflows.registry import ModelFamilyRegistry, WorkflowProfileStore
    from comfyflow.storage.recipes import RecipeStore
    from comfyflow.workflows.slots import detect_slots
    from comfyflow.tools import ComfyuiDrawTool
    from comfyflow.workflows.builder import WorkflowBuilder

    for name in ("httpx", "httpcore", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)

    class ObservedClient(ComfyUIClient):
        def __init__(self):
            super().__init__(args.host, args.port, timeout=300, request_timeout=15)
            self.submitted_ids = []

        async def submit_prompt_detail(self, *values, **kwargs):
            pid, error = await super().submit_prompt_detail(*values, **kwargs)
            if pid:
                self.submitted_ids.append(pid)
            return pid, error

    root = Path.cwd()
    client = ObservedClient()
    queue = None
    second_client = None
    sink = LocalSink()
    report = {
        "server": client.base_url,
        "chat_platform_delivery_tested": False,
        "llm_notification_tested": False,
        "started_at": time.time(),
    }
    try:
        remote_queue = await client.get_queue()
        assert remote_queue is not None, "ComfyUI is unreachable"
        records = await client._get("/history", {"max_items": 5})
        reference_id, entry = next(
            (pid, record)
            for pid, record in reversed(records.items())
            if (record.get("status") or {}).get("status_str") == "success"
        )
        workflow = copy.deepcopy(entry["prompt"][2])
        slots = detect_slots(workflow)
        assert all(
            role in slots for role in ("prompt", "negative", "sampler", "size")
        ), "reference must be a mapped text-to-image workflow"
        # Keep the server's loader, sampler, upscaler and accelerator settings.
        workflow[slots["negative"]["node"]]["inputs"][slots["negative"]["field"]] = ""
        report["reference_prompt_id"] = reference_id
        builder = WorkflowBuilder(
            PLUGIN, default_workflow="live", custom_dir=root / "workflows"
        )
        builder.save_template("live", workflow)
        profiles = WorkflowProfileStore(root)
        profiles.save("live", slots)
        families = ModelFamilyRegistry(
            [{"name": "live", "workflow": "live", "prompt_style": "natural"}]
        )
        store = RecipeStore(root)
        shared = {}
        queue = GenerationQueue(
            sink, client, store, root / "output", root, shared, notify_llm=False
        )
        shared["generation_queue"] = queue
        tool = ComfyuiDrawTool(
            client=client,
            builder=builder,
            store=store,
            profiles=profiles,
            families=families,
            output_dir=root / "output",
            shared=shared,
        )
        event = types.SimpleNamespace(
            unified_msg_origin="local:GroupMessage:queue-live",
            get_sender_id=lambda: "live-test",
            get_platform_name=lambda: "local",
        )
        wrapped = types.SimpleNamespace(context=types.SimpleNamespace(event=event))
        prompts = [
            "A small red cabin beside a calm lake, pine forest, warm sunrise, watercolor illustration, no text.",
            "A blue ceramic teapot on a wooden table, clean white background, soft studio lighting, product photo, no text.",
        ]
        report["receipts"] = []
        for prompt in prompts:
            started = time.monotonic()
            receipt = await tool.call(
                wrapped,
                model_family="live",
                prompt=prompt,
                width=512,
                height=512,
                steps=20,
                seed=secrets.randbelow(2**31),
            )
            elapsed = time.monotonic() - started
            assert "已加入生成队列" in receipt, receipt
            report["receipts"].append({"text": receipt, "seconds": elapsed})
            print(
                json.dumps(
                    {"receipt": receipt, "seconds": round(elapsed, 3)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        assert not sink.messages, "receipt must return before output delivery"
        async with asyncio.timeout(args.timeout):
            while not queue.jobs["C000001"].get("prompt_id"):
                if queue.jobs["C000001"]["state"] in {"failed", "uncertain"}:
                    raise RuntimeError(queue.describe(queue.jobs["C000001"]))
                await asyncio.sleep(0.1)
            original_pid = queue.jobs["C000001"]["prompt_id"]
            report["state_before_restart"] = {
                key: {"state": job["state"], "prompt_id": job["prompt_id"]}
                for key, job in queue.jobs.items()
            }
            print(
                json.dumps({"restart_with_original_prompt_id": original_pid}),
                flush=True,
            )
            await queue.close()
            await client.close()
            second_client = ObservedClient()
            shared = {}
            queue = GenerationQueue(
                sink,
                second_client,
                store,
                root / "output",
                root,
                shared,
                notify_llm=False,
            )
            await queue.start()
            last_states = None
            while any(
                job.get("delivery_state") not in {"sent", "failed", "unknown"}
                for job in queue.jobs.values()
            ):
                states = {key: job["state"] for key, job in queue.jobs.items()}
                if states != last_states:
                    print(json.dumps({"states": states}), flush=True)
                    last_states = states
                await asyncio.sleep(1)
            assert queue.jobs["C000001"]["prompt_id"] == original_pid
            all_submitted_ids = client.submitted_ids + second_client.submitted_ids
            assert len(all_submitted_ids) == len(set(all_submitted_ids)) == 2, (
                all_submitted_ids
            )
            for job in queue.jobs.values():
                assert job["state"] == "completed", queue.describe(job)
                assert job["delivery_state"] == "sent", queue.describe(job)
                from PIL import Image

                for path in job["paths"]:
                    with Image.open(path) as image:
                        image.verify()
            report.update(
                ok=True,
                recovered_original_prompt_id=True,
                submitted_ids=all_submitted_ids,
                jobs=queue.jobs,
                local_delivery_messages=sink.messages,
                finished_at=time.time(),
                execution_events={
                    pid: second_client.execution_progress(pid)
                    for pid in all_submitted_ids
                },
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "submitted_ids": all_submitted_ids,
                        "images": [
                            path for job in queue.jobs.values() for path in job["paths"]
                        ],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    except BaseException as error:
        report.update(ok=False, error=str(error))
        raise
    finally:
        if queue:
            await queue.close()
        await client.close()
        if second_client:
            await second_client.close()
        (root / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    previous_directory = Path.cwd()
    try:
        os.chdir(args.output)
        asyncio.run(run(args))
    finally:
        os.chdir(previous_directory)
