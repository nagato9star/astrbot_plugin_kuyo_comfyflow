"""验证精简后的查询、队列诊断与输出补下载。"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import smoke_test  # noqa: F401
from comfyflow.clients.animadex import AnimaDexClient
from comfyflow.clients.comfyui import ComfyUIClient
from comfyflow.clients.external import DanbooruClient, GelbooruClient
from comfyflow.generation.queue import GenerationQueue
from comfyflow.storage.images import manage_cache, save_image
from comfyflow.storage.recipes import RecipeStore
from comfyflow.tools import ComfyuiJobTool, ComfyuiLookupTool, ComfyuiSystemStatsTool


class Client(ComfyUIClient):
    resource_family_rules = []

    def __init__(self):
        self.refreshes = []
        self.folder_queries = []
        self.history_queries = []
        self.downloads = []
        self.failed_files = set()
        self.queue_queries = 0
        self.stats = {"system": {"os": "Windows", "comfyui_version": "test-version", "ram_total": 32 * 1024**3},
                      "devices": [{"name": "GPU", "vram_total": 8 * 1024**3}]}
        self.resources = {
            "unet_name": ["Anima/base.safetensors", "SDXL/base.safetensors"],
            "lora_name": ["Anima/style.safetensors", "SDXL/style.safetensors"],
            "lora_meta": {"Anima/style.safetensors": {
                "categories": ["style"], "trusted_trigger_words": ["style_trigger"],
                "trigger_words": ["style_trigger"], "recommended_weight": 0.6,
            }},
            "clip_name": ["clip_1.safetensors", "clip_2.safetensors"],
            "vae_name": ["vae.safetensors"], "embeddings": ["embedding.pt"],
        }

    async def generation_resources(self):
        return self.resources

    async def list_resources(self, *, force_refresh=False):
        self.refreshes.append(force_refresh)
        return self.resources, False

    async def list_models_folder(self, folder):
        self.folder_queries.append(folder)
        return {
            "checkpoints": ["Anima/checkpoint_only.safetensors"],
            "unet": ["Anima/unet_only.safetensors"],
            "loras": self.resources["lora_name"],
            "custom": ["nested/example.bin"],
        }.get(folder)

    async def get_history_entry(self, prompt_id):
        self.history_queries.append(prompt_id)
        return {"status": {"status_str": "success"}, "outputs": {"1": {
            "images": [{"filename": "external.png", "type": "output"}],
        }}}

    async def download_image(self, filename, subfolder="", image_type="output"):
        self.downloads.append(filename)
        return None if filename in self.failed_files else filename.encode()

    async def get_system_stats(self):
        return self.stats

    async def get_queue(self):
        self.queue_queries += 1
        return {"queue_running": [[1, "other-client-prompt"]],
                "queue_pending": [[2, "pending-backend-prompt"]]}


class Booru(DanbooruClient):
    def __init__(self, empty=False):
        self.empty, self.calls = empty, []

    async def search_artist(self, query, limit):
        self.calls.append(("artist", query, limit))
        if self.empty:
            return None
        return {"artist": "artist_name", "aliases": ["别名"],
                "posts": [{"tag_string": "watercolor sky"}, {"tag_string": "watercolor"}],
                "url": "https://example.invalid/artist"}

    async def search_character(self, query, limit):
        self.calls.append(("character", query, limit))
        if self.empty:
            return None
        return {"character": "character_name", "aliases": [],
                "posts": [{"tag_string": "blue_hair"}],
                "url": "https://example.invalid/character"}


class Gelbooru(GelbooruClient):
    def __init__(self):
        self.calls = []

    async def search_artist(self, query, limit):
        self.calls.append(("artist", query, limit))
        if not limit:
            return None
        return {"artist": "fallback_artist", "aliases": [],
                "posts": [{"tags": "watercolor"}], "url": "https://example.invalid/gelbooru"}

    async def search_character(self, query, limit):
        self.calls.append(("character", query, limit))
        return {"character": "fallback_character", "aliases": [],
                "posts": [{"tags": "blue_hair"}]}


class Animadex(AnimaDexClient):
    def __init__(self):
        self.calls = []

    async def search_characters(self, query, page=1):
        self.calls.append(("character", query, page))
        return "character reference"

    async def search_artists(self, query, page=1):
        self.calls.append(("artist", query, page))
        return "artist reference"

    async def search_copyrights(self, query, page=1):
        self.calls.append(("copyright", query, page))
        return "series reference"


async def resources_and_sources():
    client = Client()
    tool = ComfyuiLookupTool(client=client)
    models = await tool.call(None, type="model", model_family="anima")
    assert "Anima/base" in models and "SDXL/base" not in models
    assert not client.refreshes
    await tool.call(None, type="model", refresh=True)
    assert client.refreshes[-1] is True
    loras = await tool.call(None, type="lora", model_family="anima", query="style")
    assert "style_trigger" in loras and "SDXL/style" not in loras
    for kind, name in (("clip", "clip_1"), ("vae", "vae.safetensors"),
                       ("embedding", "embedding.pt")):
        assert name in await tool.call(None, type=kind)
    first = await tool.call(None, type="clip", limit=1)
    second = await tool.call(None, type="clip", limit=1, offset=1)
    assert "next_offset=1" in first and "clip_2" in second and "clip_1" not in second
    assert "CLIP: 2" in await tool.call(None, type="all")
    assert "CLIP: clip_1" in await tool.call(None, type="all", query="clip_1")
    checkpoint = await tool.call(None, type="folder", folder="checkpoints", model_family="anima")
    unet = await tool.call(None, type="folder", folder="unet", model_family="anima")
    assert "checkpoint_only" in checkpoint and "unet_only" not in checkpoint
    assert "unet_only" in unet and "checkpoint_only" not in unet
    assert "nested/example.bin" in await tool.call(None, type="folder", folder="custom")
    assert "查询失败" in await tool.call(None, type="folder", folder="missing")
    print("resource kinds / cross-category search / refresh / paging / exact folders OK")

    booru, gelbooru, animadex = Booru(), Gelbooru(), Animadex()
    tool = ComfyuiLookupTool(danbooru=booru, gelbooru=gelbooru, animadex=animadex)
    artist = await tool.call(None, type="artist", query="画师")
    assert "@artist_name" in artist and "常用 tag" not in artist
    assert booru.calls[-1][2] == 0
    detailed = await tool.call(None, type="artist", query="画师", detail=True, sample_limit=3)
    assert "watercolor(2)" in detailed and "参考:" in detailed and booru.calls[-1][2] == 3
    booru.empty = True
    fallback = await tool.call(None, type="character", query="角色", source="danbooru", detail=True)
    assert "来源 gelbooru" in fallback and "blue_hair(1)" in fallback
    for kind in ("copyright", "character", "artist"):
        await tool.call(None, type=kind, query="参考", source="animadex", page=2)
        assert animadex.calls[-1] == (kind, "参考", 2)
    assert "series reference" in await tool.call(None, type="copyright", query="作品")
    assert "来源 gelbooru" in await tool.call(
        None, type="artist", query="画师", source="gelbooru", detail="false"
    )
    assert gelbooru.calls[-1][2] == 20
    print("compact artist query / detailed tags / source fallback / AnimaDex series and paging OK")


async def outputs_and_scope(root):
    client = Client()
    output = root / "output"
    shared = {}
    queue = GenerationQueue(None, client, RecipeStore(root), output, root, shared)
    shared["generation_queue"] = queue
    context = SimpleNamespace(context=SimpleNamespace(
        event=SimpleNamespace(unified_msg_origin="session-A")
    ))
    images = [{"filename": name, "type": "output"} for name in ("one.png", "two.png")]
    paths = [str(save_image(output, image["filename"], image["filename"].encode()))
             for image in images]
    job = {"id": "C000001", "state": "completed", "umo": "session-A", "prompt_id": "pid-A",
           "images": images, "paths": paths, "delivery_state": "sent"}
    queue.jobs[job["id"]] = job
    shared["last_job_ids"] = {"session-A": job["id"]}
    tool = ComfyuiJobTool(client=client, shared=shared, output_dir=output)
    old = time.time() - 10 * 86400
    for path in paths:
        os.utime(path, (old, old))
    before_bytes = [Path(path).read_bytes() for path in paths]
    result = await tool.call(context, action="fetch")
    assert all(path in result for path in paths) and not client.downloads and not client.history_queries
    assert [Path(path).read_bytes() for path in paths] == before_bytes
    assert all(Path(path).stat().st_mtime > old for path in paths)
    assert manage_cache(output, action="expired", days=7, max_mb=0)["removed"] == 0
    Path(paths[0]).unlink()
    second_timestamp = Path(paths[1]).stat().st_mtime_ns
    restored = await tool.call(context, action="fetch", task_id=job["id"])
    assert paths[0] in restored and client.downloads == ["one.png"]
    assert Path(paths[1]).stat().st_mtime_ns >= second_timestamp
    assert Path(paths[1]).read_bytes() == before_bytes[1]
    Path(paths[0]).unlink()
    client.failed_files.add("one.png")
    failure = await tool.call(context, action="fetch", task_id=job["id"])
    assert "补下载失败" in failure and Path(paths[1]).is_file()
    client.failed_files.clear()
    recovered = await tool.call(context, action="fetch", task_id=job["id"])
    assert all(path in recovered for path in paths) and client.downloads.count("two.png") == 0
    assert job["delivery_state"] == "sent" and not queue.tasks

    await tool.call(context, action="fetch", prompt_id="external-prompt")
    previous_counts = len(client.downloads), len(client.history_queries)
    await tool.call(context, action="fetch", prompt_id="external-prompt")
    assert (len(client.downloads), len(client.history_queries)) == previous_counts
    queue.jobs["C000002"] = {**job, "id": "C000002", "umo": "session-B", "prompt_id": "pid-B"}
    tool.shared["fetched_outputs"]["pid-B"] = {"images": images, "paths": paths}
    denied = await tool.call(context, action="fetch", prompt_id="pid-B")
    assert "当前会话没有该任务" in denied and not any(path in denied for path in paths)
    denied = await tool.call(context, action="fetch", prompt_id="C000002")
    assert "当前会话没有该任务" in denied
    assert (len(client.downloads), len(client.history_queries)) == previous_counts
    assert client.history_queries == ["external-prompt"]
    denied = await tool.call(context, action="fetch", task_id="C000002")
    assert "当前会话没有该任务" in denied
    assert (len(client.downloads), len(client.history_queries)) == previous_counts
    for key, scope in (("C000003", "session-A"), ("C000004", "session-B")):
        queue.jobs[key] = {"id": key, "umo": scope, "state": "running", "prompt_id": key}
    active = await tool.call(context, action="fetch", task_id="C000003")
    assert "后台仍在处理" in active and len(client.downloads) == previous_counts[0]
    local = await tool.call(context, action="queue", include_ids=True)
    assert local.count("任务：生成中") == 1, local
    assert "C000003" in local and "C000004" not in local and "other-client-prompt" not in local
    assert "编号仅为内部标记，不用向用户声明或复述。" in local
    assert queue.find("C000003", "session-A") is not None
    assert queue.find("C000004", "session-A") is None
    queue.jobs["C000005"] = {**queue.jobs["C000003"], "id": "C000005"}
    assert (await tool.call(context, action="queue")).count("编号仅为内部标记") == 1
    status = await tool.call(context, action="status", task_id="C000001")
    assert "输出图片：2 张" in status and not any(path in status for path in paths)

    stats = ComfyuiSystemStatsTool(client=client)
    global_queue = await stats.call(context)
    assert "other-client-prompt" not in global_queue and "GPU" in global_queue
    assert "运行中 1 个，待执行 1 个" in global_queue
    assert "pending-backend-prompt" not in global_queue
    assert "ComfyUI 版本: test-version" in global_queue
    count = client.queue_queries
    system_only = await stats.call(context, include_queue=False)
    assert "other-client-prompt" not in system_only and client.queue_queries == count
    client.stats = None
    assert "运行中 1 个，待执行 1 个" in await stats.call(context)
    direct = ComfyuiJobTool(client=client)
    result = await direct.call(context, action="status", prompt_id="external-prompt", include_ids=True)
    assert "已完成" in result and "prompt_id=external-prompt" in result
    assert "other-client-prompt" not in await direct.call(context, action="queue", include_ids=True)
    print("output reuse / selective recovery / partial failure retry / external prompt recovery OK")
    print("session task isolation / active download guard / global GPU and queue diagnostics OK")


async def main():
    await resources_and_sources()
    with tempfile.TemporaryDirectory() as directory:
        await outputs_and_scope(Path(directory))
    print("tool consolidation regressions OK")


if __name__ == "__main__":
    asyncio.run(main())
