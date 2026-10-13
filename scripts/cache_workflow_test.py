"""Cache retention and output-directed workflow recognition regressions."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
import time
from pathlib import Path

import smoke_test  # noqa: F401 - offline AstrBot fixtures
from comfyflow.clients.comfyui import ComfyUIClient
from comfyflow.storage.images import manage_cache, save_image
from comfyflow.workflows.registry import WorkflowProfileStore
from comfyflow.workflows.slots import apply_slots, detect_slots
from comfyflow.tools import ComfyuiLookupTool

PLUGIN = Path(__file__).resolve().parents[1]


def image_retention(root: Path) -> None:
    pinned = save_image(root, "queued.png", b"queued image")
    stale = save_image(root, "old.png", b"unreferenced image")
    old = time.time() - 10 * 86400
    for path in (pinned, stale):
        os.utime(path, (old, old))
    result = manage_cache(
        root, action="expired", days=7, max_mb=0, protected_paths=[pinned]
    )
    assert result["removed"] == 1 and result["protected"] == 1
    assert pinned.read_bytes() == b"queued image" and not stale.exists()
    result = manage_cache(root, action="clear", protected_paths=[pinned])
    assert result["removed"] == 0 and pinned.is_file()
    pinned.write_bytes(b"damaged cache file")
    assert save_image(root, "queued.png", b"queued image") == pinned
    assert pinned.read_bytes() == b"queued image"
    assert not list(root.glob("*.tmp"))


async def resource_retention(root: Path) -> None:
    class CatalogClient(ComfyUIClient):
        async def get_object_info(self, force_refresh=False):
            return self.definitions

        async def get_lora_manager_catalog(self):
            if getattr(self, "catalog_gate", None) is not None:
                self.catalog_started.set()
                await self.catalog_gate.wait()
            return {"style.safetensors": {"base_model": "Anima"}}

        async def _fetch_lora_trigger_words(self, names, **kwargs):
            return {"style.safetensors": {"trigger_words": ["known", "selected_now"],
                                         "trusted_trigger_words": ["known", "selected_now"]}}

        async def get_embeddings(self):
            return None

    client = CatalogClient(
        "test", 8188, cache_file=root / "resources.json", cache_ttl=0
    )
    client._closed = True  # Drive enrichment explicitly, without background timing.
    original = {
        "fetched_at": 1,
        "lora_metadata_v3": True,
        "lora_metadata_v4": True,
        "lora_catalog_ready": True,
        "resources": {
            "unet_name": ["base.safetensors"],
            "lora_name": ["style.safetensors"],
            "clip_name": ["encoder.safetensors"],
            "vae_name": ["vae.safetensors"],
            "embeddings": ["known_embedding"],
            "lora_meta": {"style.safetensors": {"trigger_words": ["known"], "trusted_trigger_words": ["known"]}},
            "model_meta": {"base.safetensors": {"family": "demo"}},
        },
    }
    client._save_cache(copy.deepcopy(original))
    client.definitions = {"CLIPLoader": {"input": {"required": {"clip_name": [[]]}}}}
    try:
        refreshed = await client.generation_resources(force_refresh=True)
        assert refreshed["clip_name"] == [], "a present empty enum is authoritative"
        for field in (
            "unet_name",
            "lora_name",
            "vae_name",
            "lora_meta",
            "model_meta",
            "embeddings",
        ):
            assert refreshed[field] == original["resources"][field]
        assert client._load_cache()["lora_metadata_v4"] is True
        # Soft catalog expiry keeps full header metadata, while an explicit
        # LoRA refresh still schedules a real metadata refresh for unchanged names.
        client.definitions = {
            "LoraLoader": {
                "input": {"required": {"lora_name": [["style.safetensors"]]}}
            }
        }
        await client.generation_resources()
        assert client._load_cache()["lora_metadata_v4"] is True
        # 到期继续更新轻量目录与 Embedding，不重新下载已有完整元数据。
        await client._enrich_catalog()
        assert client._load_cache()["resources"]["lora_meta"]["style.safetensors"]["trigger_words"] == ["known"]
        assert client._load_cache()["resources"]["lora_meta"]["style.safetensors"]["trusted_trigger_words"] == ["known"]
        before_selection = copy.deepcopy(client._load_cache())
        client.catalog_started, client.catalog_gate = asyncio.Event(), asyncio.Event()
        light = asyncio.create_task(client._enrich_catalog())
        await client.catalog_started.wait()
        await client.selected_lora_metadata({"lora_meta": {}}, ["style.safetensors"], require_trusted=True)
        client.catalog_gate.set()
        await light
        assert client._load_cache()["resources"]["lora_meta"]["style.safetensors"]["trusted_trigger_words"] == ["known", "selected_now"]
        client.catalog_gate = None
        client._save_cache(before_selection)
        await client.generation_resources(force_refresh=True)
        assert client._load_cache()["lora_metadata_v4"] is False
        assert client._load_cache()["resources"]["lora_meta"]["style.safetensors"]["trigger_words"] == ["known"]
        refreshed["lora_name"].clear()
        assert client._load_cache()["resources"]["lora_name"] == ["style.safetensors"]
        await client._enrich_catalog()
        assert client._load_cache()["resources"]["embeddings"] == ["known_embedding"]
        assert client._load_cache()["resources"]["lora_meta"]["style.safetensors"]["trusted_trigger_words"] == ["known"]
        client.cache_file.write_text("{interrupted external write", encoding="utf-8")
        assert client._load_cache()["resources"]["unet_name"] == ["base.safetensors"]
        client.definitions = {
            "LoraLoader": {"input": {"required": {"lora_name": [[]]}}}
        }
        refreshed = await client.generation_resources(force_refresh=True)
        assert refreshed["lora_name"] == [] and refreshed["lora_meta"] == {}
        assert not client._load_cache()["lora_metadata_v4"]
        assert (
            json.loads(client.cache_file.read_text(encoding="utf-8"))["resources"]
            == refreshed
        )
    finally:
        await client.close()


async def fast_model_lookup() -> None:
    class FastClient(ComfyUIClient):
        async def generation_resources(self, force_refresh=False):
            return {"unet_name": ["base.safetensors"], "lora_name": [], "lora_meta": {}}

        async def list_resources(self, force_refresh=False):
            raise AssertionError("model lookup must not wait for full LoRA metadata")

    client = FastClient("test", 8188)
    try:
        result = await ComfyuiLookupTool(client=client).call(
            None, type="model", query="base.safetensors"
        )
        assert "base.safetensors" in result
    finally:
        await client.close()


def graph_recognition(root: Path) -> None:
    base = json.loads(
        (PLUGIN / "scripts/fixtures/mini_workflow.json").read_text(encoding="utf-8")
    )
    idle = {
        "idle:model": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "unused.safetensors"},
        },
        "idle:prompt": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "much longer disconnected prompt"},
        },
        "idle:sampler": {"class_type": "KSampler", "inputs": {"steps": 999}},
        "idle:qwen": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {"prompt": "preview"},
        },
        "idle:preview": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["idle:qwen", 0]},
        },
    }
    workflow = {**idle, **copy.deepcopy(base)}
    slots = detect_slots(workflow)
    assert slots["prompt"]["node"] == "2"
    assert slots["negative"]["node"] == "3"
    assert slots["model"]["node"] == "1"
    assert slots["sampler"]["node"] == "5" and "sampler_2" not in slots
    assert slots["loras"]["node"] == "7"
    before = copy.deepcopy(workflow)
    apply_slots(workflow, slots, {"prompt": "new prompt", "seed": 42})
    assert workflow["6"] == before["6"], "output inputs must remain stable"
    assert {nid: workflow[nid] for nid in idle} == idle
    assert workflow["1"] == before["1"] and workflow["7"] == before["7"]
    apply_slots(workflow, slots, {"prompt": "new prompt", "seed": 42})
    repeated = copy.deepcopy(workflow)
    apply_slots(workflow, slots, {"prompt": "new prompt", "seed": 42})
    assert workflow == repeated, (
        "identical overrides must produce identical graph inputs"
    )

    linked = copy.deepcopy(base)
    linked["2"]["inputs"]["text"] = ["prompt:source", 0]
    linked["prompt:source"] = {
        "class_type": "PrimitiveString",
        "inputs": {"value": "old"},
    }
    apply_slots(linked, detect_slots(linked), {"prompt": "changed"})
    assert linked["2"]["inputs"]["text"] == ["prompt:source", 0]
    assert linked["prompt:source"]["inputs"]["value"] == "changed"
    linked["prompt:source"]["inputs"]["value"] = ["2", 0]
    try:
        apply_slots(linked, detect_slots(linked), {"prompt": "cycle"})
    except ValueError as error:
        assert "连线无效" in str(error)
    else:
        raise AssertionError("cyclic text links must fail without severing the link")

    custom = {**copy.deepcopy(idle), **copy.deepcopy(base)}
    custom["6"]["class_type"] = "CustomFinalOutput"
    assert (
        detect_slots(custom, {"CustomFinalOutput": {"output_node": True}})["prompt"][
            "node"
        ]
        == "2"
    )
    accelerator = copy.deepcopy(base)
    accelerator["7"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": "accelerator.safetensors",
            "strength_model": 0.8,
        },
    }
    assert "loras" not in detect_slots(accelerator)
    apply_slots(accelerator, detect_slots(accelerator), {"prompt": "cat"})
    assert accelerator["7"]["inputs"]["strength_model"] == 0.8

    dual = copy.deepcopy(base)
    dual["later:1"] = copy.deepcopy(dual["5"])
    dual["later:1"]["inputs"]["steps"] = ["step:value", 0]
    dual["step:value"] = {"class_type": "PrimitiveInt", "inputs": {"value": 20}}
    dual["later:1"]["inputs"]["latent_image"] = ["5", 0]
    dual["9"]["inputs"]["samples"] = ["later:1", 0]
    dual["2"]["inputs"]["text"] = (
        "lowres, worst quality, deliberately requested positive effect"
    )
    dual_slots = detect_slots(dual)
    assert dual_slots["prompt"]["node"] == "2"
    assert dual_slots["sampler"]["node"] == "5"
    assert dual_slots["sampler_2"]["node"] == "later:1"

    qwen = {
        "10": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
        "20": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "prompt": "old",
                "negative_prompt": "bad",
                "images.image_1": ["10", 0],
            },
        },
        "30": {"class_type": "SaveImage", "inputs": {"images": ["20", 0]}},
    }
    qwen_slots = detect_slots(qwen)
    assert qwen_slots["negative"]["field"] == "negative_prompt"
    apply_slots(qwen, qwen_slots, {"prompt": "new", "negative": "blur"})
    assert qwen["20"]["inputs"]["prompt"] == "new"
    assert qwen["20"]["inputs"]["negative_prompt"] == "bad, blur"

    other = {}
    for nid, node in base.items():
        node = copy.deepcopy(node)
        for field, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in base:
                node["inputs"][field] = ["branch:" + str(value[0]), value[1]]
        other["branch:" + nid] = node
    ambiguous = {**base, **other}
    assert detect_slots(ambiguous) == {}
    profiles = WorkflowProfileStore(root)
    profiles.save("manual", {"prompt": {"node": "2", "field": "text"}})
    assert profiles.effective("manual", ambiguous)["slots"]["prompt"]["node"] == "2"


async def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        image_retention(root / "images")
        await resource_retention(root / "resources")
        await fast_model_lookup()
        graph_recognition(root / "profiles")
    print(
        "image/cache integrity / partial catalogs / offline retention / output graph / ambiguity / stable links and inputs OK"
    )


if __name__ == "__main__":
    asyncio.run(main())
