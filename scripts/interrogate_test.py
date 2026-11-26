# -*- coding: utf-8 -*-
"""提示词反推测试：PNG 元数据解析、AstrBot Provider / DeepSeek 直连 VLM、工具与 Studio API。"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

import smoke_test  # noqa: F401 - installs AstrBot stubs when absent
from astrbot.api.message_components import Image, Reply
from comfyflow.clients.vision import VisionInterrogateClient, VisionRefusalError
from comfyflow.resources.interrogate import (
    extract_image_metadata,
    format_reverse_prompt_result,
    reverse_prompt,
)
from comfyflow.storage.recipes import RecipeStore
from comfyflow.studio import api as webapi_module
from comfyflow.tools import ComfyuiReversePromptTool
from comfyflow.workflows.builder import WorkflowBuilder
from comfyflow.workflows.registry import ModelFamilyRegistry


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _make_png(chunks: list[tuple[bytes, bytes]] | None = None) -> bytes:
    ihdr = struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00" + b"\xff\x00\x00" * 16)
    parts = [b"\x89PNG\r\n\x1a\n", _png_chunk(b"IHDR", ihdr)]
    for ctype, cdata in chunks or []:
        parts.append(_png_chunk(ctype, cdata))
    parts.append(_png_chunk(b"IDAT", idat))
    parts.append(_png_chunk(b"IEND", b""))
    return b"".join(parts)


def _make_comfy_png() -> bytes:
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "Anima/model.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl, silver hair, moonlit rooftop", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres, worst quality", "clip": ["1", 1]}},
        "4": {"class_type": "LoraLoader", "inputs": {"lora_name": "Anima/watercolor.safetensors", "model": ["1", 0], "clip": ["1", 1], "strength_model": 0.7, "strength_clip": 0.7}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "steps": 24,
                "cfg": 5.5,
                "sampler_name": "er_sde",
                "scheduler": "normal",
                "seed": 123456,
                "denoise": 1.0,
            },
        },
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
    }
    raw = json.dumps(graph, ensure_ascii=False).encode("utf-8")
    return _make_png([(b"tEXt", b"prompt\x00" + raw)])


def _make_a1111_png() -> bytes:
    params = (
        "masterpiece, 1girl, blue dress, sunlight\n"
        "Negative prompt: blurry, bad anatomy\n"
        "Steps: 28, Sampler: DPM++ 2M, CFG scale: 7, Seed: 42, Size: 832x1216, Model: illustriousXL"
    ).encode("utf-8")
    compressed = zlib.compress(params)
    # iTXt: key\0 compression_flag(1) compression_method(0) lang\0 translated\0 compressed_text
    return _make_png([(b"iTXt", b"parameters\x00\x01\x00\x00\x00" + compressed)])


def _make_novelai_png() -> bytes:
    comment = json.dumps({
        "prompt": "1boy, trench coat, rainy street",
        "uc": "worst quality, lowres",
        "steps": 28,
        "scale": 5.0,
        "sampler": "k_euler_ancestral",
        "seed": 999,
    }).encode("utf-8")
    return _make_png([
        (b"tEXt", b"Software\x00NovelAI"),
        (b"tEXt", b"Comment\x00" + comment),
    ])


def test_metadata_extraction():
    comfy = extract_image_metadata(_make_comfy_png())
    assert comfy and comfy["format"] == "comfyui"
    assert "silver hair" in comfy["prompt"]
    assert "lowres" in comfy["negative_prompt"]
    assert comfy["params"]["model"] == "Anima/model.safetensors"
    assert comfy["params"]["loras"] == ["Anima/watercolor.safetensors"]
    assert comfy["params"]["steps"] == 24

    a1111 = extract_image_metadata(_make_a1111_png())
    assert a1111 and a1111["format"] == "a1111"
    assert "blue dress" in a1111["prompt"]
    assert a1111["negative_prompt"] == "blurry, bad anatomy"
    assert a1111["params"]["Model"] == "illustriousXL"

    nai = extract_image_metadata(_make_novelai_png())
    assert nai and nai["format"] == "novelai"
    assert "trench coat" in nai["prompt"]
    assert "worst quality" in nai["negative_prompt"]
    assert extract_image_metadata(_make_png()) is None
    print("  metadata extraction OK")


async def test_vision_astrbot_provider_and_fallback():
    calls = []

    class FakeContext:
        def get_config(self):
            return {"provider_settings": {"default_image_caption_provider_id": "global-vlm"}}

        async def llm_generate(self, *, chat_provider_id, prompt, system_prompt=None, image_urls=None, **kwargs):
            calls.append((chat_provider_id, prompt, system_prompt, image_urls, kwargs))
            if chat_provider_id == "global-vlm":
                raise RuntimeError("429 Too Many Requests")
            return SimpleNamespace(completion_text="1girl, solo, silver hair, rooftop")

    vision = VisionInterrogateClient(
        FakeContext(),
        provider_id="",
        fallback_provider_ids=["backup-vlm", "global-vlm"],
        thinking_effort="low",
        retry_delay=0.01,
        max_retries=2,
    )
    assert vision.resolve_provider_candidates() == ["global-vlm", "backup-vlm"]
    text, backend = await vision.describe_image(
        image_path="C:/tmp/test.png",
        image_bytes=_make_png(),
        system_prompt="SYS",
        user_prompt="USR",
    )
    assert text == "1girl, solo, silver hair, rooftop"
    assert backend == "provider:backup-vlm"
    assert [c[0] for c in calls] == ["global-vlm", "global-vlm", "backup-vlm"]
    assert calls[-1][3] == ["C:/tmp/test.png"]
    assert calls[-1][4] == {"reasoning_effort": "low"}
    print("  vision AstrBot provider + fallback OK")


async def test_vision_direct_deepseek_api():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, json={"error": "rate limit"})
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "deepseek-flash"
        assert body["reasoning_effort"] == "low"
        assert body["messages"][0]["role"] == "system"
        user_msg = body["messages"][1]
        assert user_msg["role"] == "user"
        img_part = user_msg["content"][1]
        assert img_part["type"] == "image_url"
        assert img_part["image_url"]["url"].startswith("data:image/png;base64,")
        assert img_part["image_url"]["detail"] == "high"
        return httpx.Response(200, json={"choices": [{"message": {"content": "A silver-haired girl on a moonlit rooftop."}}]})

    vision = VisionInterrogateClient(
        api_base="https://api.deepseek.com/v1",
        api_key="sk-test",
        api_model="deepseek-flash",
        thinking_effort="low",
        image_detail="high",
        retry_delay=0.01,
        max_retries=2,
    )
    vision._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        text, backend = await vision.describe_image(
            image_path="",
            image_bytes=_make_png(),
            system_prompt="SYS",
            user_prompt="USR",
        )
    finally:
        await vision.close()

    assert len(requests) == 2
    assert str(requests[0].url) == "https://api.deepseek.com/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer sk-test"
    assert text == "A silver-haired girl on a moonlit rooftop."
    assert backend == "direct:deepseek-flash"
    print("  vision direct DeepSeek API OK")


async def test_tool_and_studio_api():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output_dir = root / "output"
        output_dir.mkdir()
        comfy_img = output_dir / "comfy.png"
        plain_img = output_dir / "plain.png"
        comfy_img.write_bytes(_make_comfy_png())
        plain_img.write_bytes(_make_png())

        outside_img = root / "outside.png"
        outside_img.write_bytes(_make_png())

        calls = []

        class FakeContext:
            def get_config(self):
                return {}

            async def llm_generate(self, *, chat_provider_id, prompt, system_prompt=None, image_urls=None, **kwargs):
                calls.append((chat_provider_id, prompt, system_prompt, image_urls))
                if "Danbooru" in (system_prompt or ""):
                    return SimpleNamespace(completion_text="1girl, red_scarf, night_city")
                return SimpleNamespace(completion_text="A girl wearing a red scarf in a neon city.")

        families = ModelFamilyRegistry([
            {"name": "anima", "workflow": "anima-v3", "prompt_style": "danbooru"},
            {"name": "flux", "workflow": "flux-dev", "prompt_style": "natural"},
        ])
        vision = VisionInterrogateClient(FakeContext(), provider_id="vision-main")
        shared = {"last_image_paths": {"chat:1": str(plain_img)}}
        tool = ComfyuiReversePromptTool(
            vision=vision,
            output_dir=output_dir,
            shared=shared,
            families=families,
        )
        tool.refresh_schema()
        assert tool.parameters["properties"]["model_family"]["enum"] == ["anima", "flux"]

        # 1) Metadata fast path on message attachment (never calls VLM)
        ctx_meta = SimpleNamespace(
            context=SimpleNamespace(
                event=SimpleNamespace(
                    unified_msg_origin="chat:1",
                    message_obj=SimpleNamespace(message=[Reply(chain=[Image(comfy_img)])]),
                )
            )
        )
        res_meta = await tool.call(ctx_meta)
        assert "图片内嵌元数据 (comfyui)" in res_meta
        assert "moonlit rooftop" in res_meta
        assert calls == []

        # 2) read_metadata=False + model_family="anima" -> routes to danbooru VLM prompt
        res_forced = await tool.call(ctx_meta, read_metadata=False, model_family="anima", focus="只看服饰")
        assert "视觉模型 (danbooru | provider:vision-main)" in res_forced
        assert "1girl, red_scarf, night_city" in res_forced
        assert "只看服饰" in calls[-1][1]
        assert "comfyui_lookup" in res_forced

        # 3) Fallback to last_image_paths + natural style
        ctx_plain = SimpleNamespace(
            context=SimpleNamespace(
                event=SimpleNamespace(
                    unified_msg_origin="chat:1",
                    message_obj=SimpleNamespace(message=[]),
                )
            )
        )
        res_last = await tool.call(ctx_plain, style="natural")
        assert "视觉模型 (natural | provider:vision-main)" in res_last
        assert "red scarf" in res_last

        # 4) Reject outside path
        res_outside = await tool.call(ctx_plain, image_path=str(outside_img))
        assert "反推失败" in res_outside

        # 5) Unconfigured VLM on plain image gives helpful error
        tool_no_vlm = ComfyuiReversePromptTool(
            vision=VisionInterrogateClient(SimpleNamespace(get_config=lambda: {})),
            output_dir=output_dir,
            shared=shared,
            families=families,
        )
        res_err = await tool_no_vlm.call(ctx_plain)
        assert "尚未配置反推视觉模型" in res_err

        # 6) StudioApi.interrogate
        body = {}

        async def fake_body():
            return dict(body)

        builder = WorkflowBuilder(plugin_dir=Path(__file__).resolve().parent.parent, custom_dir=root / "workflows")
        store = RecipeStore(root)
        api = webapi_module.StudioApi(
            None, builder, store, output_dir, shared,
            families=families, vision_client=vision,
        )
        with patch.object(webapi_module, "_body", fake_body), patch.object(
            webapi_module, "_json", lambda payload, status=200: (status, payload)
        ):
            body.update({"image_data": "data:image/png;base64," + base64.b64encode(_make_a1111_png()).decode("ascii")})
            status, payload = await api.interrogate()
            assert status == 200 and payload["ok"] is True
            assert payload["source"] == "metadata" and payload["format"] == "a1111"

            body.clear()
            body.update({
                "image_data": base64.b64encode(_make_png()).decode("ascii"),
                "model_family": "anima",
            })
            status, payload = await api.interrogate()
            assert status == 200 and payload["ok"] is True
            assert payload["source"] == "vision" and payload["style"] == "danbooru"
            assert calls[-1][3][0].startswith("data:image/png;base64,")

            body.clear()
            body.update({"image_data": "not-valid-base64!!!"})
            status, payload = await api.interrogate()
            assert status == 400 and payload["ok"] is False

            body.clear()
            body.update({"image_path": str(outside_img)})
            status, payload = await api.interrogate()
            assert status == 400 and payload["ok"] is False

        # Verify formatter output stays readable
        assert "生成参数" in format_reverse_prompt_result(extract_image_metadata(_make_comfy_png()))
        _ = await reverse_prompt(
            image_path=plain_img,
            image_bytes=_make_png(),
            style="auto",
            vision=vision,
        )
    print("  tool + Studio API OK")


async def test_refusal_handling_and_custom_prompt():
    # 1) Direct API content_filter falls back immediately without retrying same direct endpoint
    direct_calls = 0

    def filter_handler(request: httpx.Request) -> httpx.Response:
        nonlocal direct_calls
        direct_calls += 1
        return httpx.Response(200, json={"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]})

    provider_calls = []

    class FakeContext:
        def get_config(self):
            return {}

        async def llm_generate(self, *, chat_provider_id, prompt, system_prompt=None, image_urls=None, **kwargs):
            provider_calls.append((chat_provider_id, system_prompt))
            if chat_provider_id == "strict-vlm":
                return SimpleNamespace(completion_text="I'm sorry, I can't assist with analyzing this image.")
            return SimpleNamespace(completion_text="1girl, swimsuit, beach, sunlight")

    vision = VisionInterrogateClient(
        FakeContext(),
        provider_id="strict-vlm",
        fallback_provider_ids=["permissive-vlm"],
        system_prompt_override="CUSTOM SYSTEM TEMPLATE",
        api_base="https://api.deepseek.com/v1",
        api_key="sk-test",
        api_model="deepseek-flash",
        retry_delay=0.01,
        max_retries=3,
    )
    vision._http = httpx.AsyncClient(transport=httpx.MockTransport(filter_handler))
    try:
        res = await reverse_prompt(
            image_path="",
            image_bytes=_make_png(),
            style="danbooru",
            read_metadata=False,
            vision=vision,
        )
    finally:
        await vision.close()

    assert direct_calls == 1
    assert provider_calls == [
        ("strict-vlm", "CUSTOM SYSTEM TEMPLATE"),
        ("permissive-vlm", "CUSTOM SYSTEM TEMPLATE"),
    ]
    assert res["backend"] == "provider:permissive-vlm"
    assert res["prompt"] == "1girl, swimsuit, beach, sunlight"

    # 2) All providers refusing raises VisionRefusalError instead of returning refusal text
    class AllRefuseContext:
        def get_config(self):
            return {}

        async def llm_generate(self, **kwargs):
            return SimpleNamespace(completion_text="抱歉，我无法协助处理该图片。")

    refusing_vision = VisionInterrogateClient(AllRefuseContext(), provider_id="strict-only", max_retries=3, retry_delay=0.01)
    try:
        await refusing_vision.describe_image(image_path="x.png", image_bytes=_make_png(), user_prompt="tag")
        raise AssertionError("expected VisionRefusalError")
    except VisionRefusalError:
        pass
    print("  refusal detection + fallback + custom prompt OK")


async def main():
    print("[interrogate] 提示词反推测试")
    test_metadata_extraction()
    await test_vision_astrbot_provider_and_fallback()
    await test_vision_direct_deepseek_api()
    await test_tool_and_studio_api()
    await test_refusal_handling_and_custom_prompt()
    print("ALL OK")


if __name__ == "__main__":
    asyncio.run(main())
