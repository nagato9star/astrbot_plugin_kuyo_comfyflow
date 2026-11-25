"""验证 civitai_search 三模式、lookup(model_info) 与 job(action=history) 的行为。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import smoke_test  # noqa: F401 - install AstrBot stubs for fast offline test
from comfyflow.clients.comfyui import ComfyUIClient
from comfyflow.clients.external import CivitaiClient
from comfyflow.tools.jobs import ComfyuiJobTool
from comfyflow.tools.reference import (
    ComfyuiCivitaiSearchTool,
    ComfyuiLookupTool,
    _civitai_image_recipe_lines,
)

PASS, FAIL = 0, 0


def check(name: str, cond: bool, info: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[PASS] {name} {info}")
    else:
        FAIL += 1
        print(f"[FAIL] {name} {info}")


class StubCivitai(CivitaiClient):
    def __init__(self):
        self.calls = []

    async def search_images(self, query="", limit=5, nsfw=False, sort="Most Reactions",
                            period="AllTime", model_id=None, model_version_id=None):
        self.calls.append(("images", dict(query=query, limit=limit, nsfw=nsfw, sort=sort,
                                          period=period, model_id=model_id,
                                          model_version_id=model_version_id)))
        return [{
            "meta": {"Model": "flux-d", "prompt": "1girl", "negativePrompt": "bad",
                     "sampler": "euler", "steps": 28, "cfgScale": 4.5, "seed": 42},
            "url": "https://img.example/1.png",
            "modelId": 5, "modelVersionId": 77,
        }]

    async def search_models(self, query="", types="", limit=5, sort="Highest Rated",
                            period="AllTime", base_models=None, nsfw=None):
        self.calls.append(("models", dict(query=query, types=types, limit=limit, sort=sort,
                                          period=period, base_models=base_models, nsfw=nsfw)))
        return [{
            "id": 11, "name": "Flux Style", "type": "LORA",
            "creator": {"username": "someone"},
            "stats": {"downloadCount": 100, "thumbsUpCount": 9},
            "modelVersions": [
                {"id": 77, "name": "v1", "baseModel": "Flux.1 D",
                 "trainedWords": ["fluxstyle", "soft light"]},
                {"id": 78, "name": "v2", "baseModel": "Flux.1 D", "trainedWords": []},
            ],
            "tags": ["style", "anime"],
        }]

    async def get_model_version(self, version_id):
        self.calls.append(("version", version_id))
        return {
            "id": 77, "modelId": 11, "name": "v1", "baseModel": "Flux.1 D",
            "trainedWords": ["fluxstyle"],
            "description": "A flux style lora",
            "files": [{"name": "flux_style.safetensors", "sizeKB": 228000}],
            "model": {"name": "Flux Style", "type": "LORA"},
            "images": [{"meta": {"prompt": "demo"}, "url": "https://img.example/2.png"}],
        }


class StubClient(ComfyUIClient):
    resource_family_rules = []

    def __init__(self):
        pass

    async def list_resources(self):
        return ({"lora_name": ["style.safetensors"],
                 "unet_name": ["flux-base.safetensors"],
                 "lora_meta": {}, "model_meta": {}}, False)

    async def get_lora_manager_metadata(self, name):
        return None

    async def get_model_metadata(self, name, folders=None):
        return {"modelspec.title": "Style LoRA", "modelspec.author": "tester"}

    async def list_history(self, max_items=12):
        return [{"prompt_id": "pid-aaa", "number": 42, "status": "success",
                 "has_workflow": True, "images": ["out1.png", "out2.png"]},
                {"prompt_id": "pid-bbb", "number": 41, "status": "success",
                 "has_workflow": False, "images": []}]


async def main() -> None:
    # civitai_search images 模式：过滤参数透传 + 版本 id 输出
    civ = StubCivitai()
    tool = ComfyuiCivitaiSearchTool(client=civ)
    out = await tool.call(None, query="landscape", sort="comments", period="week",
                          model_version_id=77)
    kind, params = civ.calls[-1]
    check("images sort/period 归一化",
          kind == "images" and params["sort"] == "Most Comments" and params["period"] == "Week")
    check("images version 过滤透传", params["model_version_id"] == 77)
    check("images 配方行", "sampler/steps/cfg/seed: euler / 28 / 4.5 / 42" in out)
    check("images 输出版本 id", "model_version_id=77" in out)

    # models 模式：家族映射为 baseModels 过滤，输出可下钻 id
    out = await tool.call(None, mode="models", query="flux style", model_family="flux")
    kind, params = civ.calls[-1]
    check("models baseModels 家族映射",
          kind == "models" and "Flux.1 D" in (params["base_models"] or []))
    check("models 输出版本 id", "model_version_id=77" in out)
    check("models 输出触发词", "fluxstyle" in out)
    check("models 输出 model_id", "model_id=11" in out)

    # base_model 精确值优先于家族映射
    await tool.call(None, mode="models", query="x", model_family="flux", base_model="Pony")
    check("models base_model 精确覆盖", civ.calls[-1][1]["base_models"] == ["Pony"])

    # version 模式：详情 + 示例图配方
    out = await tool.call(None, mode="version", model_version_id=77)
    check("version 调用", civ.calls[-1] == ("version", 77))
    check("version baseModel/触发词", "Flux.1 D" in out and "fluxstyle" in out)
    check("version 示例图配方", "prompt: demo" in out)

    # 校验错误路径
    check("缺 query 报错", "不能为空" in await tool.call(None, mode="images", query=""))
    check("version 缺 id 报错", "model_version_id" in await tool.call(None, mode="version"))
    check("坏 mode 报错", "仅支持" in await tool.call(None, mode="bogus", query="x"))

    # 回归：CivitaiClient.search_images 必须带 withMeta=true
    seen_params = []
    def mock_civitai(request):
        seen_params.append(dict(request.url.params))
        return httpx.Response(200, json={"items": []})
    real_civ = CivitaiClient()
    real_civ._client = httpx.AsyncClient(base_url="https://civitai.red", transport=httpx.MockTransport(mock_civitai))
    await real_civ.search_images(model_version_id=14479)
    await real_civ.close()
    check("search_images 发送 withMeta=true", seen_params and seen_params[0].get("withMeta") == "true")

    # 回归：无 model_version_id 的关键词 images 搜索先匹配模型再拉配方图，并优先保留有 meta 的图
    class KeywordCivitai(CivitaiClient):
        async def search_models(self, query="", types="", limit=5, sort="Highest Rated",
                                period="AllTime", base_models=None, nsfw=None):
            return [{"id": 7505, "name": "Hu Tao | Genshin Impact",
                     "modelVersions": [{"id": 14479, "baseModel": "SD 1.5"}]}]

        async def search_images(self, query="", limit=5, nsfw=False, sort="Most Reactions",
                                period="AllTime", model_id=None, model_version_id=None):
            if model_version_id == 14479:
                return [
                    {"id": 1, "meta": None, "baseModel": "SD 1.5", "modelVersionIds": [14479], "url": "https://img/no-meta.png"},
                    {"id": 2, "meta": {"prompt": "1girl, hu tao", "sampler": "DPM++ SDE", "steps": 30, "cfgScale": 7, "seed": 0,
                                        "Model": "urn:air:sd15:checkpoint:civitai:1@2",
                                        "civitaiResources": [{"type": "checkpoint", "modelName": "AbyssOrangeMix2"},
                                                             {"type": "lora", "modelName": "HuTaoLoRA", "weight": 0.8}]},
                     "baseModel": "SD 1.5", "modelVersionIds": [14479], "url": "https://img/with-meta.png"},
                ]
            return []

    kw_tool = ComfyuiCivitaiSearchTool(client=KeywordCivitai())
    kw_out = await kw_tool.call(None, mode="images", query="hu tao", limit=1)
    check("关键词 images 优先选带配方图", "1girl, hu tao" in kw_out and "no-meta.png" not in kw_out)
    check("模型名解析 civitaiResources 与 baseModel", "AbyssOrangeMix2（SD 1.5）" in kw_out)
    check("LoRA 资源摘要", "HuTaoLoRA(0.8)" in kw_out)
    check("seed=0 不被当成问号", "DPM++ SDE / 30 / 7 / 0" in kw_out)
    check("modelVersionIds 列表提取", "model_version_id=14479" in kw_out)

    empty_meta_lines = "\n".join(_civitai_image_recipe_lines({"baseModel": "Flux.1 D", "modelVersionIds": [99, 100]}, 1))
    check("无 meta 图不输出全问号", "原图未附带生成参数" in empty_meta_lines and "Flux.1 D" in empty_meta_lines)

    # lookup(model_info)：唯一匹配 → 家族 + 本地元数据
    client = StubClient()
    lookup = ComfyuiLookupTool(client=client)
    out = await lookup.call(None, type="model_info", query="style.safetensors")
    check("model_info 唯一匹配", "【本地模型】style.safetensors" in out and "标题: Style LoRA" in out)
    check("model_info 家族标注", "家族=" in out)
    out = await lookup.call(None, type="model_info", query="flux-base", types="Checkpoint")
    check("model_info types 限定底模", "【本地模型】flux-base.safetensors" in out)
    out = await lookup.call(None, type="model_info", query="missing-file")
    check("model_info 无匹配提示", "未找到唯一匹配" in out)

    # job history：后端最近记录含图片数与 prompt_id
    job = ComfyuiJobTool(client=client)
    out = await job._call(None, action="history")
    check("history 列表", "#42 success" in out and "pid-aaa" in out)
    check("history 输出计数", "输出 2 张" in out)
    check("history fetch 提示", "action=fetch" in out)

    print()
    if FAIL:
        print(f"{FAIL} 项失败")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    asyncio.run(main())
