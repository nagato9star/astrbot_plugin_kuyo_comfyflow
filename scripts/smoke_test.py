# -*- coding: utf-8 -*-
"""冒烟测试：部署/修改后快速验证插件核心逻辑（无 AstrBot 环境也可跑）。

用法：python scripts/smoke_test.py
覆盖：工作流角色覆盖 / 默认值优先级 / 模板管理 / 提交错误解析 / 缓存原子写。
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import httpx

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGIN_DIR))

if "astrbot" not in sys.modules:
    import types
    from typing import Generic, TypeVar

    _astrbot = types.ModuleType("astrbot")
    _api = types.ModuleType("astrbot.api")

    class _Logger:
        def warning(self, *a, **k):
            pass

        def info(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

    _api.logger = _Logger()
    _T = TypeVar("_T")

    class _FunctionTool(Generic[_T]):
        active = True

    class _MessageChain:
        def file_image(self, path):
            return self

    class _ContextWrapper(Generic[_T]):
        pass

    _api.FunctionTool = _FunctionTool
    _components = types.ModuleType("astrbot.api.message_components")

    class _Image:
        def __init__(self, path):
            self.path = path

        async def convert_to_file_path(self):
            return str(self.path)

    class _Reply:
        def __init__(self, chain=None):
            self.chain = chain or []

    _components.Image = _Image
    _components.Reply = _Reply
    _event = types.ModuleType("astrbot.api.event")
    _event.AstrMessageEvent = object
    _event.MessageChain = _MessageChain
    _core = types.ModuleType("astrbot.core")
    _agent = types.ModuleType("astrbot.core.agent")
    _run_context = types.ModuleType("astrbot.core.agent.run_context")
    _run_context.ContextWrapper = _ContextWrapper
    _astr_context = types.ModuleType("astrbot.core.astr_agent_context")
    _astr_context.AstrAgentContext = object
    sys.modules["astrbot"] = _astrbot
    sys.modules["astrbot.api"] = _api
    sys.modules["astrbot.api.message_components"] = _components
    sys.modules["astrbot.api.event"] = _event
    sys.modules["astrbot.core"] = _core
    sys.modules["astrbot.core.agent"] = _agent
    sys.modules["astrbot.core.agent.run_context"] = _run_context
    sys.modules["astrbot.core.astr_agent_context"] = _astr_context

from comfyflow.clients.comfyui import (  # noqa: E402
    ComfyUIClient,
    execution_error_message,
    image_media_type,
)
from comfyflow.workflows.ui_conversion import api_to_ui, build_extra_pnginfo  # noqa: E402
from comfyflow.workflows.registry import ModelFamilyRegistry, WorkflowProfileStore  # noqa: E402
from comfyflow.storage.recipes import RecipeStore  # noqa: E402
from comfyflow.workflows.slots import (  # noqa: E402
    apply_slots,
    collect_trigger_words,
    detect_slots,
    normalize_workflow,
    node_options_for_slot,
    parse_node_option,
    read_current_values,
    resolve_size,
    slots_from_config,
    ui_to_api,
)
from comfyflow.workflows.builder import WorkflowBuilder  # noqa: E402
from comfyflow.workflows.graph import validate_api_workflow, validate_ui_snapshot  # noqa: E402
from comfyflow.tools import (  # noqa: E402
    ComfyuiDrawTool,
    ComfyuiEditTool,
    ComfyuiRecipeDrawTool,
    ComfyuiRecipeTool,
)
from comfyflow.studio.config import refresh_config_options  # noqa: E402
from astrbot.api.message_components import Image, Reply  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

PLUGIN = _PLUGIN_DIR
# 模板在插件数据目录（插件包不含模板）
_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "plugin_data" / "astrbot_plugin_comfyui_direct"
_CUSTOM_DIR = (_DATA_DIR / "workflows") if (_DATA_DIR / "workflows").is_dir() else None


def _builder(default_workflow: str = "anima-v3") -> WorkflowBuilder:
    return WorkflowBuilder(plugin_dir=PLUGIN, default_workflow=default_workflow, custom_dir=_CUSTOM_DIR)


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_slot_mapping_generic() -> None:
    wf = _load_fixture("mini_workflow.json")
    slots = detect_slots(wf)
    assert slots["prompt"]["node"] == "2"
    assert slots["negative"]["node"] == "3"
    assert slots["model"]["node"] == "1"
    assert slots["sampler"]["node"] == "5"
    assert slots["size"]["node"] == "4"
    assert slots["loras"]["node"] == "7"
    values = read_current_values(wf, slots)
    assert values["model"] == "base.safetensors"
    assert values["width"] == 832 and values["height"] == 1216
    assert values["steps"] == 20
    assert values["loras"][0]["name"] == "style.safetensors"
    apply_slots(
        wf,
        slots,
        {
            "prompt": "cat",
            "model": "other.safetensors",
            "width": 1024,
            "height": 1024,
            "steps": 12,
            "loras": [{"name": "new.safetensors", "strength": 0.4}],
            "seed": 9,
        },
    )
    assert wf["2"]["inputs"]["text"] == "cat"
    assert wf["1"]["inputs"]["unet_name"] == "other.safetensors"
    assert wf["4"]["inputs"]["width"] == 1024
    assert wf["5"]["inputs"]["steps"] == 12
    assert wf["5"]["inputs"]["seed"] == 9
    assert wf["7"]["inputs"]["lora_1"]["lora"] == "new.safetensors"
    print("  slot mapping generic OK")


def test_power_lora_dynamic_slots() -> None:
    """保存的 LoRA 必须能注入原本没有 lora_N 的 Power Loader。"""
    wf = {
        "30": {
            "class_type": "Power Lora Loader (rgthree)",
            "inputs": {"model": ["19", 0], "clip": ["17", 0]},
        }
    }
    slots = {"loras": {"node": "30", "field": "lora"}}
    apply_slots(
        wf,
        slots,
        {
            "loras": [
                {"name": "Krea-2\\krea2-masterpieces-v51.safetensors", "strength": 0.8},
                {"name": "Krea-2\\second.safetensors", "strength": 0.65},
            ]
        },
    )
    inputs = wf["30"]["inputs"]
    assert inputs["lora_1"] == {
        "on": True,
        "lora": "Krea-2\\krea2-masterpieces-v51.safetensors",
        "strength": 0.8,
    }
    assert inputs["lora_2"]["on"] is True
    assert inputs["lora_2"]["strength"] == 0.65
    apply_slots(wf, slots, {"loras": []})
    assert inputs["lora_1"]["on"] is False
    assert inputs["lora_2"]["on"] is False
    print("  power lora dynamic slots OK")


def test_power_lora_reuses_matching_slot() -> None:
    """Krea2 预留在 lora_2 的文件名应在原槽位启用。"""
    wf = {
        "30": {
            "class_type": "Power Lora Loader (rgthree)",
            "inputs": {
                "lora_1": {
                    "on": False,
                    "lora": "Krea-2\\krea2_vrchat photography style.safetensors",
                    "strength": 1,
                },
                "lora_2": {
                    "on": False,
                    "lora": "Krea-2\\krea2-masterpieces-v51.safetensors",
                    "strength": 1,
                },
            },
        }
    }
    apply_slots(
        wf,
        {"loras": {"node": "30", "field": "lora"}},
        {
            "loras": [
                {"name": "Krea-2\\krea2-masterpieces-v51.safetensors", "strength": 0.8}
            ]
        },
    )
    assert wf["30"]["inputs"]["lora_1"]["on"] is False
    assert wf["30"]["inputs"]["lora_2"] == {
        "on": True,
        "lora": "Krea-2\\krea2-masterpieces-v51.safetensors",
        "strength": 0.8,
    }
    print("  power lora matching slot OK")


def test_power_lora_preserves_independent_accelerator() -> None:
    """清空 Power 占位槽时必须保留 Anima 双采样的独立加速分支。"""
    wf = {
        "458": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "Anima\\anima_baseV10.safetensors"},
        },
        "428": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "Anima\\加速_Turbo_v2.9.safetensors",
                "strength_model": 0.8,
                "model": ["458", 0],
            },
            "_meta": {"title": "加速"},
        },
        "478": {
            "class_type": "Power Lora Loader (rgthree)",
            "inputs": {
                "model": ["458", 0],
                "lora_1": {
                    "on": False,
                    "lora": "Anima\\风格_占位.safetensors",
                    "strength": 1.0,
                },
            },
            "_meta": {"title": "权重Lora加载器"},
        },
        "509": {
            "class_type": "XB_ROCmKSamplerAdvanced",
            "inputs": {"model": ["478", 0], "end_at_step": 12},
            "_meta": {"title": "一采"},
        },
        "510": {
            "class_type": "XB_ROCmKSamplerAdvanced",
            "inputs": {
                "model": ["428", 0],
                "start_at_step": 12,
                "latent": ["509", 0],
            },
            "_meta": {"title": "二采"},
        },
    }
    slots = {"loras": {"node": "478", "field": "lora"}}
    accelerator = json.loads(json.dumps(wf["428"], ensure_ascii=False))

    apply_slots(
        wf,
        slots,
        {"loras": [{"name": "Anima\\人物_测试.safetensors", "strength": 0.7}]},
    )
    assert wf["478"]["inputs"]["lora_1"]["on"] is True
    assert wf["428"] == accelerator

    apply_slots(wf, slots, {"loras": []})
    assert wf["478"]["inputs"]["lora_1"]["on"] is False
    assert wf["428"] == accelerator
    assert wf["510"]["inputs"]["model"] == ["428", 0]
    print("  power lora preserves independent accelerator OK")


def test_slot_mapping_anima_like() -> None:
    wf = _load_fixture("anima_like.json")
    slots = detect_slots(wf)
    assert slots["prompt"]["node"] == "22"
    assert slots["artist"]["node"] == "20"
    assert slots["quality"]["node"] == "21"
    assert slots["trigger_words"]["node"] == "23"
    apply_slots(wf, slots, {"prompt": "1girl, garden", "artist": "(@x:1.0)", "trigger_words": "@t"})
    assert wf["22"]["inputs"]["prompt"] == "1girl, garden"
    assert wf["20"]["inputs"]["prompt"] == "(@x:1.0)"
    assert wf["23"]["inputs"]["prompt"] == "@t"
    print("  slot mapping anima-like OK")


def test_config_dropdown_and_size() -> None:
    assert parse_node_option("353 — CR Prompt Text — 主提示词") == "353"
    assert parse_node_option("") == ""
    slots = slots_from_config(
        {"prompt": "2 — CLIPTextEncode — 正面提示词", "sampler": "5 — KSampler"}
    )
    assert slots["prompt"]["node"] == "2"
    assert slots["sampler"]["node"] == "5"
    assert resolve_size(832, 1216, "landscape") == (1216, 832)
    assert resolve_size(832, 1216, "portrait") == (832, 1216)
    assert resolve_size(1024, 1024, "portrait") == (832, 1216)
    assert resolve_size(1024, 1024, "landscape") == (1216, 832)
    # 传裸节点 id 时不得重复插入同一 label（配置下拉曾因此出现双份选项）
    wf = _load_fixture("mini_workflow.json")
    for selected in ("2", "2 — CLIPTextEncode — 正面提示词", ""):
        options = node_options_for_slot(wf, "prompt", selected)
        assert len(options) == len(set(options)), options
    assert node_options_for_slot(wf, "prompt", "2")[1].startswith("2 —")
    print("  config dropdown parse OK")


def test_recipe_store_and_draw_schema() -> None:
    with tempfile.TemporaryDirectory() as td:
        store = RecipeStore(Path(td))
        wf = _load_fixture("mini_workflow.json")
        store.bootstrap(workflow_name="mini", wf=wf, config_slots={"prompt": "2", "sampler": "5"})
        rec = store.default()
        assert rec is not None
        assert rec["slots"]["prompt"]["node"] == "2"
        assert rec["defaults"]["model"] == "base.safetensors"
        assert "seed" not in rec["defaults"]
        store.save(
            {
                "name": "立绘",
                "workflow": "mini",
                "slots": rec["slots"],
                "defaults": rec["defaults"],
            }
        )
        assert "立绘" in store.names()
        catalog = store.catalog()
        assert "立绘" in catalog
        assert "base" in catalog
        preferred = RecipeStore(Path(td), preferred_default="立绘")
        assert preferred.default()["name"] == "立绘"
        hist_id = "abc123"
        store.save_history(
            {
                "prompt_id": hist_id,
                "workflow": "mini",
                "slots": rec["slots"],
                "values": rec["defaults"],
                "prompt": "1girl",
            }
        )
        copied = store.recipe_from_history(hist_id, "从历史")
        assert copied["name"] == "从历史"
        assert copied["defaults"]["model"] == "base.safetensors"
    print("  recipe store / draw schema OK")


def test_model_family_routing_and_recipe_decoupling() -> None:
    registry = ModelFamilyRegistry(
        [
            {
                "__template_key": "family",
                "name": "anima",
                "workflow": "anime-flow",
                "prompt_style": "danbooru",
            },
            {
                "__template_key": "family",
                "name": "Krea2",
                "workflow": "photo-flow",
                "prompt_style": "natural",
            },
        ]
    )
    assert registry.names() == ["anima", "Krea2"]
    assert registry.get("KREA2").workflow == "photo-flow"
    assert registry.by_workflow("anime-flow").name == "anima"

    with tempfile.TemporaryDirectory() as td:
        store = RecipeStore(Path(td))
        saved = store.save(
            {
                "name": "柔光立绘",
                "family": "Krea2",
                "workflow": "should-not-be-saved",
                "slots": {"prompt": {"node": "2"}},
                "defaults": {
                    "model": "krea2-photo.safetensors",
                    "loras": [{"name": "soft-light.safetensors", "strength": 0.7}],
                    "steps": 18,
                },
            }
        )
        assert saved["family"] == "Krea2"
        assert "workflow" not in saved and "template" not in saved and "slots" not in saved
        assert registry.resolve_recipe(saved).name == "Krea2"
        disk = json.loads(store.path_for(saved["id"]).read_text(encoding="utf-8"))
        assert "workflow" not in disk and "slots" not in disk
        disabled = store.save(
            {
                "name": "无 LoRA",
                "family": "anima",
                "defaults": {"loras": []},
            }
        )
        assert disabled["defaults"]["loras"] == []
    print("  model family routing / recipe decoupling OK")


def test_workflow_profile_store() -> None:
    with tempfile.TemporaryDirectory() as td:
        profiles = WorkflowProfileStore(Path(td))
        wf = _load_fixture("mini_workflow.json")
        effective = profiles.effective("mini", wf)
        assert effective["slots"]["prompt"]["node"] == "2"
        profiles.save(
            "mini",
            {
                **effective["slots"],
                "prompt": {"node": "3", "field": "text"},
            },
            ["99"],
        )
        reloaded = WorkflowProfileStore(Path(td)).effective("mini", wf)
        assert reloaded["slots"]["prompt"]["node"] == "3"
        assert reloaded["drop_nodes"] == ["99"]
        profiles.save("minimal", {"prompt": {"node": "2"}}, [])
        minimal = profiles.effective("minimal", wf)
        assert set(minimal["slots"]) == {"prompt"}

        configured = WorkflowProfileStore(
            Path(td),
            configured=[
                {
                    "__template_key": "mapping",
                    "workflow": "mini",
                    "prompt": "2 — CLIPTextEncode — 正面提示词",
                    "sampler": "5",
                    "sampler_2": "",
                }
            ],
        )
        assert configured.configured("MINI")["prompt"]["node"] == "2"
        configured_effective = configured.effective("mini", wf)
        assert configured_effective["source"] == "config"
        assert configured_effective["slots"]["prompt"]["node"] == "2"
        assert configured_effective["slots"]["sampler"]["node"] == "5"
        assert set(configured_effective["slots"]) == {"prompt", "sampler"}
        # 配置映射是运行时覆盖；磁盘中的 Workflow Studio 档案保持可回退。
        assert configured.get("mini")["slots"]["prompt"]["node"] == "3"
        fallback = WorkflowProfileStore(Path(td)).effective("mini", wf)
        assert fallback["source"] == "profile"
        assert fallback["slots"]["prompt"]["node"] == "3"

        legacy_store = RecipeStore(Path(td) / "legacy")
        legacy = legacy_store.save(
            {
                "name": "旧配方",
                "workflow": "legacy-flow",
                "slots": {"prompt": {"node": "7"}},
                "defaults": {"model": "base.safetensors"},
            }
        )
        assert profiles.import_legacy_recipe(legacy) is True
        assert profiles.get("legacy-flow")["slots"]["prompt"]["node"] == "7"
    print("  workflow profile store OK")


def test_llm_entry_schemas() -> None:
    schema = json.loads((PLUGIN / "_conf_schema.json").read_text(encoding="utf-8"))
    family_schema = schema["model_families"]
    assert family_schema["type"] == "template_list"
    assert {"name", "workflow", "prompt_style", "description"}.issubset(
        family_schema["templates"]["family"]["items"]
    )
    mapping_schema = schema["workflow_node_mappings"]
    assert mapping_schema["type"] == "template_list"
    assert {
        "workflow",
        "prompt",
        "model",
        "loras",
        "size",
        "sampler",
        "sampler_2",
        "negative",
        "artist",
        "quality",
        "trigger_words",
        "clip",
        "vae",
        "guidance",
    }.issubset(mapping_schema["templates"]["mapping"]["items"])
    registry = ModelFamilyRegistry(
        [
            {"name": "anima", "workflow": "anime-flow", "prompt_style": "danbooru"},
            {"name": "krea2", "workflow": "photo-flow", "prompt_style": "natural"},
        ]
    )
    draw = ComfyuiDrawTool(families=registry)
    draw.refresh_schema()
    assert draw.name == "comfyui_draw"
    assert draw.parameters["required"] == ["model_family", "prompt"]
    assert draw.parameters["properties"]["model_family"]["enum"] == ["anima", "krea2"]
    assert draw.parameters["properties"]["lora"]["type"] == "array"

    with tempfile.TemporaryDirectory() as td:
        store = RecipeStore(Path(td), preferred_default="柔光")
        store.save(
            {
                "name": "柔光",
                "family": "krea2",
                "defaults": {"loras": [{"name": "soft.safetensors", "strength": 0.7}]},
            }
        )
        recipe_draw = ComfyuiRecipeDrawTool(store=store, families=registry)
        recipe_draw.refresh_schema()
        assert recipe_draw.name == "comfyui_recipe_draw"
        assert recipe_draw.parameters["required"] == ["prompt"]
        assert recipe_draw.parameters["properties"]["recipe"]["enum"] == ["柔光"]
        recipe_tool = ComfyuiRecipeTool(store=store, families=registry)
        recipe_tool.refresh_schema()
        assert recipe_tool.parameters["required"] == ["action"]
        assert recipe_tool.parameters["properties"]["model_family"]["enum"] == ["anima", "krea2"]
        assert recipe_tool.parameters["properties"]["lora"]["type"] == "array"
        assert "description" in recipe_tool.parameters["properties"]
        assert "delete" not in recipe_tool.parameters["properties"]["action"]["enum"]
        recipe_tool.allow_delete = True
        recipe_tool.refresh_schema()
        assert "delete" in recipe_tool.parameters["properties"]["action"]["enum"]
    print("  llm entry schemas OK")


def test_family_and_recipe_generation_paths() -> None:
    import copy
    import types

    class FakeClient(ComfyUIClient):
        timeout = 1

        def __init__(self):
            self.submitted = []

        async def generation_resources(self):
            return (await self.list_resources())[0]

        async def list_resources(self):
            return (
                {
                    "unet_name": ["base.safetensors", "other.safetensors"],
                    "lora_name": ["style.safetensors", "soft.safetensors"],
                    "lora_meta": {n: {"model_family": "demo"} for n in ("style.safetensors", "soft.safetensors")},
                    "model_meta": {n: {"model_family": "demo"} for n in ("base.safetensors", "other.safetensors")},
                },
                False,
            )

        async def submit_prompt_detail(self, workflow):
            self.submitted.append(copy.deepcopy(workflow))
            return f"pid-{len(self.submitted)}", None

        async def get_history_entry(self, prompt_id):
            return {
                "status": {"status_str": "success"},
                "outputs": {
                    "6": {
                        "images": [
                            {
                                "filename": f"{prompt_id}.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                },
            }

        async def download_image(self, *args, **kwargs):
            return b"fake-image"

    class FakeEvent:
        unified_msg_origin = "test:family-routing"

        def __init__(self):
            self.sent = 0

        async def send(self, chain):
            self.sent += 1

    async def run():
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            builder = WorkflowBuilder(plugin_dir=PLUGIN, default_workflow="mini", custom_dir=root / "workflows")
            builder.save_template("mini", _load_fixture("mini_workflow.json"))
            registry = ModelFamilyRegistry(
                [{"name": "demo", "workflow": "mini", "prompt_style": "natural"}]
            )
            profiles = WorkflowProfileStore(root)
            profiles.ensure("mini", builder.load_template("mini"))
            store = RecipeStore(root, preferred_default="柔光")
            client = FakeClient()
            event = FakeEvent()
            context = types.SimpleNamespace(context=types.SimpleNamespace(event=event))
            draw = ComfyuiDrawTool(
                client=client,
                builder=builder,
                store=store,
                output_dir=root / "output",
                shared={},
                families=registry,
                profiles=profiles,
            )
            (root / "output").mkdir()
            result = await draw.call(
                context,
                model_family="demo",
                prompt="a silver cat",
                model="other",
                lora='[{"name":"soft","strength":0.55}]',
                steps=12,
                save_as="柔光",
            )
            assert "图片已发送" in result
            assert "本地路径:" in result
            assert store.list_history(1)[0]["family"] == "demo"
            assert client.submitted[-1]["2"]["inputs"]["text"] == "a silver cat"
            assert client.submitted[-1]["1"]["inputs"]["unet_name"] == "other.safetensors"
            assert client.submitted[-1]["7"]["inputs"]["lora_1"]["lora"] == "soft.safetensors"
            assert client.submitted[-1]["5"]["inputs"]["steps"] == 12
            saved = store.get("柔光")
            assert saved["family"] == "demo" and "workflow" not in saved

            recipe_draw = ComfyuiRecipeDrawTool(
                draw_tool=draw,
                store=store,
                families=registry,
            )
            result2 = await recipe_draw.call(context, recipe="柔光", prompt="a blue bird", seed=8)
            assert "图片已发送" in result2
            assert "本地路径:" in result2
            assert store.list_history(1)[0]["recipe"] == "柔光"
            assert client.submitted[-1]["2"]["inputs"]["text"] == "a blue bird"
            assert client.submitted[-1]["1"]["inputs"]["unet_name"] == "other.safetensors"
            assert client.submitted[-1]["5"]["inputs"]["seed"] == 8
            assert event.sent == 2

            builder.save_template("mini-v2", _load_fixture("mini_workflow.json"))
            moved_registry = ModelFamilyRegistry(
                [{"name": "demo", "workflow": "mini-v2", "prompt_style": "natural"}]
            )
            profiles.ensure("mini-v2", builder.load_template("mini-v2"))
            moved_draw = ComfyuiDrawTool(
                client=client,
                builder=builder,
                store=store,
                output_dir=root / "output",
                shared={},
                families=moved_registry,
                profiles=profiles,
            )
            moved_recipe_draw = ComfyuiRecipeDrawTool(
                draw_tool=moved_draw,
                store=store,
                families=moved_registry,
            )
            result3 = await moved_recipe_draw.call(context, recipe="柔光", prompt="new workflow")
            assert "图片已发送" in result3
            assert store.list_history(1)[0]["workflow"] == "mini-v2"
            assert event.sent == 3
            before = len(client.submitted)
            client.resource_family_rules = [
                {"kind": "lora", "family": "sdxl", "pattern": "soft.safetensors"}
            ]
            blocked = await recipe_draw.call(context, recipe="柔光", prompt="blocked")
            assert "家族不匹配" in blocked and len(client.submitted) == before

    asyncio.run(run())
    print("  family / recipe generation paths OK")


def test_qwen_edit_upload_and_output_path() -> None:
    import copy
    import types

    png = b"\x89PNG\r\n\x1a\nqwen-edit-input"
    edited_png = b"\x89PNG\r\n\x1a\nqwen-edit-output"
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
        "2": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {"prompt": "old prompt", "image": ["1", 0]},
        },
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
    }

    class FakeClient(ComfyUIClient):
        def __init__(self):
            self.uploads = []
            self.submitted = []
            self.timeout = 1

        async def upload_image(self, filename, content):
            self.uploads.append((filename, content))
            return filename, None

        async def submit_prompt_detail(self, submitted):
            self.submitted.append(copy.deepcopy(submitted))
            return f"edit-{len(self.submitted)}", None

        async def get_history_entry(self, prompt_id):
            return {
                "status": {"status_str": "success"},
                "outputs": {"3": {"images": [{"filename": "edited.png", "type": "output"}]}},
            }

        async def download_image(self, *args, **kwargs):
            return edited_png

    class FakeEvent:
        unified_msg_origin = "test:qwen-edit"

        def __init__(self, images):
            self.message_obj = types.SimpleNamespace(message=images)
            self.sent = 0

        async def send(self, chain):
            self.sent += 1

    async def run():
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.png"
            source.write_bytes(png)
            builder = WorkflowBuilder(plugin_dir=PLUGIN, custom_dir=root / "workflows")
            builder.save_template("qwen-edit", workflow)
            families = ModelFamilyRegistry([
                {"name": "qwen", "workflow": "qwen-generate", "edit_workflow": "qwen-edit"}
            ])
            assert families.editable_names() == ["qwen"]
            profiles = WorkflowProfileStore(root)
            profile = profiles.ensure("qwen-edit", builder.load_template("qwen-edit"))
            assert profile["slots"]["prompt"]["node"] == "2"
            assert profile["slots"]["source_image"]["node"] == "1"
            client = FakeClient()
            event = FakeEvent([Image(source)])
            context = types.SimpleNamespace(context=types.SimpleNamespace(event=event))
            tool = ComfyuiEditTool(
                client=client, builder=builder, store=RecipeStore(root),
                output_dir=root / "output", shared={}, families=families, profiles=profiles,
            )
            tool.refresh_schema()
            assert tool.parameters["properties"]["edit_workflow"]["enum"] == ["qwen"]
            result = await tool.call(context, model_family="qwen", prompt="make the sky blue")
            assert "图片已编辑并发送" in result and "本地路径:" in result
            assert client.uploads[0][1] == png
            assert client.submitted[0]["1"]["inputs"]["image"] == client.uploads[0][0]
            assert client.submitted[0]["2"]["inputs"]["prompt"] == "make the sky blue"
            assert workflow["1"]["inputs"]["image"] == "old.png"
            assert event.sent == 1
            saved = tool.store.list_history(1)[0]
            assert saved["entry"] == "edit" and saved["local_path"] in result

            event.message_obj.message = []
            second = await tool.call(
                context, model_family="qwen", prompt="add stars", image_path=saved["local_path"]
            )
            assert "图片已编辑并发送" in second and client.uploads[-1][1] == edited_png
            assert event.sent == 2

            third = await tool.call(context, model_family="qwen", prompt="make it warmer")
            assert "图片已编辑并发送" in third and client.uploads[-1][1] == edited_png
            event.message_obj.message = [Reply([Image(source)])]
            fourth = await tool.call(context, model_family="qwen", prompt="restore colors")
            assert "图片已编辑并发送" in fourth and client.uploads[-1][1] == png
            assert event.sent == 4
            assert client.uploads[0][0] == client.uploads[3][0], "identical edit inputs must keep their remote filename"

            event.message_obj.message = [Image(source), Image(source)]
            before = len(client.uploads)
            ambiguous = await tool.call(context, model_family="qwen", prompt="change color")
            assert "image_index" in ambiguous and len(client.uploads) == before
            rejected = await tool.call(
                context, model_family="qwen", prompt="change color", image_path=str(source)
            )
            assert "image_path" in rejected and len(client.uploads) == before

    asyncio.run(run())
    print("  Qwen edit upload / mapped nodes / output path OK")


def test_qwen_edit_graph_scoped_detection() -> None:
    import copy

    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "old.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "old prompt"}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "latent_image": ["4", 0], "steps": 30}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024}},
        "5": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
        "10": {"class_type": "LoadImage", "inputs": {"image": "orphan.png"}},
        "11": {"class_type": "TextEncodeQwenImageEdit", "inputs": {"prompt": "orphan", "image": ["10", 0]}},
        "12": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "13": {"class_type": "ImageScale", "inputs": {"image": ["12", 0]}},
        "14": {"class_type": "TextEncodeQwenImageEdit", "inputs": {"prompt": "edit", "image": ["13", 0], "clip": ["15", 0], "vae": ["16", 0]}},
        "15": {"class_type": "CLIPLoader", "inputs": {"clip_name": "edit-clip.safetensors"}},
        "16": {"class_type": "VAELoader", "inputs": {"vae_name": "edit-vae.safetensors"}},
        "17": {"class_type": "UNETLoader", "inputs": {"unet_name": "edit.safetensors"}},
        "18": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216}},
        "19": {"class_type": "KSampler", "inputs": {"model": ["17", 0], "positive": ["14", 0], "latent_image": ["18", 0], "steps": 20}},
        "20": {"class_type": "VAEDecode", "inputs": {"samples": ["19", 0], "vae": ["16", 0]}},
        "21": {"class_type": "SaveImage", "inputs": {"images": ["20", 0]}},
    }
    slots = detect_slots(wf)
    assert {role: spec["node"] for role, spec in slots.items() if role in {
        "prompt", "source_image", "sampler", "model", "clip", "vae", "size"
    }} == {
        "prompt": "14", "source_image": "12", "sampler": "19", "model": "17",
        "clip": "15", "vae": "16", "size": "18",
    }

    linked_prompt = copy.deepcopy(wf)
    linked_prompt["22"] = {"class_type": "CR Prompt Text", "inputs": {"prompt": "linked edit"}}
    linked_prompt["14"]["inputs"]["prompt"] = ["22", 0]
    assert detect_slots(linked_prompt)["prompt"]["node"] == "22"

    ambiguous_source = copy.deepcopy(wf)
    ambiguous_source["13"]["inputs"]["second_image"] = ["10", 0]
    slots = detect_slots(ambiguous_source)
    assert slots["prompt"]["node"] == "14" and "source_image" not in slots

    ambiguous_branch = copy.deepcopy(wf)
    ambiguous_branch["30"] = {"class_type": "LoadImage", "inputs": {"image": "other.png"}}
    ambiguous_branch["31"] = {"class_type": "TextEncodeQwenImageEdit", "inputs": {"prompt": "edit two", "image": ["30", 0]}}
    ambiguous_branch["32"] = {"class_type": "KSampler", "inputs": {"model": ["17", 0], "positive": ["31", 0], "latent_image": ["18", 0]}}
    ambiguous_branch["33"] = {"class_type": "SaveImage", "inputs": {"images": ["32", 0]}}
    assert detect_slots(ambiguous_branch) == {}
    with tempfile.TemporaryDirectory() as td:
        profiles = WorkflowProfileStore(
            Path(td), configured=[{"workflow": "qwen-edit", "prompt": "31", "source_image": "30"}]
        )
        manual = profiles.effective("qwen-edit", ambiguous_branch)
        assert manual["source"] == "config"
        assert manual["slots"]["prompt"]["node"] == "31"
        assert manual["slots"]["source_image"]["node"] == "30"
    print("  Qwen edit graph-scoped and ambiguous detection OK")


def test_split_family_config_dropdowns() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        builder = WorkflowBuilder(plugin_dir=root / "plugin", custom_dir=root / "workflows")
        wf = _load_fixture("mini_workflow.json")
        builder.save_template("qwen-generate", wf)
        builder.save_template("qwen-edit", {
            "1": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
            "2": {"class_type": "TextEncodeQwenImageEdit", "inputs": {"prompt": "edit", "image": ["1", 0]}},
        })
        store = RecipeStore(root)
        store.save({"name": "日常", "family": "qwen", "defaults": {}})

        schema = json.loads((_PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))
        class FakeConfig(dict):
            pass

        config = FakeConfig({
            "model_families": [
                {"__template_key": "family", "name": "qwen", "workflow": "qwen-generate", "edit_workflow": "legacy-edit"}
            ],
            "edit_families": [
                {"__template_key": "edit_family", "model_family": "qwen", "workflow": "qwen-edit"}
            ],
            "workflow_node_mappings": [{"workflow": "qwen-edit"}],
            "default_workflow": "qwen-generate",
            "default_recipe": "日常",
        })
        config.schema = schema
        refresh_config_options(config, builder, store)
        gen_items = schema["model_families"]["templates"]["family"]["items"]
        edit_items = schema["edit_families"]["templates"]["edit_family"]["items"]
        assert gen_items["workflow"]["options"] == ["qwen-edit", "qwen-generate"]
        assert edit_items["workflow"]["options"] == ["", "qwen-edit", "qwen-generate"]
        assert edit_items["model_family"]["options"] == ["qwen"]
        assert schema["workflow_node_mappings"]["templates"]["mapping"]["items"]["workflow"]["options"] == ["qwen-edit", "qwen-generate"]
        assert schema["default_recipe"]["options"] == ["日常"]
        assert config["model_families"][0]["workflow"] == "qwen-generate"
        dynamic = next(
            template for template in schema["workflow_node_mappings"]["templates"].values()
            if template["name"] == "qwen-edit 的节点映射"
        )
        assert dynamic["items"]["workflow"]["default"] == "qwen-edit"
        prompt_choice = next(x for x in dynamic["items"]["prompt"]["options"] if "TextEncodeQwenImageEdit" in x)
        image_choice = next(x for x in dynamic["items"]["source_image"]["options"] if "LoadImage" in x)
        assert len(dynamic["items"]["source_image"]["options"]) == 2
        parsed = slots_from_config({"prompt": prompt_choice, "source_image": image_choice})
        assert parsed["prompt"]["node"] == "2" and parsed["source_image"]["node"] == "1"

        registry = ModelFamilyRegistry(config["model_families"], edit_raw=config["edit_families"])
        assert registry.get("qwen").workflow == "qwen-generate"
        assert registry.get("qwen").edit_workflow == "qwen-edit"
        assert registry.editable_names() == ["qwen"]
        legacy = ModelFamilyRegistry(config["model_families"])
        assert legacy.get("qwen").edit_workflow == "legacy-edit"

        builder.save_template("new-local", wf)
        refresh_config_options(config, builder, store)
        assert "new-local" in gen_items["workflow"]["options"]
        config["model_families"][0]["workflow"] = "legacy-missing"
        refresh_config_options(config, builder, store)
        assert "legacy-missing" in gen_items["workflow"]["options"]
        assert any("legacy-missing" in label and "未找到" in label for label in gen_items["workflow"]["labels"])

    print("  split family config / live local dropdowns / legacy route OK")


def test_ui_to_api() -> None:
    ui = {
        "nodes": [
            {
                "id": 3,
                "type": "KSampler",
                "title": "采样",
                "widgets_values": [11, "randomize", 30, 4.5, "euler", "karras", 1],
                "inputs": [{"name": "model", "link": 1}],
            },
            {"id": 4, "type": "UNETLoader", "widgets_values": ["m.safetensors"]},
        ],
        "links": [[1, 4, 0, 3, 0, "MODEL"]],
    }
    wf = ui_to_api(ui)
    assert wf["3"]["class_type"] == "KSampler"
    assert wf["3"]["inputs"]["model"] == ["4", 0]
    assert wf["3"]["inputs"]["seed"] == 11
    assert wf["4"]["class_type"] == "UNETLoader"
    print("  ui to api OK")


def test_ui_to_api_retained_widgets_and_power_lora() -> None:
    """Anima-style UI exports retain linked widget slots and rgthree presets."""
    sampler_required = {
        "model": ["MODEL"],
        "add_noise": [["enable", "disable"]],
        "noise_seed": ["INT", {"default": 0}],
        "steps": ["INT", {"default": 20}],
        "cfg": ["FLOAT", {"default": 8.0}],
        "sampler": [["er_sde", "euler"]],
        "scheduler": [["sgm_uniform", "simple"]],
        "positive": ["CONDITIONING"],
        "negative": ["CONDITIONING"],
        "latent": ["LATENT"],
        "start_at_step": ["INT", {"default": 0}],
        "end_at_step": ["INT", {"default": 10000}],
        "return_with_leftover_noise": [["enable", "disable"]],
        "cleanup": [["不做任何清理"]],
    }
    object_info = {
        "XB_ROCmKSamplerAdvanced": {
            "input": {"required": sampler_required},
            "input_order": {"required": list(sampler_required)},
        },
        "Power Lora Loader (rgthree)": {
            "input": {"required": {"model": ["MODEL"], "clip": ["CLIP"]}}
        },
        "CR Prompt Text": {"input": {"required": {"prompt": ["STRING", {}]}}},
    }
    ui = {
        "nodes": [
            {
                "id": 10,
                "type": "XB_ROCmKSamplerAdvanced",
                "inputs": [
                    {"name": "model", "link": 1},
                    {"name": "steps", "link": 2},
                    {"name": "end_at_step", "link": 3},
                ],
                "widgets_values": [
                    "enable", 123, "randomize", 20, 4.6, "er_sde",
                    "sgm_uniform", 0, 10000, "enable", "不做任何清理",
                ],
            },
            {
                "id": 20,
                "type": "Power Lora Loader (rgthree)",
                "inputs": [{"name": "model", "link": 4}],
                "widgets_values": [
                    {}, {"type": "Power Lora Loader (rgthree)"},
                    {"on": False, "lora": "optional.safetensors", "strength": 0.8, "strengthTwo": 0.8},
                    {},
                ],
            },
            {"id": 30, "type": "CR Prompt Text", "widgets_values": ["fixed"]},
        ],
        "links": [
            [1, 1, 0, 10, 0, "MODEL"],
            [2, 2, 0, 10, 1, "INT"],
            [3, 3, 0, 10, 2, "INT"],
            [4, 1, 0, 20, 0, "MODEL"],
        ],
    }
    api = ui_to_api(ui, object_info)
    sampler = api["10"]["inputs"]
    assert sampler["steps"] == ["2", 0]
    assert sampler["end_at_step"] == ["3", 0]
    assert sampler["cfg"] == 4.6
    assert sampler["sampler"] == "er_sde"
    assert sampler["scheduler"] == "sgm_uniform"
    assert sampler["return_with_leftover_noise"] == "enable"
    assert sampler["cleanup"] == "不做任何清理"
    assert api["20"]["inputs"]["lora_1"] == {
        "on": False, "lora": "optional.safetensors", "strength": 0.8,
    }
    assert api["30"]["inputs"]["prompt"] == "fixed"

    snapshot = api_to_ui(api, object_info)
    rebuilt = next(node for node in snapshot["nodes"] if node["id"] == 10)
    assert rebuilt["widgets_values"] == [
        "enable", 123, "fixed", None, 4.6, "er_sde", "sgm_uniform",
        0, None, "enable", "不做任何清理",
    ]
    print("  ui to api retained widgets / Power LoRA slots OK")


def test_ui_import_prunes_unavailable_orphans() -> None:
    object_info = {
        "LoadImage": {"input": {"required": {"image": [["source.png"]]}}, "output": ["IMAGE"]},
        "SaveImage": {"input": {"required": {"images": ["IMAGE"]}}, "output_node": True},
    }
    ui = {
        "nodes": [
            {"id": 1, "type": "LoadImage", "widgets_values": ["source.png"]},
            {"id": 2, "type": "SaveImage", "inputs": [{"name": "images", "link": 1}]},
            {"id": 9, "type": "MarkdownNote", "widgets_values": ["UI-only note"]},
        ],
        "links": [[1, 1, 0, 2, 0, "IMAGE"]],
    }
    wf = normalize_workflow(ui, object_info)
    assert set(wf) == {"1", "2"}
    ui["nodes"][1]["inputs"][0]["link"] = 2
    ui["links"].append([2, 9, 0, 2, 0, "IMAGE"])
    try:
        normalize_workflow(ui, object_info)
    except ValueError as e:
        assert "9 (MarkdownNote)" in str(e)
    else:
        raise AssertionError("output-dependent missing nodes must be rejected")
    print("  UI import orphan pruning / active missing-node guard OK")


def test_api_to_ui_linked_widget_positions() -> None:
    """连线 widget 保留固定位置，后续值不得整体前移。"""
    sampler_inputs = {
        "model": ["MODEL"],
        "add_noise": [["enable", "disable"]],
        "noise_seed": ["INT", {"default": 0, "control_after_generate": True}],
        "steps": ["INT", {"default": 20}],
        "cfg": ["FLOAT", {"default": 8.0}],
        "sampler_name": [["euler_ancestral", "er_sde"]],
        "scheduler": [["sgm_uniform", "normal"]],
        "start_at_step": ["INT", {"default": 0}],
        "end_at_step": ["INT", {"default": 10000}],
        "return_with_leftover_noise": [["enable", "disable"]],
    }
    object_info = {
        "Int": {
            "input": {"required": {"value": ["INT", {"default": 0}]}},
            "output": ["INT"],
            "output_name": ["INT"],
        },
        "EmptyLatentImage": {
            "input": {
                "required": {
                    "width": ["INT", {"default": 512}],
                    "height": ["INT", {"default": 512}],
                    "batch_size": ["INT", {"default": 1}],
                }
            },
            "output": ["LATENT"],
            "output_name": ["LATENT"],
        },
        "KSamplerAdvanced": {"input": {"required": sampler_inputs}},
        "XB_ROCmKSamplerAdvanced": {"input": {"required": sampler_inputs}},
    }
    api = {
        "10": {"class_type": "Int", "inputs": {"value": 512}},
        "11": {"class_type": "Int", "inputs": {"value": 512}},
        "12": {"class_type": "Int", "inputs": {"value": 863754493834392}},
        "13": {"class_type": "Int", "inputs": {"value": 24}},
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": ["10", 0], "height": ["11", 0], "batch_size": 1},
        },
        "20": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["99", 0],
                "add_noise": "enable",
                "noise_seed": ["12", 0],
                "steps": 8,
                "cfg": 1,
                "sampler_name": "euler_ancestral",
                "scheduler": "sgm_uniform",
                "start_at_step": 0,
                "end_at_step": 7,
                "return_with_leftover_noise": "enable",
            },
        },
        "21": {
            "class_type": "XB_ROCmKSamplerAdvanced",
            "inputs": {
                "model": ["99", 0],
                "add_noise": "enable",
                "noise_seed": 123,
                "steps": ["13", 0],
                "cfg": 4.6,
                "sampler_name": "er_sde",
                "scheduler": "sgm_uniform",
                "start_at_step": 0,
                "end_at_step": 12,
                "return_with_leftover_noise": "enable",
            },
        },
        "22": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["99", 0],
                "add_noise": "disable",
                "noise_seed": 456,
                "steps": 16,
                "cfg": 2.5,
                "sampler_name": "er_sde",
                "scheduler": "normal",
                "start_at_step": 2,
                "end_at_step": 15,
                "return_with_leftover_noise": "disable",
            },
        },
    }

    ui = api_to_ui(api, object_info)
    by_id = {node["id"]: node for node in ui["nodes"]}
    assert by_id[4]["widgets_values"] == [None, None, 1]
    assert by_id[20]["widgets_values"] == [
        "enable",
        None,
        "fixed",
        8,
        1,
        "euler_ancestral",
        "sgm_uniform",
        0,
        7,
        "enable",
    ]
    assert by_id[21]["widgets_values"] == [
        "enable",
        123,
        "fixed",
        None,
        4.6,
        "er_sde",
        "sgm_uniform",
        0,
        12,
        "enable",
    ]
    assert by_id[22]["widgets_values"] == [
        "disable",
        456,
        "fixed",
        16,
        2.5,
        "er_sde",
        "normal",
        2,
        15,
        "disable",
    ]
    print("  api to ui linked widget positions OK")


def test_api_to_ui_composite_node_ids() -> None:
    api = {
        "13": {"class_type": "ResolutionSelector", "inputs": {"aspect_ratio": "1:1 (Square)"}},
        "459:470": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "459:474": {"class_type": "TextEncodeQwenImage21", "inputs": {"images.image_1": ["459:470", 0]}},
        "461": {"class_type": "SaveImageAdvanced", "inputs": {"images": ["459:474", 0]}},
    }
    snapshot = api_to_ui(api)
    nodes = snapshot["nodes"]
    node_ids = {node["id"] for node in nodes}
    assert len(nodes) == len(api)
    assert len(node_ids) == len(api) and all(isinstance(node_id, int) for node_id in node_ids)
    by_type = {node["type"]: node for node in nodes}
    assert set(by_type) == {"ResolutionSelector", "LoadImage", "TextEncodeQwenImage21", "SaveImageAdvanced"}
    assert len(snapshot["links"]) == 2
    assert {link[1] for link in snapshot["links"]}.issubset(node_ids)
    assert {link[3] for link in snapshot["links"]}.issubset(node_ids)
    type_by_id = {node["id"]: node["type"] for node in nodes}
    assert {(type_by_id[link[1]], type_by_id[link[3]]) for link in snapshot["links"]} == {
        ("LoadImage", "TextEncodeQwenImage21"),
        ("TextEncodeQwenImage21", "SaveImageAdvanced"),
    }
    print("  api to ui retained composite subgraph nodes / links OK")


def test_api_to_ui_qwen_combined_prompt_outputs() -> None:
    api = {
        "459:11": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "459:19": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "prompt": "positive prompt",
                "negative_prompt": "negative prompt",
                "images.image_1": ["459:11", 0],
            },
        },
        "6": {
            "class_type": "XB_ROCmKSampler",
            "inputs": {
                "positive": ["459:19", 0],
                "negative": ["459:19", 1],
                "latent": ["459:19", 2],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0]}},
        "18": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }
    object_info = {
        "LoadImage": {
            "input": {"required": {"image": [["source.png"]]}},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
        },
        "TextEncodeQwenImage21": {
            "input": {
                "required": {
                    "prompt": ["STRING", {}],
                    "negative_prompt": ["STRING", {}],
                    "images.image_1": ["IMAGE"],
                }
            },
            "input_order": {"required": ["prompt", "negative_prompt", "images.image_1"]},
            "output": ["CONDITIONING", "CONDITIONING", "LATENT"],
            "output_name": ["positive", "negative", "latent"],
        },
        "XB_ROCmKSampler": {
            "input": {"required": {
                "positive": ["CONDITIONING"],
                "negative": ["CONDITIONING"],
                "latent": ["LATENT"],
            }},
            "input_order": {"required": ["positive", "negative", "latent"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
        },
        "VAEDecode": {
            "input": {"required": {"samples": ["LATENT"]}},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
        },
        "SaveImage": {"input": {"required": {"images": ["IMAGE"]}}},
    }
    snapshot = api_to_ui(api, object_info)
    by_type = {node["type"]: node for node in snapshot["nodes"]}
    encoder = by_type["TextEncodeQwenImage21"]
    assert encoder["widgets_values"] == ["positive prompt", "negative prompt"]
    assert [output["name"] for output in encoder["outputs"]] == ["positive", "negative", "latent"]
    assert sum(node["type"] == "TextEncodeQwenImage21" for node in snapshot["nodes"]) == 1

    sampler = by_type["XB_ROCmKSampler"]
    links_by_id = {link[0]: link for link in snapshot["links"]}
    for field, output_index in (("positive", 0), ("negative", 1)):
        sampler_input = next(item for item in sampler["inputs"] if item["name"] == field)
        link = links_by_id[sampler_input["link"]]
        assert link[1] == encoder["id"] and link[2] == output_index
    print("  api to ui retained combined Qwen prompt fields / outputs OK")


def test_xb_sampler_seed_control_snapshot() -> None:
    """XB legacy aliases need a seed-control widget absent from /object_info."""
    normal_inputs = {
        "model": ["MODEL"],
        "seed": ["INT", {"default": 0}],
        "steps": ["INT", {"default": 20}],
        "cfg": ["FLOAT", {"default": 8.0}],
        "sampler": [["euler", "er_sde"]],
        "scheduler": [["simple", "karras"]],
        "positive": ["CONDITIONING"],
        "negative": ["CONDITIONING"],
        "latent": ["LATENT"],
        "denoise": ["FLOAT", {"default": 1.0}],
        "cleanup": [["不做任何清理"]],
    }
    advanced_inputs = {
        "model": ["MODEL"],
        "add_noise": [["enable", "disable"]],
        "noise_seed": ["INT", {"default": 0}],
        "steps": ["INT", {"default": 20}],
        "cfg": ["FLOAT", {"default": 8.0}],
        "sampler": [["euler", "er_sde"]],
        "scheduler": [["simple", "karras"]],
        "positive": ["CONDITIONING"],
        "negative": ["CONDITIONING"],
        "latent": ["LATENT"],
        "start_at_step": ["INT", {"default": 0}],
        "end_at_step": ["INT", {"default": 10000}],
        "return_with_leftover_noise": [["disable", "enable"]],
        "cleanup": [["不做任何清理"]],
    }
    object_info = {
        "XB_ROCmKSampler": {"input": {"required": normal_inputs}},
        "XB_ROCmKSamplerAdvanced": {"input": {"required": advanced_inputs}},
        "OtherSampler": {"input": {"required": {"seed": ["INT", {"default": 0}]}}},
    }
    api = {
        "18": {
            "class_type": "XB_ROCmKSampler",
            "inputs": {
                "seed": 1077777992415812,
                "steps": 25,
                "cfg": 1.0,
                "sampler": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "cleanup": "不做任何清理",
                "model": ["1", 0],
            },
        },
        "19": {
            "class_type": "XB_ROCmKSamplerAdvanced",
            "inputs": {
                "add_noise": "enable",
                "noise_seed": ["30", 0],
                "steps": 8,
                "cfg": 1.0,
                "sampler": "euler",
                "scheduler": "simple",
                "start_at_step": 0,
                "end_at_step": 10000,
                "return_with_leftover_noise": "disable",
                "cleanup": "不做任何清理",
            },
        },
        "20": {"class_type": "OtherSampler", "inputs": {"seed": 7}},
    }
    snapshot = build_extra_pnginfo(api, object_info)["workflow"]
    by_id = {node["id"]: node for node in snapshot["nodes"]}
    assert by_id[18]["widgets_values"] == [
        1077777992415812, "fixed", 25, 1.0, "euler", "simple", 1.0, "不做任何清理"
    ]
    assert by_id[19]["widgets_values"] == [
        "enable", None, "fixed", 8, 1.0, "euler", "simple", 0, 10000,
        "disable", "不做任何清理",
    ]
    assert by_id[20]["widgets_values"] == [7]
    print("  XB sampler seed-control PNG snapshot OK")


def test_workflow_build() -> None:
    """anima-v3 五段式覆盖 + LoRA 插槽 + KSampler。模板不存在则跳过。"""
    b = _builder()
    try:
        base = b.load_template("anima-v3")
        slots = detect_slots(base)
        wf = b.build(
            prompt="1girl, garden",
            artist="(@testartist:1.0)",
            trigger_words="@f1f",
            quality="masterpiece, best quality",
            negative_prompt="bad hands",
            model="m.safetensors",
            lora='[{"name":"l1.safetensors","strength":0.5}]',
            steps=20,
            cfg=1,
            seed=1,
        )
    except FileNotFoundError:
        print("  workflow build SKIP (anima-v3 未安装)")
        return
    for role, expected in (
        ("prompt", "1girl, garden"),
        ("artist", "(@testartist:1.0)"),
        ("trigger_words", "@f1f"),
    ):
        slot = slots[role]
        assert wf[slot["node"]]["inputs"][slot["field"]] == expected
    values = read_current_values(wf, slots)
    assert values["steps"] == 20 and values["cfg"] == 1
    assert values["model"] == "m.safetensors"
    assert values["loras"] == [{"name": "l1.safetensors", "strength": 0.5}]
    # 可选 LoRA 槽位的覆盖不应改变独立加速 LoRA。
    for node_id, node in base.items():
        if node["class_type"] == "LoraLoaderModelOnly" and node_id in wf:
            assert wf[node_id] == node
    print("  workflow build OK")


def test_defaults_precedence() -> None:
    """配方 defaults 覆盖：显式值优先，0/空视为未配置。种子不复用配方旧值。"""
    from comfyflow.storage.recipes import materialize_values

    recipe = {"defaults": {"model": "a.safetensors", "steps": 20, "width": 832, "seed": 7}}
    values = materialize_values(recipe, {"prompt": "1girl", "steps": 12, "width": None})
    assert values["model"] == "a.safetensors"
    assert values["steps"] == 12
    assert values["width"] == 832
    assert values["prompt"] == "1girl"
    assert "seed" not in values
    values2 = materialize_values(recipe, {"prompt": "1girl", "seed": 99})
    assert values2["seed"] == 99
    print("  defaults precedence OK")




def test_recipe_base_slots_binding() -> None:
    """配方保存继承基底工作流的槽位映射：默认配方优先，其次同工作流配方。"""
    with tempfile.TemporaryDirectory() as td:
        store = RecipeStore(Path(td))
        assert store.base_slots_for("mini") == ({}, "")
        wf = _load_fixture("mini_workflow.json")
        store.bootstrap(workflow_name="mini", wf=wf, config_slots={"prompt": "2", "sampler": "5"})
        slots, src = store.base_slots_for("mini")
        assert slots["prompt"]["node"] == "2"
        assert src == "默认"
        store.save(
            {
                "name": "立绘",
                "workflow": "mini",
                "slots": slots,
                "defaults": {"model": "a.safetensors"},
            }
        )
        slots2, src2 = store.base_slots_for("mini")
        assert slots2["prompt"]["node"] == "2"
        assert src2 == "默认"
        # 未绑定 / 无同工作流配方时无可继承映射
        assert store.base_slots_for("") == ({}, "")
        assert store.base_slots_for("other-wf") == ({}, "")
    print("  recipe base slots binding OK")


def test_dual_sampler_external_int() -> None:
    """双采样只写映射节点；显式第二段与共享外联节点按槽位联动。"""
    wf = _load_fixture("dual_sampler.json")
    slots = detect_slots(wf)
    assert slots["prompt"]["node"] == "2"
    assert slots["sampler"]["node"] in ("20", "21")
    assert slots.get("sampler_2", {}).get("node") in ("20", "21")
    assert slots["sampler"]["node"] != slots["sampler_2"]["node"]
    values = read_current_values(wf, slots)
    assert values["steps"] == 8
    assert "seed" not in values
    apply_slots(wf, slots, {"prompt": "cat", "steps": 20, "seed": 99})
    assert wf["2"]["inputs"]["text"] == "cat"
    assert wf["20"]["inputs"]["steps"] == ["10", 0]
    assert wf["21"]["inputs"]["steps"] == ["11", 0]
    assert wf["20"]["inputs"]["noise_seed"] == ["12", 0]
    assert wf["10"]["inputs"]["value"] == 20
    assert wf["11"]["inputs"]["value"] == 20
    assert wf["12"]["inputs"]["value"] == 99

    # 只映射第一段时，第二段独立的步数节点保持模板值。
    primary_only = _load_fixture("dual_sampler.json")
    primary_only["13"] = {"class_type": "Int", "inputs": {"value": 67890}}
    primary_only["21"]["inputs"]["noise_seed"] = ["13", 0]
    apply_slots(primary_only, {"sampler": {"node": "20"}}, {"steps": 30, "seed": 99})
    assert primary_only["10"]["inputs"]["value"] == 30
    assert primary_only["11"]["inputs"]["value"] == 8
    assert primary_only["12"]["inputs"]["value"] == 99
    assert primary_only["13"]["inputs"]["value"] == 67890

    # V6 式双采样共享同一个总步数节点时，写第一段即可让两段自然读取新值。
    shared = _load_fixture("dual_sampler.json")
    shared["21"]["inputs"]["steps"] = ["10", 0]
    apply_slots(shared, {"sampler": {"node": "20"}}, {"steps": 30})
    assert shared["10"]["inputs"]["value"] == 30
    assert shared["20"]["inputs"]["steps"] == ["10", 0]
    assert shared["21"]["inputs"]["steps"] == ["10", 0]

    # CFG 等浮点参数沿外联数值节点写入时保留小数并维持连线。
    linked_cfg = _load_fixture("dual_sampler.json")
    linked_cfg["13"] = {"class_type": "Int", "inputs": {"value": 1}}
    linked_cfg["20"]["inputs"]["cfg"] = ["13", 0]
    apply_slots(linked_cfg, {"sampler": {"node": "20"}}, {"cfg": 4.6})
    assert linked_cfg["13"]["inputs"]["value"] == 4.6
    assert linked_cfg["20"]["inputs"]["cfg"] == ["13", 0]

    # 映射到整数节点时同样能改步数
    wf2 = _load_fixture("dual_sampler.json")
    slots2 = dict(slots)
    slots2["sampler"] = {"node": "10", "field": "value"}
    apply_slots(wf2, slots2, {"steps": 16, "seed": 3})
    assert wf2["10"]["inputs"]["value"] == 16
    assert wf2["11"]["inputs"]["value"] == 16
    assert wf2["12"]["inputs"]["value"] == 3
    print("  dual sampler external int OK")


def test_collect_trigger_words() -> None:
    meta = {
        "a.safetensors": {"trigger_words": ["zoda", "1girl"]},
        "b.safetensors": {"trigger_words": ["zoda", "maid"]},
    }
    text = collect_trigger_words(
        meta, [{"name": "a.safetensors"}, {"name": "b.safetensors"}]
    )
    assert text == "zoda, 1girl, maid"
    assert collect_trigger_words(meta, []) == ""
    print("  collect trigger words OK")


def test_lora_list_input() -> None:
    """LLM 直接传数组对象（非 JSON 字符串）时，lora 不能丢。"""
    b = _builder()
    try:
        wf = b.build(
            prompt="1girl",
            lora=[{"name": "l1.safetensors", "strength": 0.5}],
        )
    except FileNotFoundError:
        print("  lora list/dict input SKIP (anima-v3 未安装)")
        return
    slots = detect_slots(b.load_template("anima-v3"))
    assert read_current_values(wf, slots)["loras"] == [{"name": "l1.safetensors", "strength": 0.5}]
    wf2 = b.build(prompt="1girl", lora={"name": "l2.safetensors", "strength": 0.7})
    assert read_current_values(wf2, slots)["loras"] == [{"name": "l2.safetensors", "strength": 0.7}]
    print("  lora list/dict input OK")


def test_template_management() -> None:
    with tempfile.TemporaryDirectory() as td:
        b2 = WorkflowBuilder(plugin_dir=PLUGIN, custom_dir=Path(td))
        wf = _load_fixture("mini_workflow.json")
        b2.save_template("mini", wf)
        names = [t["name"] for t in b2.list_templates()]
        assert "mini" in names
        b2.delete_template("mini")
    print("  template management OK")


def test_graph_workflow_validation() -> None:
    wf = {
        "445": {"class_type": "LoadImage", "inputs": {"image": "reference.png"}},
        "446": {"class_type": "SaveImage", "inputs": {"images": ["445", 0]}},
        "447": {"class_type": "CLIPTextEncode", "inputs": {"text": "original prompt"}},
    }
    ui = {
        "nodes": [
            {"id": 445, "type": "LoadImage", "outputs": [{"name": "IMAGE"}]},
            {"id": 446, "type": "SaveImage"},
            {"id": 447, "type": "CLIPTextEncode"},
        ],
        "links": [[1, 445, 0, 446, 0, "IMAGE"]],
    }
    assert set(validate_api_workflow(wf)) == {"445", "446", "447"}
    assert validate_ui_snapshot(ui, wf) is ui
    with tempfile.TemporaryDirectory() as td:
        builder = WorkflowBuilder(plugin_dir=PLUGIN, custom_dir=Path(td))
        builder.save_template("standard", wf)
        builder.save_ui_template("standard", ui)
        loaded = builder.load_template("standard")
        assert set(loaded) == {"445", "446", "447"}
        profiles = WorkflowProfileStore(Path(td) / "profiles")
        assert profiles.effective("standard", loaded)["drop_nodes"] == []
        apply_slots(loaded, {"prompt": {"node": "447", "field": "text"}}, {"prompt": "new prompt"}, drop_nodes=[])
        assert loaded["447"]["inputs"]["text"] == "new prompt"
        assert set(loaded) == {"445", "446", "447"}
        assert builder.load_ui_template("standard") == ui
        builder.delete_template("standard")
        assert builder.load_ui_template("standard") is None
    invalid = json.loads(json.dumps(wf))
    invalid["446"]["inputs"]["images"] = ["445", 1]
    try:
        validate_ui_snapshot(ui, invalid)
    except ValueError as e:
        assert "输出端口" in str(e)
    else:
        raise AssertionError("missing output port was accepted")
    print("  graph API/UI validation and standard node IDs OK")


def test_graph_webapi_submission() -> None:
    from comfyflow.studio import api as webapi_module

    wf = {
        "445": {"class_type": "LoadImage", "inputs": {"image": "reference.png"}},
        "446": {"class_type": "SaveImage", "inputs": {"images": ["445", 0]}},
    }
    ui = {
        "nodes": [
            {"id": 445, "type": "LoadImage", "outputs": [{"name": "IMAGE"}]},
            {"id": 446, "type": "SaveImage"},
        ],
        "links": [[1, 445, 0, 446, 0, "IMAGE"]],
    }

    class FakeClient:
        def __init__(self):
            self.submitted = []

        async def submit_prompt_detail(self, workflow, ui_workflow=None):
            self.submitted.append((workflow, ui_workflow))
            return "graph-pid", None

        async def get_history_entry(self, prompt_id):
            assert prompt_id == "graph-pid"
            return {"status": {"status_str": "success"}, "outputs": {
                "446": {"images": [{"filename": "graph.png", "type": "output"}]}
            }}

        async def download_image(self, filename, subfolder="", image_type="output"):
            assert filename == "graph.png" and image_type == "output"
            return b"\x89PNG\r\n\x1a\nimage-data"

    async def run():
        body = {"name": "standard", "workflow": wf, "ui_workflow": ui}

        async def fake_body():
            return body

        old_body, old_json, old_query = webapi_module._body, webapi_module._json, webapi_module._query
        webapi_module._body = fake_body
        webapi_module._json = lambda data, status=200: {**data, "_status": status}
        webapi_module._query = lambda key, default="": "graph-pid" if key == "pid" else default
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                builder = WorkflowBuilder(plugin_dir=PLUGIN, custom_dir=root / "workflows")
                store = RecipeStore(root / "recipes")
                client = FakeClient()
                shared = {}
                api = webapi_module.StudioApi(client, builder, store, root / "output", shared)
                saved = await api.save_graph()
                assert saved["ok"] and builder.load_ui_template("standard") == ui
                submitted = await api.run_graph()
                assert submitted["prompt_id"] == "graph-pid"
                assert client.submitted == [(wf, ui)]
                assert shared["web_pending_runs"]["graph-pid"]["entry"] == "graph"
                result = await api.generate_poll()
                assert result["done"] and result["data_url"].startswith("data:image/png;base64,")
                history = store.list_history(1)[0]
                assert history["entry"] == "graph" and history["workflow"] == "standard"
        finally:
            webapi_module._body, webapi_module._json, webapi_module._query = old_body, old_json, old_query

    asyncio.run(run())
    print("  graph WebUI save / remote submission / history OK")


def test_submit_error_parse() -> None:
    def handler(request):
        return httpx.Response(
            400,
            json={
                "error": {"message": "ValueError: 模型不存在"},
                "node_errors": {"458": {"errors": [{"message": "File not found", "node_id": "458"}]}},
            },
        )

    c = ComfyUIClient(host="t", port=1, cache_file=None)
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://t:1")

    async def run():
        pid, err = await c.submit_prompt_detail({})
        assert pid is None and err and "模型不存在" in err and "458" in err
        await c.close()

    asyncio.run(run())
    print("  submit error parse OK")


def test_execution_status_and_image_media_type() -> None:
    error = execution_error_message(
        {
            "status_str": "error",
            "completed": False,
            "messages": [
                ["execution_start", {"prompt_id": "p1"}],
                [
                    "execution_error",
                    {
                        "node_id": "428",
                        "node_type": "LoraLoaderModelOnly",
                        "exception_type": "OutOfMemoryError",
                        "exception_message": "CUDA out of memory\nwhile allocating tensor",
                    },
                ],
            ],
        }
    )
    assert "428" in error
    assert "LoraLoaderModelOnly" in error
    assert "OutOfMemoryError" in error
    assert "CUDA out of memory while allocating tensor" in error
    interrupted = execution_error_message(
        {
            "status_str": "error",
            "messages": [
                [
                    "execution_interrupted",
                    {"node_id": "510", "node_type": "XB_ROCmKSamplerAdvanced"},
                ]
            ],
        }
    )
    assert interrupted == "执行已中断（节点 510 (XB_ROCmKSamplerAdvanced)）"
    assert execution_error_message({"message": "legacy error"}) == "legacy error"

    assert image_media_type(b"\x89PNG\r\n\x1a\nrest", "wrong.webp") == "image/png"
    assert image_media_type(b"RIFF\x04\x00\x00\x00WEBPrest", "wrong.png") == "image/webp"
    assert image_media_type(b"\xff\xd8\xffrest", "x.bin") == "image/jpeg"
    assert image_media_type(b"GIF89arest", "x.bin") == "image/gif"
    print("  execution status / image media type OK")


def test_webapi_original_output_and_interrupt_guard() -> None:
    from comfyflow.studio import api as webapi_module

    png = b"\x89PNG\r\n\x1a\noriginal-png-bytes"

    class FakeClient:
        def __init__(self):
            self.download_calls = []
            self.interrupt_calls = []
            self.entries = {
                "ok-pid": {
                    "status": {"status_str": "success", "completed": True, "messages": []},
                    "outputs": {
                        "9": {
                            "images": [
                                {
                                    "filename": "preview.png",
                                    "subfolder": "previews",
                                    "type": "temp",
                                },
                                {
                                    "filename": "astrbot_test_00001_.png",
                                    "subfolder": "",
                                    "type": "output",
                                }
                            ]
                        }
                    },
                },
                "error-pid": {
                    "status": {
                        "status_str": "error",
                        "completed": False,
                        "messages": [
                            [
                                "execution_error",
                                {
                                    "node_id": "509",
                                    "node_type": "XB_ROCmKSamplerAdvanced",
                                    "exception_type": "RuntimeError",
                                    "exception_message": "sampler exploded",
                                },
                            ]
                        ],
                    },
                    "outputs": {},
                },
            }

        async def get_history_entry(self, prompt_id):
            return self.entries.get(prompt_id)

        async def download_image(
            self,
            filename,
            subfolder="",
            preview=None,
            image_type="output",
        ):
            self.download_calls.append((filename, subfolder, preview, image_type))
            return png

        async def interrupt(self, prompt_id=None):
            self.interrupt_calls.append(prompt_id)
            return True

    async def run():
        query = {"pid": "ok-pid"}
        body = {}

        def fake_query(key, default=""):
            return query.get(key, default)

        async def fake_body():
            return dict(body)

        def fake_json(data, status=200):
            return {**data, "_status": status}

        old_query, old_body, old_json = (
            webapi_module._query,
            webapi_module._body,
            webapi_module._json,
        )
        webapi_module._query = fake_query
        webapi_module._body = fake_body
        webapi_module._json = fake_json
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                output_dir = root / "output"
                output_dir.mkdir()
                store = RecipeStore(root)
                shared = {
                    "web_pending_runs": {
                        "ok-pid": {"workflow": "mini", "prompt": "cat"},
                        "error-pid": {"workflow": "mini", "prompt": "cat"},
                        "interrupt-pid": {"workflow": "mini", "prompt": "cat"},
                    }
                }
                client = FakeClient()
                api = webapi_module.StudioApi(
                    client,
                    None,
                    store,
                    output_dir,
                    shared,
                )

                success = await api.generate_poll()
                assert success["ok"] is True and success["done"] is True
                assert success["data_url"].startswith("data:image/png;base64,")
                assert client.download_calls == [
                    ("astrbot_test_00001_.png", "", None, "output")
                ]
                history = store.list_history(1)[0]
                saved_path = Path(history["local_path"])
                assert saved_path.suffix == ".png"
                assert saved_path.read_bytes() == png

                client.entries["preview-pid"] = {
                    "status": {"status_str": "success", "completed": True},
                    "outputs": {"9": {"images": [
                        {"filename": "preview.png", "subfolder": "previews", "type": "temp"}
                    ]}},
                }
                query["pid"] = "preview-pid"
                preview_result = await api.generate_poll()
                assert preview_result["ok"] is True
                assert client.download_calls[-1] == ("preview.png", "previews", None, "temp")
                assert Path(store.list_history(1)[0]["local_path"]).read_bytes() == png

                query["pid"] = "error-pid"
                failed = await api.generate_poll()
                assert "509" in failed["error"] and "sampler exploded" in failed["error"]
                assert "error-pid" not in shared["web_pending_runs"]

                body.clear()
                missing = await api.interrupt()
                assert missing["ok"] is False and missing["_status"] == 400
                assert client.interrupt_calls == []

                body["prompt_id"] = "unknown-pid"
                unknown = await api.interrupt()
                assert unknown["ok"] is False and unknown["_status"] == 404
                assert client.interrupt_calls == []

                body["prompt_id"] = "interrupt-pid"
                interrupted = await api.interrupt()
                assert interrupted["ok"] is True
                assert client.interrupt_calls == ["interrupt-pid"]
        finally:
            webapi_module._query = old_query
            webapi_module._body = old_body
            webapi_module._json = old_json

    asyncio.run(run())
    print("  webapi original output / interrupt guard OK")


def test_studio_recipe_workflow_independence() -> None:
    from comfyflow.studio import api as webapi_module
    from unittest.mock import patch

    class FakeConfig(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.saved = 0
            self.fail = False
            self.schema = json.loads((_PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))

        def save_config(self):
            if self.fail:
                raise OSError("disk unavailable")
            self.saved += 1

    class FakeTool:
        active = True

        def __init__(self):
            self.refreshes = 0

        def refresh_schema(self):
            self.refreshes += 1

    async def run():
        body = {}
        old_body, old_json, old_query = (
            webapi_module._body, webapi_module._json, webapi_module._query,
        )

        async def fake_body():
            return dict(body)

        def fake_json(data, status=200):
            return {**data, "_status": status}

        webapi_module._body, webapi_module._json = fake_body, fake_json
        webapi_module._query = lambda key, default="": default
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                builder = WorkflowBuilder(plugin_dir=PLUGIN, custom_dir=root / "workflows")
                for name in ("gen-a", "gen-b", "edit-a", "edit-b"):
                    builder.save_template(name, _load_fixture("mini_workflow.json"))
                store = RecipeStore(root)
                profiles = WorkflowProfileStore(root)
                profiles.save("gen-a", {"prompt": {"node": "2"}}, [])
                config = FakeConfig({
                    "model_families": [{"__template_key": "family", "name": "qwen", "workflow": "gen-a"}],
                    "edit_families": [{"__template_key": "edit_family", "model_family": "qwen", "workflow": "edit-a"}],
                    "workflow_node_mappings": [],
                })
                families = ModelFamilyRegistry(config["model_families"], edit_raw=config["edit_families"])
                draw_tool, recipe_tool, edit_tool = FakeTool(), FakeTool(), FakeTool()
                api = webapi_module.StudioApi(
                    None, builder, store, root / "output", {},
                    draw_tool, recipe_tool, edit_tool, families, profiles,
                    plugin_config=config,
                )
                routes = []

                class FakeContext:
                    def register_web_api(self, path, handler, methods, description):
                        routes.append(path)

                webapi_module.register_web_apis(
                    FakeContext(), None, builder, store, root / "output", {},
                    draw_tool, recipe_tool, edit_tool, families, profiles,
                    plugin_config=config,
                )
                assert f"/{webapi_module.PLUGIN_NAME}/workflow/bind" in routes

                body.update({
                    "name": "静态配方", "family": "qwen", "defaults": {"steps": 18},
                    "profile_slots": {"prompt": {"node": "999"}},
                    "drop_nodes": ["2"],
                })
                saved = await api.save_recipe()
                assert saved["ok"] is True
                assert store.get("静态配方")["defaults"]["steps"] == 18
                recipe_path = store.path_for(saved["recipe"]["id"])
                assert "slots" not in json.loads(recipe_path.read_text(encoding="utf-8"))
                assert profiles.get("gen-a")["slots"]["prompt"]["node"] == "2"
                original_recipe = store.get("静态配方")
                with patch.object(builder, "load_template", side_effect=AssertionError("recipe read touched workflow")):
                    loaded = await api.get_recipe()
                assert loaded["ok"] is True and loaded["recipe"]["name"] == "静态配方"

                body.clear()
                updated_workflow = _load_fixture("mini_workflow.json")
                updated_workflow["2"]["inputs"]["text"] = "updated workflow prompt"
                body.update({"name": "gen-a", "workflow": updated_workflow})
                imported = await api.import_workflow()
                assert imported["ok"] is True
                assert builder.load_template("gen-a")["2"]["inputs"]["text"] == "updated workflow prompt"
                assert store.get("静态配方") == original_recipe
                assert profiles.get("gen-a")["slots"]["prompt"]["node"] == "2"

                body.clear()
                body.update({
                    "workflow": "gen-a",
                    "slots": {"prompt": {"node": "2"}, "sampler": {"node": "5"}},
                    "drop_nodes": [],
                })
                mapped = await api.save_workflow_profile()
                assert mapped["ok"] is True
                assert profiles.get("gen-a")["slots"]["sampler"]["node"] == "5"
                assert store.get("静态配方") == original_recipe

                body.clear()
                body.update({"mode": "generate", "family": "qwen", "workflow": "gen-b"})
                bound = await api.save_workflow_binding()
                assert bound["ok"] is True
                assert families.get("qwen").workflow == "gen-b"
                assert builder.default_workflow == "gen-b"
                assert store.get("静态配方") == original_recipe
                assert profiles.get("gen-a")["slots"]["prompt"]["node"] == "2"

                body["mode"], body["workflow"] = "edit", "edit-b"
                bound_edit = await api.save_workflow_binding()
                assert bound_edit["ok"] is True
                assert any("LoadImage" in warning for warning in bound_edit["warnings"])
                assert families.get("qwen").edit_workflow == "edit-b"
                assert edit_tool.active is True
                assert config.saved == 2
                assert store.get("静态配方") == original_recipe

                body["workflow"] = ""
                unbound = await api.save_workflow_binding()
                assert unbound["ok"] is True
                assert families.get("qwen").edit_workflow == ""
                assert edit_tool.active is False
                assert store.get("静态配方") == original_recipe

                body["workflow"] = "missing"
                rejected = await api.save_workflow_binding()
                assert rejected["ok"] is False and config.saved == 3

                config.fail = True
                body["workflow"] = "edit-a"
                failed_save = await api.save_workflow_binding()
                assert failed_save["ok"] is False and config.saved == 3
                assert families.get("qwen").edit_workflow == ""
                assert config["edit_families"][0]["workflow"] == ""
        finally:
            webapi_module._body, webapi_module._json = old_body, old_json
            webapi_module._query = old_query

    asyncio.run(run())
    print("  studio recipe/workflow independence and routing OK")


def test_cache_atomic() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "models.json"
        c = ComfyUIClient(host="t", port=1, cache_file=p, cache_ttl=0)
        c._save_cache({"fetched_at": 1, "resources": {"unet_name": ["a.safetensors"]}})
        assert p.exists() and "a.safetensors" in p.read_text(encoding="utf-8")
    print("  cache atomic write OK")


def test_image_cache_lifecycle() -> None:
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor
    from comfyflow.storage.images import manage_cache, save_image

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "output"
        a = save_image(root, "../same.png", b"first")
        b = save_image(root, "same.png", b"second")
        assert a != b and a.parent == root.resolve()
        assert a.read_bytes() == b"first" and b.read_bytes() == b"second"
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda _: save_image(root, "same.png", b"first"), range(8)))
        assert set(paths) == {a}
        assert not list(root.glob("*.tmp"))
        assert manage_cache(root, action="clear")["protected"] == 2
        old = time.time() - 10 * 86400
        os.utime(a, (old, old))
        # Reusing an image refreshes its retention.
        save_image(root, "same.png", b"first")
        assert manage_cache(root, action="expired")["removed"] == 0
        os.utime(a, (old, old))
        keep = root / "history.json"
        keep.write_text("{}")
        nested = root / "nested"
        nested.mkdir()
        (nested / "keep.png").write_bytes(b"keep")
        result = manage_cache(root, action="expired", days=7, max_mb=0)
        assert result["removed"] == 1 and result["freed"] == 5 and b.exists()
        assert keep.exists() and (nested / "keep.png").exists()
        os.utime(b, (old, old))
        assert manage_cache(root, action="expired", days=0, max_mb=0)["removed"] == 0
        big = save_image(root, "large.png", b"x" * (1024 * 1024 + 1))
        os.utime(big, (old + 10, old + 10))
        limited = manage_cache(root, action="expired", days=0, max_mb=1)
        assert limited["bytes"] <= 1024 * 1024 and limited["removed"] == 2
        fresh = save_image(root, "clear.png", b"clear")
        os.utime(fresh, (old, old))
        assert manage_cache(root, action="clear")["removed"] == 1
    print("  image cache atomic dedup / retention / size / clear OK")


def test_manual_profile_skips_detection() -> None:
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as td:
        profiles = WorkflowProfileStore(Path(td))
        slots = {"prompt": {"node": "2", "field": "text"}}
        profiles.save("custom", slots)
        with patch("comfyflow.workflows.registry.detect_slots", side_effect=AssertionError("unexpected detection")):
            assert profiles.effective("custom", {})["slots"]["prompt"]["node"] == "2"
        try:
            apply_slots({}, slots, {"prompt": "test"})
        except ValueError as e:
            assert "映射" in str(e)
        else:
            raise AssertionError("stale prompt mapping must fail before submission")
    print("  manual profile bypass / stale mapping guard OK")


def test_resource_family_queries() -> None:
    from comfyflow.resources.catalog import filename_matches, filter_family, resource_family, selection_error
    from comfyflow.tools import ComfyuiLookupTool
    from comfyflow.tools.common import _match_resource

    resources = {
        "unet_name": [
            "Anima/base.safetensors", "SDXL/base.safetensors", "mystery.safetensors",
            "Anima/miao_miaoHarem_anima16.safetensors",
        ],
        "lora_name": [f"Anima/style{i:02}.safetensors" for i in range(16)] + [
            "Krea2/style.safetensors", "SDXL/style.safetensors", "unknown.safetensors",
            "Anima/wrong.safetensors",
        ],
        "lora_meta": {
            "Anima/wrong.safetensors": {"base_model": "SDXL 1.0"},
            "Anima/style00.safetensors": {"categories": ["watercolor"]},
        },
    }
    class Client(ComfyUIClient):
        resource_family_rules = []
        def __init__(self):
            pass
        async def generation_resources(self):
            return (await self.list_resources())[0]

        async def list_resources(self, **kwargs):
            return resources, False
        async def list_models_folder(self, folder):
            return resources["lora_name" if folder == "loras" else "unet_name"]

    assert resource_family("Anima/wrong.safetensors", {"base_model": "SDXL 1.0"})[0] == "sdxl"
    assert resource_family("Krea2\\test.safetensors")[0] == "krea2"
    assert resource_family("flux1-krea-dev.safetensors")[0] == "flux"
    assert resource_family("animal.safetensors")[0] == ""
    assert resource_family("Anima/test", {"base_model": "unrecognized architecture"})[0] == ""
    rule = [{"kind": "model", "family": "krea2", "pattern": "mystery.*"}]
    assert resource_family("mystery.safetensors", rules=rule, kind="model")[0] == "krea2"
    assert resource_family("mystery.safetensors", rules=rule, kind="lora")[0] == ""
    assert len(_match_resource(resources["unet_name"], "base.safetensors")) == 2
    assert filename_matches(["Anima/miao_miaoHarem_anima16.safetensors"], "miao miao harem")[0][1] == "name"
    assert filename_matches(["Anima/miao_miaoHarem_anima16.safetensors"], "miaomao")[0][1] == "near"
    assert not filename_matches(["Anima/base.safetensors"], "xy")
    assert "unknown.safetensors" not in filter_family(resources["lora_name"], {}, "anima")
    assert selection_error(resources, {"loras": [{"name": "SDXL/style.safetensors"}]}, "anima")
    assert selection_error(resources, {"model": "SDXL/base.safetensors"}, "anima")

    async def run():
        client = Client()
        lookup = ComfyuiLookupTool(client=client)
        summary = await lookup.call(None, type="lora")
        assert "anima=16" in summary and ".safetensors" not in summary
        first = await lookup.call(None, type="lora", model_family="anima", query="style", limit=100)
        assert first.count(".safetensors") == 10 and "next_offset=10" in first
        assert "Krea2/" not in first and "unknown.safetensors" not in first and "wrong" not in first
        named = await lookup.call(None, type="lora", model_family="anima", query="SDXL/style.safetensors")
        assert "SDXL/style.safetensors" in named and "兼容性待核对" in named
        second = await lookup.call(None, type="lora", model_family="anima", query="style", offset=10)
        assert "style10" in second and "style00" not in second
        models = await lookup.call(None, type="model", model_family="sdxl")
        assert "SDXL/base" in models and "Anima/base" not in models
        named = await lookup.call(None, type="model", query="miao miao harem")
        assert "Anima/miao_miaoHarem" in named and "家族=anima" in named
        approximate = await lookup.call(None, type="model", query="miaomao")
        assert "Anima/miao_miaoHarem" in approximate and "近似名称" in approximate
        scoped_approximate = await lookup.call(None, type="model", model_family="anima", query="miaomao")
        assert "Anima/miao_miaoHarem" in scoped_approximate
        scoped_miss = await lookup.call(None, type="model", model_family="sdxl", query="miaomao")
        assert "Anima/miao_miaoHarem" not in scoped_miss and "无匹配项" in scoped_miss
        exact_unknown = await lookup.call(None, type="model", query="mystery")
        assert "mystery.safetensors" in exact_unknown and "未确认兼容性" in exact_unknown
        assert "mystery.safetensors" not in await lookup.call(None, type="model", query="mysteri")
        unknown = await lookup.call(None, type="lora", model_family="unknown")
        assert "unknown.safetensors" in unknown and "未确认兼容性" in unknown
        listing = await lookup.call(None, type="lora", model_family="krea2")
        assert "Krea2/style" in listing and "SDXL/style" not in listing
        model_listing = await lookup.call(None, type="model", model_family="sdxl")
        assert "SDXL/base" in model_listing and "Anima/base" not in model_listing
        by_name = await lookup.call(None, type="all", query="miaomao")
        assert "底模: Anima/miao_miaoHarem" in by_name and "近似名称" in by_name
        assert "SDXL/base" not in by_name
        direct_name = await lookup.call(None, type="model", query="miao-miao")
        assert "Anima/miao_miaoHarem" in direct_name and "近似名称" not in direct_name
        lora_name = await lookup.call(None, type="lora", query="stile00")
        assert "Anima/style00" in lora_name and "近似名称" in lora_name
        purpose_without_family = await lookup.call(None, type="lora", query="watercolor")
        assert "Anima/style00" not in purpose_without_family
        purpose_with_family = await lookup.call(None, type="lora", model_family="anima", query="watercolor")
        assert "Anima/style00" in purpose_with_family
        folder = await lookup.call(None, type="folder", folder="unet", model_family="anima")
        assert "Anima/base" in folder and "SDXL/base" not in folder
        folder_name = await lookup.call(None, type="folder", folder="unet", query="miaomao")
        assert "Anima/miao_miaoHarem" in folder_name and "近似名称" in folder_name
        draw = ComfyuiDrawTool(client=client)
        name, error = await draw._resolve_model("base", "anima")
        assert name == "Anima/base.safetensors" and error is None
        name, error = await draw._resolve_model("SDXL/base.safetensors", "anima")
        assert name is None and error
        loras, error = await draw._resolve_loras("SDXL/style.safetensors", "anima")
        assert loras is None and error
        loras, error = await draw._resolve_loras("style", "krea2")
        assert loras[0]["name"] == "Krea2/style.safetensors" and error is None
        name, error = await draw._resolve_model("mystery", "anima")
        assert name is None and error
        # An explicitly selected unknown file stays usable without a false compatibility claim.
        name, error = await draw._resolve_model("mystery.safetensors", "anima")
        assert name == "mystery.safetensors" and error is None
    asyncio.run(run())
    print("  family filters / paging / ambiguity / generation guards OK")


def test_online_lora_identity() -> None:
    class Online:
        async def search_models(self, *args, **kwargs):
            return [
                {"modelVersions": [{"baseModel": "SDXL", "trainedWords": ["wrong"],
                                    "files": [{"name": "other.safetensors"}]}]},
                {"modelVersions": [{"baseModel": "SDXL", "trainedWords": ["wrong-family"],
                                    "files": [{"name": "test.safetensors"}]}]},
                {"modelVersions": [{"baseModel": "Anima", "trainedWords": ["correct"],
                                    "files": [{"name": "test.safetensors"}]}]},
            ]
    async def run():
        client = ComfyUIClient(host="test", port=1, lora_manager_enabled=False)
        async def empty(*args, **kwargs):
            return {"ss_base_model_version": "Anima"}
        client.get_model_metadata = empty
        client.civitai_client = Online()
        try:
            meta = await client._fetch_lora_trigger_words(["test.safetensors"], {})
            assert meta["test.safetensors"]["base_model"] == "Anima"
            assert meta["test.safetensors"]["trigger_words"] == ["correct"]
            assert ComfyUIClient.normalize_lora_metadata({"ss_base_model_version": "sdxl_base_v1-0"})["base_model"] == "sdxl_base_v1-0"
        finally:
            await client.close()
    asyncio.run(run())
    print("  online LoRA exact-file identity / header family OK")


def main() -> None:
    print("[smoke] astrbot_plugin_comfyui_direct 冒烟测试")
    test_slot_mapping_generic()
    test_power_lora_dynamic_slots()
    test_power_lora_reuses_matching_slot()
    test_power_lora_preserves_independent_accelerator()
    test_slot_mapping_anima_like()
    test_config_dropdown_and_size()
    test_recipe_store_and_draw_schema()
    test_model_family_routing_and_recipe_decoupling()
    test_workflow_profile_store()
    test_llm_entry_schemas()
    test_family_and_recipe_generation_paths()
    test_qwen_edit_upload_and_output_path()
    test_qwen_edit_graph_scoped_detection()
    test_split_family_config_dropdowns()
    test_ui_to_api()
    test_ui_to_api_retained_widgets_and_power_lora()
    test_ui_import_prunes_unavailable_orphans()
    test_api_to_ui_linked_widget_positions()
    test_api_to_ui_composite_node_ids()
    test_api_to_ui_qwen_combined_prompt_outputs()
    test_xb_sampler_seed_control_snapshot()
    test_workflow_build()
    test_defaults_precedence()
    test_recipe_base_slots_binding()
    test_dual_sampler_external_int()
    test_collect_trigger_words()
    test_lora_list_input()
    test_template_management()
    test_graph_workflow_validation()
    test_graph_webapi_submission()
    test_submit_error_parse()
    test_execution_status_and_image_media_type()
    test_webapi_original_output_and_interrupt_guard()
    test_studio_recipe_workflow_independence()
    test_cache_atomic()
    test_image_cache_lifecycle()
    test_manual_profile_skips_detection()
    test_resource_family_queries()
    test_online_lora_identity()
    print("ALL OK")


if __name__ == "__main__":
    main()
