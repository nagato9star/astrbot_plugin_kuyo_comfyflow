"""ComfyUI Direct — 局域网直连 ComfyUI API，提供图片生成与模型查询.

2026-08-11 v2.0.0 重构：
- 全部 HTTP 改 httpx.AsyncClient，不再阻塞事件循环
- 端点/端口/超时/缓存/默认模板等参数解耦到 _conf_schema.json，可自填 IP
- 工具改用 dataclass FunctionTool（v4.5.7+ 推荐模式）注册
- 内置 anima-v3 工作流模板（五段式提示词：画师串/质量/主提示词/lora触发词 + 反向）
- 模型/LoRA/CLIP 清单自动同步缓存，离线回退
- 工具支持可选模型、LoRA、KSampler 参数（steps/cfg/sampler_name/scheduler/denoise/seed）
- 队列查询 / 生成中断（comfyui_queue / comfyui_interrupt）
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

# AstrBot 按文件路径导入 main.py，不会把插件目录加入 sys.path。
# 先自举：把本目录插到 sys.path 最前，保证兄弟模块（comfy_client 等）可顶层导入。
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

from astrbot.api import AstrBotConfig, logger  # noqa: E402
from astrbot.api.event import AstrMessageEvent, filter  # noqa: E402
from astrbot.api.star import Context, Star, register  # noqa: E402
from astrbot.core.star.star_tools import StarTools  # noqa: E402

# 热重载兼容：AstrBot 重载插件时只清 data.plugins.<root>.* 前缀的模块，
# 顶层导入的兄弟模块（external_search/tools 等）会残留旧缓存，
# 导致改代码后热重载仍在跑旧逻辑。这里在导入前主动踢掉它们。
import sys as _sys  # noqa: E402

for _pkg_prefix in ("", "astrbot_plugin_comfyui_direct.", "astrbot_plugin_kuyo_comfyflow."):
    for _m in (
        "external_search",
        "tools",
        "comfy_client",
        "workflow_builder",
        "webapi",
        "slot_mapping",
        "recipe_store",
        "model_families",
        "image_cache",
        "resource_catalog",
        "config_options",
        "character_tags",
        "generation_support",
        "generation_queue",
        "completion_reply",
        "execution_events",
        "animadex",
        "api_to_ui",
    ):
        _sys.modules.pop(_pkg_prefix + _m, None)

from character_tags import CharacterTagLookup  # noqa: E402
from animadex import AnimaDexClient  # noqa: E402
from comfy_client import ComfyUIClient  # noqa: E402
from config_options import refresh_config_options  # noqa: E402
from generation_queue import GenerationQueue  # noqa: E402
from external_search import CivitaiClient, DanbooruClient, GelbooruClient  # noqa: E402
from tools import (  # noqa: E402
    ComfyuiAnimadexTool,
    ComfyuiBooruTool,
    ComfyuiCivitaiSearchTool,
    ComfyuiFetchOutputsTool,
    ComfyuiFreeMemoryTool,
    ComfyuiGenerateTool,
    ComfyuiInterruptTool,
    ComfyuiJobTool,
    ComfyuiListModelsTool,
    ComfyuiModelInfoTool,
    ComfyuiModelsSearchTool,
    ComfyuiNodesTool,
    ComfyuiQueueTool,
    ComfyuiRecipeTool,
    ComfyuiRecipeDrawTool,
    ComfyuiRunWorkflowTool,
    ComfyuiSystemStatsTool,
    ComfyuiUploadFileTool,
    ComfyuiValidateWorkflowTool,
    ComfyuiDrawTool,
    ComfyuiEditTool,
    ComfyuiLookupTool,
)
from workflow_builder import WorkflowBuilder  # noqa: E402
from model_families import EditWorkflowRegistry, ModelFamilyRegistry, WorkflowProfileStore  # noqa: E402
from recipe_store import RecipeStore  # noqa: E402


from image_cache import manage_cache  # noqa: E402

# 与 _conf_schema.json 一致的默认值（配置缺失时的兜底）
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8188
DEFAULT_TIMEOUT = 300
DEFAULT_CACHE_TTL = 600
DEFAULT_WORKFLOW = "anima-v3"

BASIC_LLM_TOOLS = {"comfyui_draw", "comfyui_edit", "comfyui_recipe_draw", "comfyui_lookup", "comfyui_job"}
# 这些工具可能读取任意本地文件、执行未经映射的自定义节点或影响其他任务，
# 默认不交给模型；需要时由管理员显式打开配置。
LLM_UNSAFE_TOOLS = {
    "comfyui_generate",
    "comfyui_run_workflow",
    "comfyui_upload_file",
    "comfyui_free_memory",
}


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse config booleans without treating the string ``"false"`` as true."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def parse_default_lora(raw: Any) -> str:
    """把配置的 default_lora（template_list 数组 / 旧字符串）转成工具可用的 JSON 数组字符串。

    返回 "" 表示未配置（用模板原值）；否则形如 [{"name": "...", "strength": 0.8}, ...]。
    """
    if isinstance(raw, list):
        entries = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            try:
                strength = float(item.get("strength") or 0.8)
            except (TypeError, ValueError):
                strength = 0.8
            entries.append({"name": name, "strength": strength})
        if not entries:
            return ""
        return json.dumps(entries, ensure_ascii=False)
    txt = str(raw or "").strip()
    if not txt:
        return ""
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, list):
            return txt  # 旧 JSON 数组原样使用
    except ValueError:
        pass
    # 裸文件名（旧下拉遗留）包装成单条
    return json.dumps([{"name": txt, "strength": 0.8}], ensure_ascii=False)


@register(
    "astrbot_plugin_kuyo_comfyflow",
    "长门九曜",
    "局域网直连ComfyUI API，按模型家族路由工作流并支持快捷配方",
    "2.7.2",
)
class ComfyUIDirectPlugin(Star):
    """通过局域网直连ComfyUI API生成图片和查询模型。"""

    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None) -> None:
        super().__init__(context, config)
        cfg = config or {}
        try:
            host = str(cfg.get("comfyui_host") or DEFAULT_HOST).strip() or DEFAULT_HOST
            port = int(cfg.get("comfyui_port") or DEFAULT_PORT)
            timeout = float(cfg.get("comfyui_timeout") or DEFAULT_TIMEOUT)
            ttl = int(cfg.get("model_cache_ttl") or DEFAULT_CACHE_TTL)
            default_workflow = (
                str(cfg.get("default_workflow") or DEFAULT_WORKFLOW).strip()
                or DEFAULT_WORKFLOW
            )
            # danbooru_base_url 支持逗号分隔多镜像（顺序回退，如官方站+自建镜像）
            danbooru_urls = tuple(
                u.strip()
                for u in str(cfg.get("danbooru_base_url") or "").split(",")
                if u.strip()
            ) or ("https://danbooru.donmai.us",)
            gelbooru_url = str(cfg.get("gelbooru_base_url") or "").strip() or "https://gelbooru.com"
            civitai_key = str(cfg.get("civitai_api_key") or "").strip()
            # 生成默认值：LLM 不传时使用（"" / 0 = 不覆盖，用模板原值）
            defaults: dict = {
                "artist": str(cfg.get("default_artist") or "").strip(),
                "quality": str(cfg.get("default_quality") or "").strip(),
                "trigger_words": str(cfg.get("default_trigger_words") or "").strip(),
                "negative_prompt": str(cfg.get("default_negative_prompt") or "").strip(),
                "model": str(cfg.get("default_model") or "").strip(),
                "lora": parse_default_lora(cfg.get("default_lora")),
                "sampler_name": str(cfg.get("default_sampler_name") or "").strip(),
                "scheduler": str(cfg.get("default_scheduler") or "").strip(),
                "steps": int(cfg.get("default_steps") or 0),
                "cfg": float(cfg.get("default_cfg") or 0),
                "denoise": float(cfg.get("default_denoise") or 0),
                "width": int(cfg.get("default_width") or 0),
                "height": int(cfg.get("default_height") or 0),
            }
        except (TypeError, ValueError) as e:
            logger.error(f"[ComfyUIDirect] 配置解析失败，使用默认值: {e}")
            host, port = DEFAULT_HOST, DEFAULT_PORT
            timeout, ttl = DEFAULT_TIMEOUT, DEFAULT_CACHE_TTL
            default_workflow = DEFAULT_WORKFLOW
            danbooru_urls = ("https://danbooru.donmai.us",)
            gelbooru_url = "https://gelbooru.com"
            civitai_key = ""
            defaults = {}

        data_dir = StarTools.get_data_dir("astrbot_plugin_kuyo_comfyflow")
        self._output_dir = data_dir / "output"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._cache_auto_clean = _as_bool(cfg.get("image_cache_auto_clean", True))
        self._cache_days = self._cache_limit(cfg.get("image_cache_days", 7), 7)
        self._cache_max_mb = self._cache_limit(cfg.get("image_cache_max_mb", 1024), 1024)
        self._cache_task = None
        llm_tool_mode = str(cfg.get("llm_tool_mode") or "basic").strip().lower()
        allow_llm_unsafe_tools = _as_bool(cfg.get("allow_llm_unsafe_tools", False))
        lora_manager_enabled = _as_bool(cfg.get("lora_manager_enabled", True))
        default_recipe_name = str(cfg.get("default_recipe") or "默认").strip() or "默认"
        node_slots_cfg = cfg.get("node_slots") if isinstance(cfg.get("node_slots"), dict) else {}
        self._node_slots_cfg = node_slots_cfg
        self._model_families_cfg = cfg.get("model_families") or []
        self._families = ModelFamilyRegistry(
            self._model_families_cfg,
            default_workflow,
            edit_raw=cfg.get("edit_families") or [],
        )
        self._edit_workflows = EditWorkflowRegistry(
            raw=cfg.get("edit_workflows") or [],
            legacy_edit_raw=cfg.get("edit_families") or [],
            legacy_families_raw=self._model_families_cfg,
        )

        self._client = ComfyUIClient(
            host=host,
            port=port,
            timeout=timeout,
            cache_file=data_dir / "comfyui_models.json",
            cache_ttl=ttl,
            lora_manager_enabled=lora_manager_enabled,
            request_timeout=float(cfg.get("comfyui_request_timeout") or 15),
            events_enabled=bool(cfg.get("comfyui_events_enabled", True)),
        )
        self._client.resource_family_rules = cfg.get("resource_family_rules") or []
        self._civitai = CivitaiClient(api_key=civitai_key)
        # 注入 civitai 客户端到 ComfyUIClient，用于本地无触发词时在线回退
        self._client.civitai_client = self._civitai
        self._builder = WorkflowBuilder(
            plugin_dir=Path(__file__).resolve().parent,
            default_workflow=self._families.first().workflow,
            custom_dir=data_dir / "workflows",
        )
        self._store = RecipeStore(data_dir, preferred_default=default_recipe_name)
        self._profiles = WorkflowProfileStore(
            data_dir,
            configured=cfg.get("workflow_node_mappings") or [],
        )
        self._initialize_family_data(
            node_slots_cfg,
            defaults,
        )
        refresh_config_options(config, self._builder, self._store)

        shared: dict = {}
        self._generation_queue = GenerationQueue(
            self.context, self._client, self._store, self._output_dir, data_dir, shared,
            max_pending=self._cache_limit(cfg.get("generation_queue_limit", 32), 32),
            concurrency=self._cache_limit(cfg.get("generation_queue_concurrency", 1), 1),
            notify_llm=_as_bool(cfg.get("generation_notify_llm", True)),
            llm_timeout=self._cache_limit(cfg.get("generation_notify_timeout", 60), 60),
        )
        shared["generation_queue"] = self._generation_queue
        self._danbooru = DanbooruClient(base_urls=danbooru_urls)
        self._gelbooru = GelbooruClient(base_url=gelbooru_url)
        self._animadex = AnimaDexClient(
            mcp_url=str(cfg.get("animadex_mcp_url") or "http://127.0.0.1:11451/mcp"),
            timeout=float(cfg.get("animadex_timeout") or 8.0),
        )

        self._character_lookup = CharacterTagLookup(
            data_dir, danbooru=self._danbooru,
            enabled=bool(cfg.get("character_names_enabled", True)),
            names_file=str(cfg.get("character_names_file") or ""),
            aliases=cfg.get("character_aliases") or [],
            reference_url=str(cfg.get("character_reference_url", "https://animadex.net")),
            timeout=float(cfg.get("character_lookup_timeout") or 6.0),
            cache_hours=float(cfg.get("character_cache_hours") or 168.0),
        )

        self._draw_tool = ComfyuiDrawTool(
            client=self._client,
            builder=self._builder,
            store=self._store,
            output_dir=self._output_dir,
            shared=shared,
            families=self._families,
            profiles=self._profiles,
            auto_trigger_words=bool(cfg.get("auto_lora_trigger_words", False)),
        )
        self._draw_tool.refresh_schema()
        self._edit_tool = ComfyuiEditTool(
            client=self._client,
            builder=self._builder,
            store=self._store,
            output_dir=self._output_dir,
            shared=shared,
            families=self._families,
            edit_workflows=self._edit_workflows,
            profiles=self._profiles,
        )
        self._edit_tool.refresh_schema()
        if not self._edit_workflows.names():
            self._edit_tool.active = False
        self._recipe_draw_tool = ComfyuiRecipeDrawTool(
            draw_tool=self._draw_tool,
            store=self._store,
            families=self._families,
        )
        self._recipe_draw_tool.refresh_schema()
        self._draw_tool.on_schema_change = self._recipe_draw_tool.refresh_schema
        self._generation_queue.on_schema_change = self._recipe_draw_tool.refresh_schema
        self._lookup_tool = ComfyuiLookupTool(
            danbooru=self._danbooru,
            gelbooru=self._gelbooru,
            animadex=self._animadex,
            client=self._client,
            character_lookup=self._character_lookup,
        )

        recipe_tool = ComfyuiRecipeTool(
            store=self._store,
            allow_delete=allow_llm_unsafe_tools,
            builder=self._builder,
            default_workflow=self._builder.default_workflow,
            families=self._families,
        )
        recipe_tool.refresh_schema()

        tools = [
            self._draw_tool,
            self._edit_tool,
            self._recipe_draw_tool,
            self._lookup_tool,
            ComfyuiListModelsTool(client=self._client),
            ComfyuiGenerateTool(
                client=self._client,
                builder=self._builder,
                output_dir=self._output_dir,
                shared=shared,
                defaults=defaults,
                store=self._store,
                families=self._families,
                profiles=self._profiles,
                auto_trigger_words=bool(cfg.get("auto_lora_trigger_words", False)),
            ),
            ComfyuiInterruptTool(client=self._client, shared=shared),
            ComfyuiQueueTool(client=self._client),
            ComfyuiBooruTool(danbooru=self._danbooru, gelbooru=self._gelbooru),
            ComfyuiCivitaiSearchTool(client=self._civitai),
            ComfyuiAnimadexTool(client=self._animadex),
            ComfyuiModelInfoTool(client=self._client, civitai=self._civitai),
            ComfyuiRunWorkflowTool(
                client=self._client, output_dir=self._output_dir, shared=shared
            ),
            ComfyuiJobTool(client=self._client, shared=shared),
            ComfyuiFetchOutputsTool(client=self._client, output_dir=self._output_dir),
            ComfyuiSystemStatsTool(client=self._client),
            ComfyuiFreeMemoryTool(client=self._client),
            ComfyuiNodesTool(client=self._client),
            ComfyuiValidateWorkflowTool(client=self._client),
            ComfyuiUploadFileTool(client=self._client),
            ComfyuiModelsSearchTool(client=self._client),
            recipe_tool,
        ]
        for t in tools:
            if llm_tool_mode != "full" and t.name not in BASIC_LLM_TOOLS:
                t.active = False
            elif t.name in LLM_UNSAFE_TOOLS and not allow_llm_unsafe_tools:
                t.active = False
        self.context.add_llm_tools(*tools)
        for t in tools:
            t.handler_module_path = self.__class__.__module__

        try:
            from webapi import register_web_apis

            register_web_apis(
                self.context,
                self._client,
                self._builder,
                self._store,
                self._output_dir,
                shared,
                self._draw_tool,
                self._recipe_draw_tool,
                self._edit_tool,
                self._families,
                self._profiles,
                config_defaults=defaults,
                plugin_config=config,
                auto_trigger_words=bool(cfg.get("auto_lora_trigger_words", False)),
                edit_workflows=self._edit_workflows,
            )
        except Exception as e:
            logger.error(f"[ComfyUIDirect] WebUI 接口注册失败: {e}")

        try:
            self._client.start_events()
            self._client.start_warmup()
        except Exception as e:
            logger.warning(f"[ComfyUIDirect] 启动预热失败: {e}")

        logger.info(
            f"[ComfyUIDirect] 已加载 v2.4.0 | ComfyUI: {self._client.base_url} "
            f"| 模型家族: {self._families.catalog()} | 工具模式: {llm_tool_mode} | 数据目录: {data_dir}"
        )

    def _initialize_family_data(
        self,
        node_slots_cfg: dict,
        defaults: dict,
    ) -> None:
        """初始化共享工作流档案，并为旧配方补上可确定的家族标签。"""
        for row in self._store.list():
            recipe = self._store.get(str(row.get("id") or ""))
            if not recipe:
                continue
            self._profiles.import_legacy_recipe(recipe)
            family = self._families.resolve_recipe(recipe)
            if family is not None:
                self._store.attach_legacy_family(str(recipe.get("id") or ""), family.name)

        loaded: dict[str, dict] = {}
        for family_row in self._families.list():
            workflow = family_row["workflow"]
            try:
                wf = self._builder.load_template(workflow)
            except FileNotFoundError:
                logger.info(
                    f"[ComfyUIDirect] 模型家族「{family_row['name']}」的工作流尚未导入: {workflow}"
                )
                continue
            except Exception as e:
                logger.warning(f"[ComfyUIDirect] 读取模型家族工作流失败 {workflow}: {e}")
                continue
            loaded[workflow] = wf
            configured = node_slots_cfg if workflow == self._builder.default_workflow else None
            self._profiles.ensure(workflow, wf, configured)

        recipe_defaults = dict(defaults)
        raw_lora = defaults.get("lora")
        if raw_lora:
            try:
                parsed = json.loads(raw_lora) if isinstance(raw_lora, str) else raw_lora
                if isinstance(parsed, list):
                    recipe_defaults["loras"] = parsed
            except ValueError:
                pass

        first_family = self._families.first()
        wf = loaded.get(first_family.workflow)
        has_legacy_slots = any(str(value or "").strip() for value in node_slots_cfg.values())
        has_legacy_defaults = has_legacy_slots or any(
            value not in (None, "", 0, 0.0, [], {})
            for value in recipe_defaults.values()
        )
        if wf is not None and has_legacy_defaults:
            # One-time migration for explicit old settings. Importing a workflow
            # alone must not create or alter a static recipe.
            self._store.bootstrap(
                workflow_name=first_family.workflow,
                wf=wf,
                config_slots=node_slots_cfg,
                config_defaults=recipe_defaults,
                family=first_family.name,
            )

    @staticmethod
    def _cache_limit(value: Any, default: int) -> int:
        try:
            return max(0, int(value))
        except (ValueError, TypeError, OverflowError):
            return default

    async def initialize(self) -> None:
        await self._generation_queue.start()
        if self._cache_auto_clean and self._cache_task is None:
            self._cache_task = asyncio.create_task(self._cache_cleanup_loop())

    @filter.on_waiting_llm_request()
    async def generation_turn_begin(self, event: AstrMessageEvent) -> None:
        self._generation_queue.begin_turn(event)

    @filter.after_message_sent()
    async def generation_turn_end(self, event: AstrMessageEvent) -> None:
        self._generation_queue.end_turn(event)

    async def _manage_image_cache(self, action: str) -> dict:
        protected_paths = {
            path
            for job in self._generation_queue.jobs.values()
            if job.get("delivery_state") in {"pending", "sending", "unknown"}
            for path in job.get("paths", [])
        }
        if action in {"expired", "status"}:
            shared = self._generation_queue.shared
            protected_paths.update(shared.get("last_image_paths", {}).values())
            protected_paths.update(
                result["path"]
                for result in shared.get("web_result_cache", {}).values()
                if result.get("path")
            )
        return await asyncio.to_thread(
            manage_cache, self._output_dir, action=action,
            days=self._cache_days, max_mb=self._cache_max_mb,
            protected_paths=protected_paths,
        )

    async def _cache_cleanup_loop(self) -> None:
        while True:
            try:
                result = await self._manage_image_cache("expired")
                if result["removed"] or result["errors"]:
                    logger.info(f"[ComfyUIDirect] 图片缓存清理: {result}")
            except Exception as e:
                logger.warning(f"[ComfyUIDirect] 图片缓存清理失败: {e}")
            await asyncio.sleep(3600)

    @filter.command("comfyui_cache")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def cache_command(self, event: AstrMessageEvent, action: str = "status"):
        """图片缓存：status 查看，expired 按配置清理，clear 清空（保留最近 5 分钟文件）。"""
        action = action.strip().lower()
        if action not in {"status", "clear", "expired"}:
            yield event.plain_result("用法：/comfyui_cache [status|expired|clear]")
            return
        try:
            result = await self._manage_image_cache(action)
        except OSError as e:
            yield event.plain_result(f"缓存操作失败：{e}")
            return
        yield event.plain_result(
            f"图片缓存：{result['files']} 个，{result['bytes'] / 1048576:.1f} MiB。"
            f"已清理 {result['removed']} 个，释放 {result['freed'] / 1048576:.1f} MiB；"
            f"近期文件保留 {result['protected']} 个，失败 {result['errors']} 个。\n"
            f"目录：{self._output_dir}"
        )

    async def terminate(self) -> None:
        """插件重载/卸载时关闭异步连接。"""
        await self._generation_queue.close()
        if self._cache_task is not None:
            self._cache_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cache_task
        await self._client.close()
        await self._character_lookup.close()
        await self._danbooru.close()
        await self._gelbooru.close()
        await self._civitai.close()
        await self._animadex.close()
        logger.info("[ComfyUIDirect] 已卸载，连接已关闭")
