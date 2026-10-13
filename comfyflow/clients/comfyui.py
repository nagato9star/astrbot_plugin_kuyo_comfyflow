"""ComfyUI HTTP 客户端：异步请求 + 模型/LoRA/CLIP 清单自动同步缓存.

2026-08-11 v2.0.0 重构：
- 改用 httpx.AsyncClient，全部异步，避免阻塞 AstrBot 事件循环
- 资源清单（UNET/LoRA/CLIP）自动同步并缓存到插件数据目录，离线回退缓存
- 端点/端口/超时由外部（配置）注入，不再写死
"""

from __future__ import annotations

import asyncio
import copy
import html
import json
import mimetypes
import re
import struct
import time
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from astrbot.api import logger
from ..resources.catalog import infer_family
from ..paths import PLUGIN_DIR

# ZeroTier 虚拟局域网偶尔抽风（瞬时丢包/路径切换），这类连接级异常可安全重试：
# - ConnectError/ConnectTimeout：TCP 连接未建立，请求肯定没发出去，重试无副作用
# - ReadTimeout：GET 幂等可重试；但 POST /prompt 例外（响应丢了不代表服务端没收到，
#   重试可能重复提交出双图），提交只对"连接未建立"类异常重试
RETRYABLE_EXC: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
)
SUBMIT_RETRYABLE_EXC: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
)
# 重试次数与退避（秒）：attempts 次尝试，间隔 backoff * 2^i
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 0.8


# /object_info 中各节点类的资源字段（class_type -> 字段名）
RESOURCE_FIELDS: dict[str, str] = {
    "UNETLoader": "unet_name",
    "UNETLoaderGGUF": "unet_name",
    "UnetLoaderGGUF": "unet_name",
    "CheckpointLoaderSimple": "unet_name",
    "CheckpointLoader": "unet_name",
    "LoraLoaderModelOnly": "lora_name",
    "LoraLoader": "lora_name",
    "LoraLoaderModelOnly (rgthree)": "lora_name",
    "CLIPLoader": "clip_name",
    "VAELoader": "vae_name",
}

# ComfyUI 自带节点和常见自定义节点使用的资源输入名。按输入名扫描比只按
# class_type 扫描更耐用，同时仍限制在明确的模型字段，避免把普通字符串选项混进清单。
RESOURCE_INPUT_TARGETS: dict[str, str] = {
    "unet_name": "unet_name",
    "ckpt_name": "unet_name",
    "lora_name": "lora_name",
    "clip_name": "clip_name",
    "vae_name": "vae_name",
}

# rgthree 的 Power Lora Loader：插槽字段 lora_1..lora_N 里也提供可选 LoRA 列表
POWER_LORA_CLASS = "Power Lora Loader (rgthree)"
# 模型文件扩展名（识别 object_info 里的资源文件名）
MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".sft", ".bin", ".gguf")

LORA_MANAGER_NEGATIVE_RETRY_SECONDS = 60.0
CIVITAI_LORA_CACHE_SECONDS = 24 * 60 * 60


def safe_output_path(output_dir: Path, filename: str) -> Path:
    """Return a path confined to output_dir, even if ComfyUI returns a path-like name."""
    name = PurePosixPath(str(filename).replace("\\", "/")).name
    if not name or name in (".", ".."):
        raise ValueError("ComfyUI 返回了无效输出文件名")
    root = output_dir.resolve()
    target = (root / name).resolve()
    if target.parent != root:
        raise ValueError("ComfyUI 输出文件名越界")
    return target


def image_media_type(content: bytes, filename: str = "") -> str:
    """根据文件魔数返回 data URL 使用的图片 MIME。"""
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type if mime_type and mime_type.startswith("image/") else "application/octet-stream"


def image_dimensions(content: bytes) -> tuple[int, int] | None:
    """Read width/height from the image formats accepted by comfyui_edit."""
    if content.startswith(b"\x89PNG\r\n\x1a\n") and len(content) >= 24:
        width, height = struct.unpack_from(">II", content, 16)
        return (width, height) if width and height else None
    if content.startswith((b"GIF87a", b"GIF89a")) and len(content) >= 10:
        width, height = struct.unpack_from("<HH", content, 6)
        return (width, height) if width and height else None
    if content.startswith(b"\xff\xd8\xff"):
        sof_markers = {
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        }
        offset = 2
        while offset + 4 <= len(content):
            if content[offset] != 0xFF:
                offset += 1
                continue
            while offset < len(content) and content[offset] == 0xFF:
                offset += 1
            if offset >= len(content):
                break
            marker = content[offset]
            offset += 1
            if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
                continue
            if offset + 2 > len(content):
                break
            segment_size = struct.unpack_from(">H", content, offset)[0]
            if segment_size < 2 or offset + segment_size > len(content):
                break
            if marker in sof_markers and segment_size >= 7:
                height, width = struct.unpack_from(">HH", content, offset + 3)
                return (width, height) if width and height else None
            offset += segment_size
    if len(content) >= 30 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        offset = 12
        while offset + 8 <= len(content):
            chunk_type = content[offset : offset + 4]
            chunk_size = struct.unpack_from("<I", content, offset + 4)[0]
            start = offset + 8
            end = start + chunk_size
            if end > len(content):
                break
            data = content[start:end]
            if chunk_type == b"VP8X" and len(data) >= 10:
                width = 1 + int.from_bytes(data[4:7], "little")
                height = 1 + int.from_bytes(data[7:10], "little")
                return (width, height) if width and height else None
            if chunk_type == b"VP8 " and len(data) >= 10 and data[3:6] == b"\x9d\x01\x2a":
                width = struct.unpack_from("<H", data, 6)[0] & 0x3FFF
                height = struct.unpack_from("<H", data, 8)[0] & 0x3FFF
                return (width, height) if width and height else None
            if chunk_type == b"VP8L" and len(data) >= 5 and data[0] == 0x2F:
                b1, b2, b3, b4 = data[1:5]
                width = 1 + ((b2 & 0x3F) << 8) + b1
                height = 1 + ((b4 & 0x0F) << 10) + (b3 << 2) + ((b2 & 0xC0) >> 6)
                return (width, height) if width and height else None
            offset = end + (chunk_size & 1)
    return None


def execution_error_message(status: Any, fallback: str = "执行出错") -> str:
    """从 ComfyUI ExecutionStatus.messages 提取节点级错误或中断原因。"""
    if not isinstance(status, dict):
        return fallback

    plain_message = ""
    messages = status.get("messages")
    if isinstance(messages, list):
        for item in reversed(messages):
            event = ""
            data: dict = {}
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                event = str(item[0] or "")
                data = item[1] if isinstance(item[1], dict) else {}
            elif isinstance(item, dict):
                event = str(item.get("type") or item.get("event") or "")
                nested = item.get("data")
                data = nested if isinstance(nested, dict) else item
            elif isinstance(item, str) and item.strip() and not plain_message:
                plain_message = item.strip()
                continue

            node_id = str(data.get("node_id") or "").strip()
            node_type = str(data.get("node_type") or data.get("class_type") or "").strip()
            node = f"节点 {node_id}" if node_id else ""
            if node_type:
                node = f"{node} ({node_type})" if node else f"节点类型 {node_type}"

            if event == "execution_interrupted":
                return f"执行已中断（{node}）" if node else "执行已中断"
            if event == "execution_error" or data.get("exception_message"):
                detail = str(
                    data.get("exception_message")
                    or data.get("message")
                    or data.get("exception_type")
                    or fallback
                ).strip()
                detail = " ".join(detail.splitlines())[:800]
                error_type = str(data.get("exception_type") or "").strip()
                if error_type and error_type not in detail:
                    detail = f"{error_type}: {detail}"
                return f"{node}：{detail}" if node else detail

    legacy = str(status.get("message") or "").strip()
    return legacy or plain_message or fallback


class ComfyUIClient:
    """封装与 ComfyUI 的全部 HTTP 交互。"""

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = 300.0,
        cache_file: Path | None = None,
        cache_ttl: int = 600,
        lora_manager_enabled: bool = True,
        request_timeout: float = 15.0,
        events_enabled: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.request_timeout = max(0.1, float(request_timeout))
        self.events_enabled = events_enabled
        self.cache_file = cache_file
        self.cache_ttl = cache_ttl
        self.lora_manager_enabled = bool(lora_manager_enabled)
        self._lora_manager_available: bool | None = None
        self._lora_manager_checked_at = 0.0
        # Civitai 只是最后的补全来源；按文件名缓存命中和未命中，避免每次
        # 资源清单 TTL 到期都对全部 LoRA 重复搜索。
        self._civitai_lora_cache: dict[str, tuple[float, dict | None]] = {}
        self._client: httpx.AsyncClient | None = None
        self._resource_lock = asyncio.Lock()
        # /object_info 缓存（构造 UI 快照用），进程内一次
        self._object_info: dict | None = None
        self._object_info_at = 0.0
        self._object_info_lock = asyncio.Lock()
        self._memory_cache: dict | None = None
        self._cache_stamp = None
        self._metadata_task: asyncio.Task | None = None
        self._full_metadata_task: asyncio.Task | None = None
        self._warmup_task: asyncio.Task | None = None
        self._events = None
        self._closed = False
        # 可选：civitai 客户端，用于本地 metadata 无触发词时在线回退
        self.civitai_client = None


    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.request_timeout, connect=min(5.0, self.request_timeout)),
            )
        return self._client


    async def close(self) -> None:
        """关闭底层连接，插件卸载（terminate）时调用。"""
        self._closed = True
        tasks = [task for task in (getattr(self, "_metadata_task", None), getattr(self, "_full_metadata_task", None), getattr(self, "_warmup_task", None)) if task and not task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if getattr(self, "_events", None) is not None:
            await self._events.close()
        if self._client is not None:
            await self._client.aclose()
            self._client = None


    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        timeout: float | None = None,
        attempts: int = RETRY_ATTEMPTS,
        retry_exc: tuple[type[Exception], ...] = RETRYABLE_EXC,
        log_errors: bool = True,
        label: str = "",
    ) -> httpx.Response | None:
        """带退避重试的 HTTP 请求；仅对连接级异常（ZeroTier 抽风）重试。

        4xx/5xx 等明确失败不重试。最终失败返回 None。
        log_errors=False 时静默失败（不打 ERROR 日志），用于预期可能 404 的探测请求。
        """
        last_exc: Exception | None = None
        for i in range(attempts):
            if self._closed:
                return None
            try:
                resp = await self.client.request(
                    method, path, params=params, json=json_body,
                    timeout=self.request_timeout if timeout is None else timeout,
                )
                resp.raise_for_status()
                return resp
            except retry_exc as e:
                last_exc = e
                if i < attempts - 1:
                    await asyncio.sleep(RETRY_BACKOFF * (2**i))
                    continue
            except httpx.HTTPError as e:
                last_exc = e
                break
        if log_errors:
            where = f" {label}" if label else ""
            logger.error(
                f"[ComfyUIDirect] {method} {self.base_url}{path}{where} 失败 "
                f"({attempts} 次尝试): {last_exc}"
            )
        return None


    async def _get(self, path: str, params: dict | None = None) -> Any:
        """GET（幂等，自动重试），失败返回 None。"""
        resp = await self._request_with_retry("GET", path, params=params)
        if resp is None:
            return None
        try:
            return resp.json()
        except ValueError as e:
            # 空响应体/非 JSON（如 ComfyUI 刚启动未就绪时）按失败处理
            logger.warning(f"[ComfyUIDirect] GET {self.base_url}{path} 响应非 JSON: {e}")
            return None

    async def get_object_info(self, force_refresh: bool = False) -> dict | None:
        async with self._object_info_lock:
            if (not force_refresh and self._object_info is not None
                    and time.monotonic() - self._object_info_at < max(self.cache_ttl, 60)):
                return self._object_info
            data = await self._get("/object_info")
            if isinstance(data, dict) and data:
                self._object_info = data
                self._object_info_at = time.monotonic()
                return self._object_info
            return None if force_refresh else self._object_info


    async def ping(self, timeout: float = 5.0) -> bool:
        """轻量连通性探测（用于 WebUI 状态灯），ZeroTier 抽风时重试 2 次。"""
        for i in range(2):
            try:
                resp = await self.client.get("/system_stats", timeout=timeout)
                if resp.status_code == 200:
                    return True
            except RETRYABLE_EXC:
                if i == 0:
                    await asyncio.sleep(0.8)
                    continue
            except httpx.HTTPError:
                pass
            return False
        return False

    async def _get_object_info_cached(self) -> dict | None:
        """拉取 /object_info（进程内缓存），失败返回 None。"""
        return await self.get_object_info()


    def _meta_debug(self, msg: str) -> None:
        """独立文件级日志：不依赖 astrbot logger，用于判定新代码是否真的在跑。"""
        try:
            p = PLUGIN_DIR / "metadata_debug.log"
            with open(p, "a", encoding="utf-8") as f:
                f.write(time.strftime('%m-%d %H:%M:%S') + ' ' + msg + chr(10))
        except Exception:  # noqa: BLE001
            pass

    async def submit_prompt_detail(
        self, workflow: dict, ui_workflow: dict | None = None
    ) -> tuple[str | None, str | None]:
        """提交工作流，返回 (prompt_id, 错误信息)。成功时错误信息为 None。

        提交时同步构造 UI 格式快照塞进 extra_data.extra_pnginfo，
        ComfyUI 保存 PNG 时会内嵌 workflow 元数据，前端拖图即可完整
        还原（含 rgthree lora 槽）。快照构造失败则裸提交兜底。

        400 时解析 ComfyUI 的 {"error": {...}, "node_errors": {...}} 结构，
        把真实失败原因（缺节点/模型不存在等）带回来，而不是笼统报"无法连接"。

        重试策略（ZeroTier 抽风防护）：只对"TCP 连接未建立"的异常重试
        （ConnectError/ConnectTimeout = 请求肯定没发出去，安全）；ReadTimeout
        不重试——服务端可能已收下任务，重试会重复出图，改为提示"可能已提交"。
        """
        if self._closed:
            return None, "插件连接已关闭，请重新加载后确认任务状态。"
        body: dict = {"prompt": workflow}
        events = self.start_events()
        if events is not None:
            body["client_id"] = events.client_id
        self._meta_debug("submit_called")
        try:
            try:
                from ..workflows.ui_conversion import build_extra_pnginfo
            except ImportError:
                from astrbot_plugin_kuyo_comfyflow.api_to_ui import build_extra_pnginfo  # 沙箱/包环境兜底

            if ui_workflow is not None:
                body["extra_data"] = {"extra_pnginfo": {"workflow": ui_workflow}}
                self._meta_debug("inject_original_ui")
            else:
                objinfo = await self._get_object_info_cached()
                if objinfo:
                    extra_pnginfo = build_extra_pnginfo(workflow, objinfo)
                    if extra_pnginfo:
                        body["extra_data"] = {"extra_pnginfo": extra_pnginfo}
                        self._meta_debug("inject_ok")
                    else:
                        self._meta_debug("build_returned_none")
                else:
                    self._meta_debug("objinfo_none")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ComfyUIDirect] 元数据注入失败，按裸提交继续: {e}")
            self._meta_debug(f"inject_fail: {e!r}")
        last_err: Exception | None = None
        for i in range(2):
            try:
                resp = await self.client.post("/prompt", json=body)
                last_err = None
            except SUBMIT_RETRYABLE_EXC as e:
                last_err = e
                if i == 0:
                    logger.warning(
                        f"[ComfyUIDirect] POST /prompt 连接失败（ZeroTier 抽风?），"
                        f"重试: {e}"
                    )
                    await asyncio.sleep(1.2)
                    continue
            except httpx.ReadTimeout as e:
                # 响应超时：任务可能已提交，不重试，让上层查队列确认
                logger.error(
                    f"[ComfyUIDirect] POST /prompt 响应超时，任务可能已提交，"
                    f"请查队列确认: {e}"
                )
                return None, "提交请求超时，任务可能已在排队（请用 comfyui_queue 确认是否重复）"
            except httpx.HTTPError as e:
                last_err = e
                break
            break
        if last_err is not None:
            logger.error(f"[ComfyUIDirect] POST {self.base_url}/prompt 失败: {last_err}")
            return None, f"无法连接 ComfyUI（{self.base_url}）：{last_err}"
        if resp.status_code != 200:
            msg = f"ComfyUI 返回 HTTP {resp.status_code}"
            hints: list[str] = []
            try:
                body = resp.json()
                err = body.get("error", {})
                node_errors = body.get("node_errors") or {}
                if err:
                    msg = f"{err.get('message', msg)}"
                if node_errors:
                    parts = []
                    for nid, detail in node_errors.items():
                        if not isinstance(detail, dict):
                            continue
                        cls = detail.get("class_type") or "?"
                        for e0 in detail.get("errors") or []:
                            info = e0.get("extra_info") or {}
                            param = info.get("input_name") or e0.get("details") or "?"
                            bad = info.get("input_value")
                            seg = f"节点{nid}({cls}) 参数「{param}」"
                            if bad is not None:
                                seg += f" 值「{str(bad)[:80]}」"
                            seg += f": {e0.get('message', '?')}"
                            parts.append(seg)
                            if info.get("input_name") in (
                                "unet_name",
                                "ckpt_name",
                                "lora_name",
                                "vae_name",
                            ):
                                hints.append(
                                    "模型/LoRA 文件名请用 comfyui_list_models 返回的完整路径"
                                    "（可能带子目录前缀，如 Anima\\xxx.safetensors）"
                                )
                    if parts:
                        msg += "（" + "；".join(parts[:3]) + "）"
            except (ValueError, AttributeError):
                pass
            if hints:
                msg += "。提示：" + "；".join(sorted(set(hints)))
            logger.error(f"[ComfyUIDirect] 提交失败: {msg}")
            return None, msg
        try:
            data = resp.json()
        except ValueError:
            return None, "ComfyUI 响应格式异常"
        pid = data.get("prompt_id")
        if not pid:
            logger.error("[ComfyUIDirect] 未获取到 prompt_id")
            return None, "ComfyUI 未返回 prompt_id"
        return pid, None


    async def poll_history(self, prompt_id: str) -> dict | None:
        """查询执行历史（outputs 部分）；未完成/不存在返回 None。"""
        history = await self._get(f"/history/{prompt_id}")
        if history and prompt_id in history:
            return history[prompt_id].get("outputs", {})
        return None

    async def get_history_entry(self, prompt_id: str) -> dict | None:
        """查询完整历史条目（含 status/message，用于识别执行失败）。"""
        history = await self._get(f"/history/{prompt_id}")
        if history and prompt_id in history:
            return history[prompt_id]
        return None

    async def list_history(self, max_items: int = 12) -> list[dict]:
        """最近执行记录，用于 WebUI 从 ComfyUI 导入工作流。"""
        history = await self._get("/history")
        if not isinstance(history, dict):
            return []
        rows: list[dict] = []
        for pid, entry in history.items():
            if not isinstance(entry, dict):
                continue
            prompt = entry.get("prompt")
            number = 0
            wf = None
            if isinstance(prompt, list) and len(prompt) >= 3:
                try:
                    number = int(prompt[0])
                except (TypeError, ValueError):
                    number = 0
                if isinstance(prompt[2], dict):
                    wf = prompt[2]
            rows.append(
                {
                    "prompt_id": pid,
                    "number": number,
                    "status": (entry.get("status") or {}).get("status_str") or "",
                    "workflow": wf,
                    "has_workflow": isinstance(wf, dict) and bool(wf),
                }
            )
        rows.sort(key=lambda r: r.get("number") or 0, reverse=True)
        return rows[:max_items]

    async def get_queue(self) -> dict | None:
        """GET /queue → {"queue_running": [...], "queue_pending": [...]}。"""
        return await self._get("/queue")

    async def get_system_stats(self) -> dict | None:
        """GET /system_stats → 系统/设备/显存信息。"""
        return await self._get("/system_stats")

    # /view_metadata 各资源类型的候选目录（safetensors 头部元数据）
    METADATA_FOLDERS: dict[str, list[str]] = {
        "unet_name": ["unet", "checkpoints"],
        "lora_name": ["loras"],
        "clip_name": ["clip"],
        "vae_name": ["vae"],
    }

    async def get_model_metadata(
        self, filename: str, folders: list[str] | None = None
    ) -> dict | None:
        """GET /view_metadata/{folder}?filename=：读 safetensors 头部 __metadata__。

        按资源目录候选逐个尝试；无元数据头/文件不存在/空响应返回 None。
        folders 传 None 时尝试全部资源目录，否则只试指定目录（避免逐目录 404）。
        404 时不走重试也不打 ERROR 日志，仅 debug 级别记录。
        """
        if folders is None:
            folders = [f for names in self.METADATA_FOLDERS.values() for f in names]
        for folder in folders:
            resp = await self._request_with_retry(
                "GET",
                f"/view_metadata/{folder}",
                params={"filename": filename},
                attempts=1,
                log_errors=False,
            )
            if resp is None:
                continue
            try:
                return resp.json()
            except ValueError:
                continue
        return None

    # ------------------------------------------------------------------
    # LoRA Manager metadata: remote ComfyUI plugin API, optional
    # ------------------------------------------------------------------

    _LORA_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
        "style": ("style", "styles", "风格", "画风"),
        "character": ("character", "characters", "角色", "人物"),
        "concept": ("concept", "concepts", "概念"),
        "outfit": ("outfit", "clothing", "服装", "穿搭"),
        "pose": ("pose", "poses", "姿势"),
        "background": ("background", "背景"),
        "effect": ("effect", "effects", "特效"),
        "object": ("object", "物体"),
    }

    @staticmethod
    def _metadata_values(value: Any) -> list[str]:
        if isinstance(value, (list, tuple, set)):
            values = value
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                values = parsed
            else:
                values = text.split(",")
        else:
            return []
        result: list[str] = []
        for item in values:
            text = str(item or "").strip(" \ufeff\u200b\u200c")
            if text and text not in result:
                result.append(text)
        return result

    @staticmethod
    def _clean_description(value: Any, limit: int = 480) -> str:
        if value is None:
            return ""
        text = html.unescape(re.sub(r"<[^>]+>", " ", str(value)))
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) <= limit else text[:limit].rstrip() + "…"

    @classmethod
    def _lora_categories(cls, tags: list[str]) -> list[str]:
        categories: list[str] = []
        for tag in tags:
            lowered = tag.casefold().strip()
            for category, aliases in cls._LORA_CATEGORY_ALIASES.items():
                if any(lowered == alias.casefold() for alias in aliases):
                    if category not in categories:
                        categories.append(category)
                    break
        return categories

    @staticmethod
    def _usage_tips(value: Any) -> dict | str | None:
        if isinstance(value, dict):
            return value or None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                return ComfyUIClient._clean_description(text, 240)
            if isinstance(parsed, dict):
                return parsed or None
            return ComfyUIClient._clean_description(text, 240)
        return None

    @classmethod
    def normalize_lora_metadata(
        cls, metadata: dict | None, fallback: dict | None = None
    ) -> dict:
        """Convert LoRA Manager/Civitai metadata into a small LLM-safe record."""
        metadata = metadata if isinstance(metadata, dict) else {}
        fallback = fallback if isinstance(fallback, dict) else {}
        civitai = metadata.get("civitai")
        civitai = civitai if isinstance(civitai, dict) else {}
        civitai_model = civitai.get("model")
        civitai_model = civitai_model if isinstance(civitai_model, dict) else {}

        tags: list[str] = []
        for value in (
            metadata.get("tags"),
            civitai_model.get("tags"),
            civitai.get("tags"),
            fallback.get("tags"),
        ):
            for tag in cls._metadata_values(value):
                if tag not in tags:
                    tags.append(tag)

        trigger_words: list[str] = []
        for value in (
            civitai.get("trainedWords"),
            metadata.get("trainedWords"),
            fallback.get("trigger_words"),
        ):
            for word in cls._metadata_values(value):
                if word not in trigger_words:
                    trigger_words.append(word)

        categories = cls._lora_categories(tags)
        for value in (
            metadata.get("categories"),
            metadata.get("category"),
            civitai_model.get("categories"),
            civitai_model.get("category"),
        ):
            for category in cls._metadata_values(value):
                if category not in categories:
                    categories.append(category)
        for category in cls._metadata_values(fallback.get("categories")):
            if category not in categories:
                categories.append(category)

        description = (
            metadata.get("modelDescription")
            or civitai_model.get("description")
            or metadata.get("description")
            or civitai.get("description")
            or fallback.get("description")
        )
        usage_tips = cls._usage_tips(
            metadata.get("usage_tips") or fallback.get("usage_tips")
        )
        notes = cls._clean_description(
            metadata.get("notes") or fallback.get("notes"), 240
        )
        base_model = (
            metadata.get("base_model")
            or metadata.get("baseModel")
            or metadata.get("ss_base_model_version")
            or metadata.get("modelspec.architecture")
            or civitai.get("baseModel")
            or fallback.get("base_model")
        )
        model_name = (
            metadata.get("model_name")
            or civitai_model.get("name")
            or fallback.get("model_name")
        )
        source = (
            metadata.get("_source")
            or metadata.get("metadata_source")
            or fallback.get("source")
            or ("lora_manager" if metadata else "")
        )

        result: dict[str, Any] = {}
        if trigger_words:
            result["trigger_words"] = trigger_words[: cls.TRIGGER_TOP_N]
        if source:
            result["source"] = source
        if categories:
            result["categories"] = categories
        if tags:
            result["tags"] = tags[:24]
        if model_name:
            result["model_name"] = cls._clean_description(model_name, 160)
        if base_model:
            result["base_model"] = cls._clean_description(base_model, 80)
        description = cls._clean_description(description)
        if description:
            result["description"] = description
        if usage_tips:
            result["usage_tips"] = usage_tips
        if notes:
            result["notes"] = notes
        return result

    @classmethod
    def _civitai_item_metadata(cls, item: dict) -> dict:
        version = (item.get("modelVersions") or [{}])[0]
        if not isinstance(version, dict):
            version = {}
        model = {
            "name": item.get("name") or "",
            "description": item.get("description") or "",
            "tags": item.get("tags") or [],
            "type": item.get("type") or "",
        }
        return {
            "_source": "civitai",
            "model_name": item.get("name") or "",
            "base_model": version.get("baseModel") or "",
            "modelDescription": item.get("description") or "",
            "tags": item.get("tags") or [],
            "civitai": {
                "type": item.get("type") or version.get("type") or "",
                "baseModel": version.get("baseModel") or "",
                "trainedWords": version.get("trainedWords") or [],
                "description": version.get("description") or "",
                "model": model,
            },
        }

    @staticmethod
    def _lora_key_variants(value: Any) -> list[str]:
        text = str(value or "").replace("\\", "/").strip().strip("/")
        if not text:
            return []
        lowered = text.casefold()
        base = lowered.rsplit("/", 1)[-1]
        stem = base
        for ext in MODEL_EXTS:
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        return list(dict.fromkeys((lowered, base, stem)))

    @classmethod
    def _index_lora_manager_items(cls, items: list[dict]) -> dict[str, dict]:
        index: dict[str, dict] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                merged = dict(metadata)
                for key, value in item.items():
                    merged.setdefault(key, value)
                item = merged
            for field in ("file_path", "relative_path", "file_name", "model_name", "name"):
                for key in cls._lora_key_variants(item.get(field)):
                    index.setdefault(key, item)
        return index

    def _lora_manager_probe_allowed(self) -> bool:
        if not self.lora_manager_enabled:
            return False
        if self._lora_manager_available is False:
            return (
                time.monotonic() - self._lora_manager_checked_at
                >= LORA_MANAGER_NEGATIVE_RETRY_SECONDS
            )
        return True

    def _set_lora_manager_available(self, available: bool) -> None:
        self._lora_manager_available = available
        self._lora_manager_checked_at = time.monotonic()

    async def get_lora_manager_catalog(self) -> dict[str, dict]:
        """Fetch the LoRA Manager listing and index it by path/name.

        The endpoint is optional; a missing LoRA Manager is treated as a normal
        condition so the existing safetensors/Civitai fallbacks keep working.
        """
        if not self._lora_manager_probe_allowed():
            return {}
        items: list[dict] = []
        page = 1
        total_pages = 1
        while page <= total_pages and page <= 100:
            resp = await self._request_with_retry(
                "GET",
                "/api/lm/loras/list",
                params={"page": page, "page_size": 100, "sort_by": "name"},
                attempts=1,
                log_errors=False,
            )
            if resp is None:
                self._set_lora_manager_available(False)
                return {}
            try:
                data = resp.json()
            except ValueError:
                self._set_lora_manager_available(False)
                return {}
            if isinstance(data, dict):
                page_items = data.get("items") or data.get("models") or []
                items.extend(x for x in page_items if isinstance(x, dict))
                try:
                    total_pages = max(1, int(data.get("total_pages") or 1))
                except (TypeError, ValueError):
                    total_pages = 1
            elif isinstance(data, list):
                items.extend(x for x in data if isinstance(x, dict))
                total_pages = 1
            else:
                self._set_lora_manager_available(False)
                return {}
            page += 1
        self._set_lora_manager_available(True)
        return self._index_lora_manager_items(items)

    async def get_lora_manager_metadata(self, filename: str) -> dict | None:
        """Fetch one LoRA Manager record for model-info fallback."""
        if not self._lora_manager_probe_allowed():
            return None
        resp = await self._request_with_retry(
            "GET",
            "/api/lm/loras/metadata",
            params={"file_path": filename},
            attempts=1,
            log_errors=False,
        )
        if resp is not None:
            try:
                data = resp.json()
            except ValueError:
                data = None
            if isinstance(data, dict):
                metadata = data.get("metadata")
                if isinstance(metadata, dict):
                    self._set_lora_manager_available(True)
                    return metadata

        # Older LoRA Manager versions expose the pieces separately.
        pieces: dict[str, Any] = {}
        for path, key, params in (
            ("/api/lm/loras/model-description", "modelDescription", {"file_path": filename}),
            ("/api/lm/loras/usage-tips-by-path", "usage_tips", {"relative_path": filename}),
            ("/api/lm/loras/get-trigger-words", "trainedWords", {"name": filename}),
        ):
            response = await self._request_with_retry(
                "GET", path, params=params, attempts=1, log_errors=False
            )
            if response is None:
                continue
            try:
                data = response.json()
            except ValueError:
                continue
            if isinstance(data, dict):
                value = data.get(key)
                if value not in (None, "", [], {}):
                    pieces[key] = value
        if pieces:
            self._set_lora_manager_available(True)
            return pieces
        return None

    # ------------------------------------------------------------------
    # LoRA 触发词：从 safetensors 头部元数据提取，随清单缓存
    # ------------------------------------------------------------------

    # 触发词候选元数据键（civitai 训练器写入 __metadata__）
    TRIGGER_KEYS = ("ss_activation_tags", "ss_tag_frequency", "ss_dataset_tags")
    # tag_frequency 按出现次数取 top N 展示
    TRIGGER_TOP_N = 12

    # 通用 danbooru 标签黑名单：出现在 ss_tag_frequency 高频区但不是风格触发词。
    # 这些 tag 几乎出现在所有训练集里，作为"触发词"只会污染提示词。
    _GENERIC_TAG_BLACKLIST: set[str] = {
        "1girl", "1boy", "solo", "looking at viewer", "smile", "open mouth",
        "closed mouth", "blush", "simple background", "white background",
        "long hair", "short hair", "black hair", "blonde hair", "brown hair",
        "blue eyes", "red eyes", "green eyes", "brown eyes", "purple eyes",
        "bangs", "hair between eyes", "upper body", "full body", "close-up",
        "breasts", "large breasts", "small breasts", "medium breasts",
        "sitting", "standing", "lying", "kneeling", "walking",
        "outdoors", "indoors", "sky", "cloudy sky", "blue sky",
        "shirt", "skirt", "dress", "pants", "shoes", "socks",
        "gloves", "hat", "glasses", "jewelry", "hair ornament",
        "collared shirt", "long sleeves", "short sleeves", "sleeveless",
        "bare shoulders", "bare legs", "bare arms", "barefoot",
        "blush stickers", "hetero", "twintails", "ponytail",
        "artwork", "official art", "highres", "lowres",
    }

    @staticmethod
    def _extract_trigger_words(meta: dict | None) -> tuple[list[str], str]:
        """从 safetensors 头部元数据提取 LoRA 触发词。

        优先级：ss_activation_tags（作者显式指定）> ss_tag_frequency（按频率 top N）
        > ss_dataset_tags（训练集全部 tag 截断）。返回 (触发词列表, 来源标识)。

        ss_tag_frequency 回退时会过滤通用 danbooru 标签，只保留可能为
        风格/角色触发词的 tag。频率极低（≤1）的也排除。
        """
        if not meta:
            return [], ""

        def _clean(tag: str) -> str:
            # 清洗训练器写入的脏字符（BOM/零宽/空白）
            return tag.strip(" \ufeff\u200b\u200c").strip()

        activation = meta.get("ss_activation_tags")
        if activation:
            tags = [_clean(t) for t in str(activation).split(",") if _clean(t)]
            if tags:
                return tags, "activation"
        freq = meta.get("ss_tag_frequency")
        if freq:
            try:
                data = json.loads(freq) if isinstance(freq, str) else freq
                if isinstance(data, dict):
                    def _cnt(v: Any) -> float:
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            return 0.0
                    counts: dict[str, float] = {}
                    for k, v in data.items():
                        if isinstance(v, dict):
                            # kohya 标准格式: {class: {tag: count}}，合并各 class
                            for t, c in v.items():
                                t = _clean(t)
                                if t:
                                    counts[t] = counts.get(t, 0.0) + _cnt(c)
                        else:
                            k = _clean(k)
                            if k:
                                counts[k] = counts.get(k, 0.0) + _cnt(v)
                    # 过滤通用标签 + 频率 ≤1 的杂项
                    bl = ComfyUIClient._GENERIC_TAG_BLACKLIST
                    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                    tags = [
                        t
                        for t, c in ranked[: ComfyUIClient.TRIGGER_TOP_N]
                        if c > 1 and t not in bl
                    ]
                    if tags:
                        return tags, "tag_frequency"
            except (ValueError, TypeError):
                pass
        dataset = meta.get("ss_dataset_tags")
        if dataset:
            tags = [_clean(t) for t in str(dataset).split(",") if _clean(t)]
            if tags:
                return tags[: ComfyUIClient.TRIGGER_TOP_N], "dataset"
        return [], ""

    async def _fetch_lora_trigger_words(
        self,
        lora_names: list[str],
        lora_manager_catalog: dict[str, dict] | None = None,
    ) -> dict[str, dict]:
        """同步 LoRA 的触发词、分类、标签和用途信息。

        优先级为 LoRA Manager → safetensors 头部 → Civitai 名称搜索。
        返回的是裁剪后的 LLM 安全记录，避免把完整 HTML/大对象塞进上下文。
        """
        sem = asyncio.Semaphore(8)
        online_sem = asyncio.Semaphore(4)
        out: dict[str, dict] = {}
        lora_manager_catalog = lora_manager_catalog or {}

        async def one(name: str) -> None:
            async with sem:
                # LoRA Manager 的 /list 已经带有 sidecar/Civitai 元数据。
                manager_meta = None
                for key in self._lora_key_variants(name):
                    manager_meta = lora_manager_catalog.get(key)
                    if manager_meta:
                        break
                if manager_meta is None and self._lora_manager_probe_allowed():
                    try:
                        manager_meta = await self.get_lora_manager_metadata(name)
                    except Exception as e:
                        logger.warning(f"[ComfyUIDirect] 读取 {name} LoRA Manager 信息失败: {e}")

                manager_info = self.normalize_lora_metadata(manager_meta)
                trusted_words = list(manager_info.get("trigger_words") or [])
                # LoRA Manager 通常已经带有 Civitai trainedWords；有触发词时
                # 不再为同一个文件额外读取 safetensors 头部。
                if manager_info.get("trigger_words"):
                    manager_info["trusted_trigger_words"] = trusted_words
                    out[name] = manager_info
                    return

                # 先试本地 safetensors 头部，再合并 LoRA Manager 信息。
                local_info: dict = {}
                try:
                    meta = await self.get_model_metadata(name, folders=["loras"])
                    words, source = self._extract_trigger_words(meta)
                    if source not in {"tag_frequency", "dataset"}:
                        trusted_words = list(dict.fromkeys([*trusted_words, *words]))
                    if words:
                        local_info = {"trigger_words": words, "source": source}
                    base_model = self.normalize_lora_metadata(meta).get("base_model")
                    if base_model:
                        local_info["base_model"] = base_model
                except Exception as e:
                    logger.warning(f"[ComfyUIDirect] 读取 {name} 本地触发词失败: {e}")

                info = self.normalize_lora_metadata(manager_meta, fallback=local_info)
                if info:
                    info["trusted_trigger_words"] = trusted_words
                    out[name] = info
                    if info.get("trigger_words"):
                        return

                # 本地没有 → civitai 在线回退；命中和未命中都做短期缓存。
                if self.civitai_client:
                    try:
                        # 去掉扩展名，用文件名 stem 搜索
                        stem = name
                        for ext in MODEL_EXTS:
                            if stem.casefold().endswith(ext):
                                stem = stem[: -len(ext)]
                                break
                        cache_key = stem.casefold().strip()
                        cached = self._civitai_lora_cache.get(cache_key)
                        if cached and time.monotonic() - cached[0] < CIVITAI_LORA_CACHE_SECONDS:
                            if cached[1]:
                                out[name] = self.normalize_lora_metadata(
                                    cached[1], fallback=info
                                )
                                declared = self.normalize_lora_metadata(cached[1]).get("trigger_words", [])
                                out[name]["trusted_trigger_words"] = list(dict.fromkeys([*trusted_words, *declared]))
                            return
                        async with online_sem:
                            items = await self.civitai_client.search_models(
                                stem, types="LORA", limit=3
                            )
                        selected: dict | None = None
                        for item in items:
                            # Search rank alone does not establish that an online
                            # LoRA matches this local file or model architecture.
                            basename = name.replace("\\", "/").rsplit("/", 1)[-1].casefold()
                            versions = [
                                version for version in item.get("modelVersions") or []
                                if any(str(f.get("name") or "").casefold() == basename
                                       for f in version.get("files") or [])
                            ]
                            if not versions:
                                continue
                            item = {**item, "modelVersions": versions}
                            local_family = infer_family(info.get("base_model") or "")
                            online_family = infer_family(versions[0].get("baseModel") or "")
                            if local_family and online_family and local_family != online_family:
                                continue
                            civitai_info = self.normalize_lora_metadata(
                                self._civitai_item_metadata(item), fallback=info
                            )
                            if civitai_info:
                                selected = self._civitai_item_metadata(item)
                                out[name] = self.normalize_lora_metadata(
                                    selected, fallback=info
                                )
                                declared = self.normalize_lora_metadata(selected).get("trigger_words", [])
                                out[name]["trusted_trigger_words"] = list(dict.fromkeys([*trusted_words, *declared]))
                                break
                        self._civitai_lora_cache[cache_key] = (
                            time.monotonic(), selected
                        )
                    except Exception as e:
                        logger.warning(f"[ComfyUIDirect] civitai 回退查询 {name} 失败: {e}")

        await asyncio.gather(*(one(n) for n in lora_names))
        return out


    async def get_embeddings(self) -> list[str] | None:
        """GET /embeddings → 嵌入模型名列表（去扩展名）。

        某些 ComfyUI 版本/配置下该端点返回空体或非 JSON，
        直接当作"无 embeddings"处理，不刷 WARN 日志。
        """
        resp = await self._request_with_retry(
            "GET", "/embeddings", attempts=1, log_errors=False
        )
        if resp is None:
            return None
        try:
            data = resp.json()
            if isinstance(data, list):
                return data
            return None
        except ValueError:
            return None

    async def free_memory(self, unload_models: bool = True, free_cache: bool = True) -> bool:
        """POST /free：卸载模型/清空执行器缓存，释放显存。"""
        body: dict = {}
        if unload_models:
            body["unload_models"] = True
        if free_cache:
            body["free_memory"] = True
        try:
            resp = await self.client.post("/free", json=body)
            return resp.status_code == 200
        except httpx.HTTPError as e:
            logger.error(f"[ComfyUIDirect] 释放显存失败: {e}")
            return False

    async def upload_image(self, filename: str, content: bytes) -> tuple[str | None, str | None]:
        """POST /upload/image：上传图片到 ComfyUI input 目录，返回 (文件名, 错误)。"""
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type or not mime_type.startswith("image/"):
            return None, "只允许上传图片文件（png/jpg/jpeg/webp/gif 等）"
        try:
            resp = await self.client.post(
                "/upload/image",
                files={"image": (filename, content, mime_type)},
                data={"overwrite": "true"},
            )
        except httpx.HTTPError as e:
            logger.error(f"[ComfyUIDirect] 上传图片失败: {e}")
            return None, f"上传失败：{e}"
        if resp.status_code != 200:
            return None, f"ComfyUI 返回 HTTP {resp.status_code}"
        try:
            data = resp.json()
        except ValueError:
            return None, "ComfyUI 响应格式异常"
        name = data.get("name") or data.get("subfolder")
        if not name:
            return None, "ComfyUI 未返回文件名"
        return name, None

    async def list_models_folder(self, folder: str) -> list[str] | None:
        """GET /models/{folder}：列出某目录下的模型文件。folder 如 loras/checkpoints。"""
        resp = await self._request_with_retry(
            "GET", f"/models/{folder}", label=f"列模型 {folder}"
        )
        if resp is None:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        if not isinstance(data, list):
            return None
        out = []
        for it in data:
            name = it if isinstance(it, str) else (it or {}).get("name")
            if isinstance(name, str) and name.casefold().endswith(MODEL_EXTS):
                out.append(name)
        return out

    async def interrupt(self, prompt_id: str | None = None) -> bool:
        """POST /interrupt：带 prompt_id 时定向中断该任务（新版），否则全局中断。"""
        try:
            resp = await self.client.post(
                "/interrupt",
                json={"prompt_id": prompt_id} if prompt_id else {},
            )
            return resp.status_code == 200
        except httpx.HTTPError as e:
            logger.error(f"[ComfyUIDirect] 中断请求失败: {e}")
            return False

    async def delete_queue_items(self, prompt_ids: list[str]) -> bool:
        """POST /queue {"delete": [...]}：从待执行队列移除任务（对运行中任务无效）。"""
        if not prompt_ids:
            return True
        try:
            resp = await self.client.post("/queue", json={"delete": prompt_ids})
            return resp.status_code == 200
        except httpx.HTTPError as e:
            logger.error(f"[ComfyUIDirect] 移除队列任务失败: {e}")
            return False

    async def download_image(
        self,
        filename: str,
        subfolder: str = "",
        preview: str | None = None,
        image_type: str = "output",
    ) -> bytes | None:
        """从 ComfyUI /view 下载图片（GET 幂等，ZeroTier 抽风时自动重试）。

        preview 形如 "webp;80" / "jpeg;80"：让服务端重编码预览；None 返回原图。
        """
        params: dict = {"filename": filename, "type": image_type}
        if subfolder:
            params["subfolder"] = subfolder
        if preview:
            params["preview"] = preview
        resp = await self._request_with_retry(
            "GET", "/view", params=params, label=f"下载 {filename}"
        )
        if resp is None:
            return None
        return resp.content

    # ------------------------------------------------------------------
    # 模型/LoRA/CLIP 资源清单：自动同步 + 本地缓存
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_resources(obj: dict) -> dict[str, list[str]]:
        """从 /object_info 提取各节点类的资源清单。"""
        out: dict[str, list[str]] = {
            "unet_name": [],
            "lora_name": [],
            "clip_name": [],
            "vae_name": [],
        }
        seen = {field: set() for field in out}

        def append_options(target: str, raw: Any) -> None:
            if not isinstance(raw, list) or not raw:
                return
            options = raw[0] if isinstance(raw[0], list) else []
            for value in options:
                if isinstance(value, str) and value and value not in seen[target]:
                    seen[target].add(value)
                    out[target].append(value)

        # 同时扫 required/optional，兼容自定义节点把模型选择器声明在 optional 的情况。
        for info in obj.values():
            if not isinstance(info, dict):
                continue
            inputs = info.get("input") or {}
            for section in (inputs.get("required") or {}, inputs.get("optional") or {}):
                if not isinstance(section, dict):
                    continue
                for input_name, target in RESOURCE_INPUT_TARGETS.items():
                    if input_name in section:
                        append_options(target, section[input_name])

        # rgthree 的 Power Lora Loader：lora 插槽是嵌套结构（如 lora_1 里含 "lora": [[...]]），
        # 递归收集所有含 lora 键名的模型文件名
        pl = obj.get(POWER_LORA_CLASS, {})
        for key, val in pl.get("input", {}).get("required", {}).items():
            if "lora" not in key.lower():
                continue
            if (
                isinstance(val, list)
                and val
                and isinstance(val[0], list)
            ):
                for name in val[0]:
                    if isinstance(name, str) and name.casefold().endswith(MODEL_EXTS) and name not in out["lora_name"]:
                        out["lora_name"].append(name)
            elif isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if "lora" not in sub_key.lower():
                        continue
                    if isinstance(sub_val, list) and sub_val and isinstance(sub_val[0], list):
                        for name in sub_val[0]:
                            if isinstance(name, str) and name.casefold().endswith(MODEL_EXTS) and name not in out["lora_name"]:
                                out["lora_name"].append(name)
        return out

    def _load_cache(self) -> dict | None:
        if not self.cache_file:
            return self._memory_cache
        if not self.cache_file.exists():
            return self._memory_cache
        try:
            stat = self.cache_file.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == self._cache_stamp and self._memory_cache is not None:
                return self._memory_cache
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("resources"), dict):
                self._memory_cache, self._cache_stamp = data, stamp
                return data
            return self._memory_cache
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"[ComfyUIDirect] 读取模型缓存失败: {e}")
            return self._memory_cache


    def _save_cache(self, data: dict) -> None:
        self._memory_cache = data
        if not self.cache_file:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            # 原子写：先写临时文件再 rename，避免并发预热/查询时写坏缓存
            tmp = self.cache_file.with_name(self.cache_file.name + ".tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(self.cache_file)
            stat = self.cache_file.stat()
            self._cache_stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError as e:
            logger.warning(f"[ComfyUIDirect] 写入模型缓存失败: {e}")


    async def _fetch_resources(self) -> dict | None:
        """从 ComfyUI 拉取最新资源清单；任何失败（含空响应/超时）返回 None。

        单独包裹：即使某个端点异常（如 /embeddings 空响应），也不允许把异常
        抛出到工具调用层——宁可回退缓存。
        """
        try:
            obj = await self.get_object_info()
            if not obj:
                return None
            resources = self._extract_resources(obj)
            try:
                embeddings = await self.get_embeddings()
                resources["embeddings"] = embeddings or []
            except Exception as e:
                logger.warning(f"[ComfyUIDirect] 获取 embeddings 失败，忽略: {e}")
                resources["embeddings"] = []
            # LoRA 触发词：读 safetensors 头部元数据，失败不阻塞清单
            try:
                lora_manager_catalog = await self.get_lora_manager_catalog()
                resources["lora_meta"] = await self._fetch_lora_trigger_words(
                    resources.get("lora_name", []),
                    lora_manager_catalog=lora_manager_catalog,
                )
            except Exception as e:
                logger.warning(f"[ComfyUIDirect] 同步 LoRA 触发词失败，忽略: {e}")
                resources["lora_meta"] = {}
            return resources
        except Exception as e:
            logger.error(f"[ComfyUIDirect] 资源同步失败: {e}")
            return None

    async def list_resources(
        self, force_refresh: bool = False
    ) -> tuple[dict[str, list[str]], bool]:
        """返回 (资源清单, 是否来自缓存)。

        未过期且非强制刷新时直接用缓存；过期则重新从 ComfyUI 同步；
        ComfyUI 离线时回退本地缓存，保证清单不因本机关机而丢失。
        """
        previous = self._load_cache()
        resources = await self.generation_resources(force_refresh)
        task = self._metadata_task
        if task is not None and not task.done():
            await asyncio.shield(task)
        cached = self._load_cache()
        if cached and not cached.get("lora_metadata_v4"):
            if self._full_metadata_task is None or self._full_metadata_task.done():
                self._full_metadata_task = asyncio.create_task(self._enrich_catalog(full=True))
            await asyncio.shield(self._full_metadata_task)
        cached = self._load_cache()
        return copy.deepcopy(cached["resources"]) if cached else resources, bool(previous and not force_refresh)


    async def warm_up_cache(self) -> None:
        """插件加载时后台预热资源清单（自动同步）。

        ComfyUI 刚启动时 /object_info 可能尚未就绪（空响应），最多重试 3 次。
        """
        for attempt in range(1, 4):
            resources = await self.generation_resources()
            if resources.get("unet_name") or resources.get("lora_name"):
                return
            if attempt < 3:
                await asyncio.sleep(8)
        logger.warning("[ComfyUIDirect] 预热资源清单失败（3 次尝试后放弃，将按需同步）")


    def start_events(self):
        if not getattr(self, "events_enabled", False) or getattr(self, "_closed", False):
            return None
        if self._events is None:
            from .events import ExecutionEvents
            self._events = ExecutionEvents(self.base_url, self.request_timeout)
        return self._events

    def execution_progress(self, prompt_id: str) -> dict:
        events = getattr(self, "_events", None)
        return dict(events.states.get(prompt_id, {})) if events is not None else {}

    async def wait_for_history(self, prompt_id: str, timeout: float | None = None) -> tuple[dict | None, str | None]:
        budget = self.timeout if timeout is None else max(0.0, timeout)
        deadline = time.monotonic() + budget
        events = self.start_events()
        try:
            while (remaining := deadline - time.monotonic()) > 0:
                if getattr(self, "_closed", False):
                    return None, f"等待中断，插件连接已关闭。任务可能仍在执行。prompt_id: {prompt_id}，可继续查询该任务。"
                state = self.execution_progress(prompt_id)
                if state.get("event") in {"execution_error", "execution_interrupted"}:
                    return {"status": {"status_str": "error", "messages": [[state["event"], state]]}, "outputs": {}}, None
                try:
                    if events is None:
                        entry = await asyncio.wait_for(self.get_history_entry(prompt_id), remaining)
                    else:
                        request = asyncio.create_task(self.get_history_entry(prompt_id))
                        signal = asyncio.create_task(events.wait(prompt_id, remaining))
                        try:
                            done, _ = await asyncio.wait({request, signal}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                            state = self.execution_progress(prompt_id)
                            if state.get("event") in {"execution_error", "execution_interrupted"}:
                                return {"status": {"status_str": "error", "messages": [[state["event"], state]]}, "outputs": {}}, None
                            if not done:
                                break
                            entry = await asyncio.wait_for(request, max(0.001, deadline - time.monotonic()))
                        finally:
                            for task in (request, signal):
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(request, signal, return_exceptions=True)
                except TimeoutError:
                    break
                if entry is not None:
                    return entry, None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                interval = min(0.2 if state.get("event") == "execution_success" else 2.0, remaining)
                if events is not None:
                    await events.wait(prompt_id, interval)
                else:
                    await asyncio.sleep(interval)
        finally:
            if events is not None:
                events.forget(prompt_id)
        return None, f"等待超时（{budget:g}s），任务可能仍在排队或执行。prompt_id: {prompt_id}，可继续查询该任务。"

    async def generation_resources(self, force_refresh: bool = False) -> dict:
        """Fast catalog for generation; full LoRA enrichment never holds this lock."""
        async with self._resource_lock:
            cached = self._load_cache()
            fresh = not force_refresh and cached and (self.cache_ttl > 0 and time.time() - cached.get("fetched_at", 0) < self.cache_ttl)
            if not fresh:
                obj = await self.get_object_info(force_refresh=True)
                if obj:
                    resources = self._extract_resources(obj)
                    previous = (cached or {}).get("resources", {})
                    available = set()
                    for definition in obj.values():
                        if not isinstance(definition, dict):
                            continue
                        inputs = definition.get("input") or {}
                        for section in ("required", "optional"):
                            section_inputs = inputs.get(section) or {}
                            if not isinstance(section_inputs, dict):
                                continue
                            for field, spec in section_inputs.items():
                                if field in RESOURCE_INPUT_TARGETS and isinstance(spec, list) and spec and isinstance(spec[0], list):
                                    available.add(RESOURCE_INPUT_TARGETS[field])
                    # A partial /object_info response cannot erase absent catalogs.
                    # An explicitly present empty enum remains an authoritative removal.
                    for field in resources:
                        if field not in available and field in previous:
                            resources[field] = copy.deepcopy(previous[field])
                    resources["embeddings"] = previous.get("embeddings", [])
                    resources["model_meta"] = {name: info for name, info in previous.get("model_meta", {}).items()
                                               if name in resources.get("unet_name", [])}
                    resources["lora_meta"] = {name: info for name, info in previous.get("lora_meta", {}).items()
                                              if name in resources.get("lora_name", [])}
                    same_loras = set(resources["lora_name"]) == set(previous.get("lora_name") or [])
                    refresh_loras = "lora_name" in available
                    retain_enrichment = same_loras and not (force_refresh and refresh_loras)
                    cached = {
                        **(cached or {}), "fetched_at": time.time(), "resources": resources,
                        "lora_metadata_v3": bool(retain_enrichment and not refresh_loras and (cached or {}).get("lora_metadata_v3")),
                        "lora_metadata_v4": bool(retain_enrichment and (cached or {}).get("lora_metadata_v4")),
                        "lora_catalog_ready": bool(retain_enrichment and not refresh_loras and (cached or {}).get("lora_catalog_ready")),
                    }
                    self._save_cache(cached)
            resources = copy.deepcopy((cached or {}).get("resources", {
                "unet_name": [], "lora_name": [], "clip_name": [], "vae_name": [], "lora_meta": {},
            }))
        self._start_metadata_refresh()
        return resources

    def _start_metadata_refresh(self) -> None:
        cached = self._memory_cache
        if self._closed or not cached or cached.get("lora_metadata_v3") is True or cached.get("lora_catalog_ready"):
            return
        if self._metadata_task is None or self._metadata_task.done():
            self._metadata_task = asyncio.create_task(self._enrich_catalog())

    async def _enrich_catalog(self, full: bool = False) -> None:
        cached = self._load_cache()
        if not cached:
            return
        names = list(cached["resources"].get("lora_name", []))
        try:
            catalog = await self.get_lora_manager_catalog()
            if full:
                metadata = await self._fetch_lora_trigger_words(names, lora_manager_catalog=catalog)
            else:
                metadata = {}
                for name in names:
                    raw = next((catalog[key] for key in self._lora_key_variants(name) if key in catalog), None)
                    if raw:
                        metadata[name] = raw
            embeddings = await self.get_embeddings()
            async with self._resource_lock:
                current = self._load_cache()
                if current and current.get("fetched_at") == cached.get("fetched_at"):
                    updated = copy.deepcopy(current)
                    merged = updated["resources"].setdefault("lora_meta", {})
                    for name, info in metadata.items():
                        if not full:
                            # 按最新缓存补缺字段，保留目录请求期间按需读出的可信元数据。
                            previous = merged.get(name, {})
                            trusted = self.normalize_lora_metadata(info).get("trigger_words") or []
                            info = self.normalize_lora_metadata(info, fallback=previous)
                            info["trusted_trigger_words"] = list(dict.fromkeys(
                                list(previous.get("trusted_trigger_words") or []) + trusted))
                        merged[name] = {**merged.get(name, {}), **info}
                    if embeddings is not None:
                        updated["resources"]["embeddings"] = embeddings
                    updated["lora_metadata_v3"] = full
                    updated["lora_metadata_v4"] = full or bool(updated.get("lora_metadata_v4"))
                    updated["lora_catalog_ready"] = True
                    self._save_cache(updated)
        except Exception as e:
            logger.warning(f"[ComfyUIDirect] 后台资源补全失败: {e}")

    async def selected_lora_metadata(self, resources: dict, names: list[str], require_trusted: bool = False) -> dict:
        """Enrich only selected files for compatibility checks/known triggers."""
        missing = [name for name in names if not (resources.get("lora_meta") or {}).get(name)
                   or (require_trusted and "trusted_trigger_words" not in resources["lora_meta"][name])]
        if missing:
            metadata = await self._fetch_lora_trigger_words(missing)
            resources.setdefault("lora_meta", {}).update(metadata)
            async with self._resource_lock:
                cached = self._load_cache()
                if cached:
                    updated = copy.deepcopy(cached)
                    known_names = updated["resources"].get("lora_name", [])
                    updated["resources"].setdefault("lora_meta", {}).update({name: value for name, value in metadata.items() if name in known_names})
                    self._save_cache(updated)
        return resources

    def start_warmup(self) -> None:
        if not self._closed and (self._warmup_task is None or self._warmup_task.done()):
            self._warmup_task = asyncio.create_task(self.warm_up_cache())
