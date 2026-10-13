"""ComfyUI 设备、节点、工作流诊断及文件上传。"""

from __future__ import annotations

import json
from pathlib import Path
from astrbot.api import FunctionTool
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from pydantic import ConfigDict
from pydantic import Field
from pydantic.dataclasses import dataclass
from ..clients.comfyui import ComfyUIClient
from .common import MAX_LLM_LIST_ITEMS, _as_bool


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiSystemStatsTool(FunctionTool[AstrAgentContext]):
    """查询 ComfyUI 系统/显存状态（等价 MCP system_stats）。"""

    name: str = "comfyui_system_stats"
    description: str = (
        "查询 ComfyUI 设备（GPU）、显存、系统内存及全局后端队列状态。"
        "用户想知道显存占用、能不能跑大图、为什么变慢时使用。"
    )
    parameters: dict = Field(default_factory=lambda: {
        "type": "object",
        "properties": {"include_queue": {
            "type": "boolean", "description": "是否附带整个 ComfyUI 后端的队列状态，默认 true",
        }},
    })
    client: ComfyUIClient | None = None

    @staticmethod
    def _gb(n: int | float | None) -> str:
        try:
            return f"{n / (1024 ** 3):.1f}GB"
        except (TypeError, ValueError):
            return "?"

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        stats = await self.client.get_system_stats()
        parts = ["【ComfyUI 系统状态】"]
        if stats is None:
            parts.append("系统状态获取失败。")
        stats = stats or {}
        sysinfo = stats.get("system", {})
        if sysinfo:
            if sysinfo.get("comfyui_version"):
                parts.append("ComfyUI 版本: " + str(sysinfo["comfyui_version"]))
            os_ = sysinfo.get("os", "")
            parts.append(f"系统: {os_}")
            ram_total = sysinfo.get("ram_total")
            ram_free = sysinfo.get("ram_free")
            if ram_total:
                parts.append(f"内存: 总 {self._gb(ram_total)} / 空闲 {self._gb(ram_free)}")
        for dev in stats.get("devices", []):
            name = dev.get("name", "?")
            parts.append(f"设备: {name}")
            vr = dev.get("vram_total")
            vf = dev.get("vram_free")
            if vr:
                parts.append(f"显存: 总 {self._gb(vr)} / 空闲 {self._gb(vf)}")
        if _as_bool(kwargs.get("include_queue", True), default=True):
            queue = await self.client.get_queue()
            if queue is None:
                parts.append("全局队列获取失败。")
            else:
                running = [str(item[1]) for item in queue.get("queue_running") or []
                           if isinstance(item, (list, tuple)) and len(item) > 1]
                pending = [str(item[1]) for item in queue.get("queue_pending") or []
                           if isinstance(item, (list, tuple)) and len(item) > 1]
                parts.append(f"【ComfyUI 全局队列】运行中 {len(running)} 个，待执行 {len(pending)} 个")
        return "\n".join(parts)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiFreeMemoryTool(FunctionTool[AstrAgentContext]):
    """释放 ComfyUI 显存（等价 MCP free_memory）。"""

    name: str = "comfyui_free_memory"
    description: str = (
        "请求 ComfyUI 卸载模型/清空执行器缓存以释放显存。"
        "显存不足跑不动、或想腾出显存跑大图时使用。"
        "不影响正在运行的任务（下次队列迭代时生效）。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "unload_models": {
                    "type": "boolean",
                    "description": "是否卸载模型（默认 true）",
                },
            },
        }
    )
    client: ComfyUIClient | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        unload = _as_bool(kwargs.get("unload_models", True), default=True)
        ok = await self.client.free_memory(unload_models=unload, free_cache=True)
        return "已请求释放显存（卸载模型+清缓存）。" if ok else "释放失败：无法连接 ComfyUI。"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiNodesTool(FunctionTool[AstrAgentContext]):
    """查询 ComfyUI 节点类信息（等价 MCP nodes）。"""

    name: str = "comfyui_nodes"
    description: str = (
        "查询 ComfyUI 节点类（class_type）信息：action=search 按关键词搜节点类，"
        "action=get 查某个节点类的输入/输出 schema，action=list 列出全部节点类。"
        "写自定义工作流、确认某个节点类是否存在/参数名时使用。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "get", "list"],
                    "description": "search=搜索（默认），get=查单个类 schema，list=全部节点类",
                },
                "query": {
                    "type": "string",
                    "description": "搜索关键词（action=search 时必填）或节点类名（action=get 时必填）",
                },
            },
        }
    )
    client: ComfyUIClient | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        action = str(kwargs.get("action") or "search").strip().lower()
        query = str(kwargs.get("query") or "").strip()
        obj = await self.client.get_object_info()
        if not obj:
            return "查询失败：无法连接 ComfyUI。"
        if action == "list":
            names = sorted(obj.keys())
            shown = names[:MAX_LLM_LIST_ITEMS]
            suffix = f"\n（共 {len(names)} 个，仅显示前 {MAX_LLM_LIST_ITEMS} 个；用 search 精确查找）" if len(names) > len(shown) else ""
            return "全部节点类 (" + str(len(names)) + "):\n" + ", ".join(shown) + suffix
        if action == "get":
            if not query:
                return "查询失败：query（节点类名）不能为空。"
            info = obj.get(query)
            if not info:
                return f"节点类不存在: {query}"
            inp = info.get("input", {})
            req = inp.get("required", {})
            opt = inp.get("optional", {})
            parts = [f"【节点类 {query}】"]
            if req:
                parts.append("必填输入:")
                for k, v in req.items():
                    types = v[0] if isinstance(v, list) and v else "?"
                    parts.append(f"  {k}: {types}")
            if opt:
                parts.append("可选输入:")
                for k, v in opt.items():
                    types = v[0] if isinstance(v, list) and v else "?"
                    parts.append(f"  {k}: {types}")
            return "\n".join(parts)
        if not query:
            return "查询失败：query（关键词）不能为空。"
        hits = [k for k in obj if query.lower() in k.lower()]
        if not hits:
            return f"未找到包含 '{query}' 的节点类。"
        return "匹配节点类:\n" + "\n".join(sorted(hits)[:50])


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiValidateWorkflowTool(FunctionTool[AstrAgentContext]):
    """校验工作流 JSON 的节点类是否存在于 ComfyUI（等价 MCP validate_workflow 本地版）。"""

    name: str = "comfyui_validate_workflow"
    description: str = (
        "提交前校验一个工作流 JSON：检查节点 class_type 是否都存在、"
        "必填输入是否齐全。返回 valid 与错误列表。"
        "写自定义工作流、担心提交 400 时先用它检查。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workflow": {
                    "type": "string",
                    "description": "工作流 JSON：文件路径或 JSON 字符串（必填）",
                },
            },
            "required": ["workflow"],
        }
    )
    client: ComfyUIClient | None = None

    @staticmethod
    def _load_wf(raw: str) -> dict | None:
        raw = (raw or "").strip()
        if not raw:
            return None
        # 先解析内联 JSON，避免把长 JSON 当作路径探测而触发 ENAMETOOLONG。
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as inline_error:
            try:
                p = Path(raw).expanduser()
                if not p.is_file():
                    raise ValueError(
                        f"workflow 路径与 JSON 格式均无效: {inline_error}"
                    ) from inline_error
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, RuntimeError, json.JSONDecodeError) as e:
                raise ValueError(f"读取工作流文件失败: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("workflow JSON 必须是对象（节点 id -> 节点）")
        return data

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        raw = str(kwargs.get("workflow") or "").strip()
        if not raw:
            return "校验失败：workflow 不能为空。"
        try:
            wf = self._load_wf(raw)
        except ValueError as e:
            return f"校验失败：{e}"
        if wf is None:
            return "校验失败：workflow 解析失败。"
        obj = await self.client.get_object_info()
        if not obj:
            return "校验失败：无法连接 ComfyUI。"
        errors = []
        for nid, node in wf.items():
            ct = node.get("class_type")
            if not ct:
                errors.append(f"节点 {nid}: 缺少 class_type")
                continue
            info = obj.get(ct)
            if not info:
                errors.append(f"节点 {nid}: 节点类不存在 {ct}")
                continue
            req = info.get("input", {}).get("required", {})
            ins = node.get("inputs", {})
            for k in req:
                if k not in ins:
                    errors.append(f"节点 {nid} ({ct}): 缺少必填输入 {k}")
        if errors:
            return "校验结果: invalid\n" + "\n".join(errors[:20])
        return "校验结果: valid（全部节点类与必填输入均通过）"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComfyuiUploadFileTool(FunctionTool[AstrAgentContext]):
    """上传图片到 ComfyUI input 目录（等价 MCP upload_file，图生图素材）。"""

    name: str = "comfyui_upload_file"
    description: str = (
        "把本地图片上传到 ComfyUI 的 input 目录，返回文件名，供图生图/ControlNet 工作流引用。"
        "已有本地图片文件、需要作为生成素材时使用。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "本地图片文件路径（必填）",
                },
            },
            "required": ["path"],
        }
    )
    client: ComfyUIClient | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        path = str(kwargs.get("path") or "").strip()
        if not path:
            return "上传失败：path 不能为空。"
        p = Path(path).expanduser()
        if not p.is_file():
            return f"上传失败：文件不存在 {p}"
        try:
            content = p.read_bytes()
        except OSError as e:
            return f"上传失败：读取文件出错（{e}）"
        name, err = await self.client.upload_image(p.name, content)
        if err:
            return f"上传失败：{err}"
        return f"上传成功，ComfyUI 文件名: {name}"
