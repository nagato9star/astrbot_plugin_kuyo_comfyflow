"""AnimaDex 本地角色库客户端（MCP Streamable HTTP）。

连接 33 号 PC 上跑的 animadex-mcp-server（默认 127.0.0.1:11451/mcp），
查询 36,000+ 动漫游戏角色的 Prompt Trigger / 特征标签 / 关联 LoRA，
以及画师、系列（版权）信息。本地 SQLite 缓存，离线可用、免费。

MCP 工具（工具名带连字符）：
- search-characters / get-character / search-artists / search-copyrights / get-character-facets

会话复用：initialize 后保存 mcp-session-id，后续调用直接复用；
会话失效（服务重启/超时）时自动重新 initialize 重试一次。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

DEFAULT_MCP_URL = "http://127.0.0.1:11451/mcp"
DEFAULT_TIMEOUT = 8.0
_CLIENT_INFO = {"name": "comfyui-direct", "version": "2.1.0"}
_PROTOCOL_VERSION = "2024-11-05"


def _parse_sse(text: str) -> list[dict[str, Any]]:
    """解析 MCP Streamable HTTP 响应（text/event-stream 或纯 JSON）。"""
    text = text.strip()
    if not text:
        return []
    if text.startswith("{"):
        return [json.loads(text)]
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                try:
                    out.append(json.loads(payload))
                except json.JSONDecodeError:
                    continue
    return out


def _extract_text(messages: list[dict[str, Any]]) -> str:
    """从 MCP tools/call 结果里拼接文本内容。"""
    parts: list[str] = []
    for msg in messages:
        result = msg.get("result") or {}
        if result.get("isError"):
            continue
        for item in result.get("content") or []:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
    return "\n".join(parts).strip()


class AnimaDexClient:
    """AnimaDex MCP Streamable HTTP 客户端（带会话复用）。"""

    def __init__(
        self,
        mcp_url: str = DEFAULT_MCP_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._mcp_url = mcp_url
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=3.0),
                follow_redirects=True,
            )
        return self._client

    async def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            h["mcp-session-id"] = self._session_id
        return h

    async def _initialize(self) -> None:
        client = await self._ensure_client()
        resp = await client.post(
            self._mcp_url,
            headers=await self._headers(),
            json={
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": _CLIENT_INFO,
                },
            },
        )
        resp.raise_for_status()
        sid = resp.headers.get("mcp-session-id") or resp.headers.get("Mcp-Session-Id")
        if sid:
            self._session_id = sid
        # notifications/initialized 告知服务端会话就绪
        try:
            await client.post(
                self._mcp_url,
                headers=await self._headers(),
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
        except httpx.HTTPError:
            pass

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """发送一次 JSON-RPC 请求；会话失效时自动重连重试一次。"""
        for attempt in range(2):
            if self._session_id is None:
                await self._initialize()
            client = await self._ensure_client()
            body: dict[str, Any] = {"jsonrpc": "2.0", "id": "req", "method": method}
            if params is not None:
                body["params"] = params
            try:
                resp = await client.post(
                    self._mcp_url,
                    headers=await self._headers(),
                    json=body,
                )
                resp.raise_for_status()
                messages = _parse_sse(resp.text)
                # 会话失效：服务端返回 -32600 / 401 等错误码
                if messages and any(
                    m.get("error", {}).get("code") in (-32600, -32001)
                    for m in messages
                ):
                    self._session_id = None
                    if attempt == 0:
                        continue
                return messages
            except httpx.HTTPError:
                self._session_id = None
                if attempt == 0:
                    continue
                raise
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """调用 MCP 工具，返回文本结果。失败返回空串。"""
        try:
            async with self._lock:
                messages = await self._request("tools/call", {"name": name, "arguments": arguments or {}})
            text = _extract_text(messages)
            if not text:
                # 尝试拿 structuredContent
                for msg in messages:
                    result = msg.get("result") or {}
                    if result.get("isError"):
                        continue
                    sc = result.get("structuredContent")
                    if isinstance(sc, dict) and sc.get("result"):
                        return str(sc["result"])
                return ""
            return text
        except httpx.HTTPError:
            return ""

    # ── 业务方法 ──────────────────────────────────────────────

    async def search_characters(
        self, query: str, page: int = 1, sort: str = "count"
    ) -> str:
        """搜索角色，返回名称/触发词/标签。"""
        return await self.call_tool(
            "search-characters",
            {"query": query, "page": max(int(page or 1), 1), "sort": sort or "count"},
        )

    async def get_character(self, slug: str) -> str:
        """获取单个角色详情（需 slug，如 hatsune_miku）。"""
        return await self.call_tool("get-character", {"slug": str(slug or "").strip()})

    async def search_artists(self, query: str, page: int = 1) -> str:
        """搜索画师，返回名称和触发词。"""
        return await self.call_tool(
            "search-artists", {"query": query, "page": max(int(page or 1), 1)}
        )

    async def search_copyrights(self, query: str, page: int = 1) -> str:
        """搜索系列/版权。"""
        return await self.call_tool(
            "search-copyrights", {"query": query, "page": max(int(page or 1), 1)}
        )

    async def get_character_facets(self) -> str:
        """获取热门角色/系列筛选条件。"""
        return await self.call_tool("get-character-facets", {})

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
