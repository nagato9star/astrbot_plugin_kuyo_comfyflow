"""One reconnecting ComfyUI event connection; history remains the result authority."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import OrderedDict


class ExecutionEvents:
    def __init__(self, base_url: str, request_timeout: float) -> None:
        self.url = (
            base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
            + "/ws"
        )
        self.client_id = uuid.uuid4().hex
        self.request_timeout = request_timeout
        self.signals: dict[str, asyncio.Event] = {}
        self.states: OrderedDict[str, dict] = OrderedDict()
        self.task = asyncio.create_task(self._listen())

    def handle(self, message: dict) -> None:
        kind, data = message.get("type"), message.get("data")
        if not isinstance(data, dict) or not data.get("prompt_id"):
            return
        pid = str(data["prompt_id"])
        if kind not in {
            "execution_start",
            "progress",
            "executing",
            "execution_success",
            "execution_error",
            "execution_interrupted",
        }:
            return
        self.states[pid] = {"event": kind, **data}
        self.states.move_to_end(pid)
        while len(self.states) > 128:
            self.states.popitem(last=False)
        if kind in {
            "execution_success",
            "execution_error",
            "execution_interrupted",
        } or (kind == "executing" and data.get("node") is None):
            signal = self.signals.get(pid)
            if signal:
                signal.set()

    async def _listen(self) -> None:
        try:
            import aiohttp
        except ImportError:
            return  # A missing optional transport leaves HTTP polling usable.
        delay = 1
        async with aiohttp.ClientSession(trust_env=True) as session:
            while True:
                try:
                    async with asyncio.timeout(min(self.request_timeout, 5)):
                        socket = await session.ws_connect(
                            self.url, params={"clientId": self.client_id}, heartbeat=20
                        )
                    async with socket:
                        delay = 1
                        async for message in socket:
                            if message.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    self.handle(json.loads(message.data))
                                except (ValueError, TypeError, AttributeError):
                                    continue
                            elif message.type in {
                                aiohttp.WSMsgType.ERROR,
                                aiohttp.WSMsgType.CLOSED,
                            }:
                                break
                except (aiohttp.ClientError, OSError, TimeoutError):
                    pass
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    async def wait(self, pid: str, seconds: float) -> None:
        signal = self.signals.setdefault(pid, asyncio.Event())
        # Events which arrived before waiting must still wake the first history check.
        state = self.states.get(pid, {})
        if state.get("event") in {
            "execution_success",
            "execution_error",
            "execution_interrupted",
        } or (state.get("event") == "executing" and state.get("node") is None):
            await asyncio.sleep(min(seconds, 0.2))
            return
        try:
            async with asyncio.timeout(seconds):
                await signal.wait()
        except TimeoutError:
            pass
        finally:
            signal.clear()

    def forget(self, pid: str) -> None:
        self.signals.pop(pid, None)

    async def close(self) -> None:
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        self.signals.clear()
