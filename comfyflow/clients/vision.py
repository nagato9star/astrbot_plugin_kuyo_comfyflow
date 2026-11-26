"""云端视觉模型客户端：支持 AstrBot Provider（含回落）与 OpenAI/DeepSeek 兼容直连 API。"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx
from astrbot.api import logger

from .comfyui import image_media_type

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_TIMEOUT = 45.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 1.0
_MODERATION_ERROR_TOKENS = (
    "content_policy_violation",
    "content_filter",
    "data_inspection_failed",
    "prohibited_content",
    "unsafe content",
    "safety policy",
)
_REFUSAL_TEXT_PREFIXES = (
    "i'm sorry, i can't",
    "i'm sorry, i cannot",
    "i am sorry, i cannot",
    "sorry, i can't",
    "sorry, i cannot",
    "i cannot assist",
    "i can't assist",
    "i cannot fulfill",
    "i can't fulfill",
    "i cannot analyze",
    "i cannot describe",
    "i'm unable to",
    "i am unable to",
    "as an ai",
    "抱歉，我无法",
    "抱歉，我不能",
    "对不起，我无法",
    "我无法协助",
    "我无法描述",
    "我无法分析",
    "我不能协助",
    "无法满足该请求",
)


class VisionRefusalError(RuntimeError):
    """Raised when the vision model or API gateway refuses to process the image."""


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _is_moderation_error(exc: Exception) -> bool:
    parts = [str(exc)]
    resp = getattr(exc, "response", None)
    if resp is not None:
        parts.append(str(getattr(resp, "text", "") or ""))
    combined = " ".join(parts).lower()
    return any(token in combined for token in _MODERATION_ERROR_TOKENS)


def _looks_like_refusal_text(text: str) -> bool:
    norm = " ".join(text.strip().lower().split())
    if not norm or len(norm) > 400:
        return False
    return any(norm.startswith(prefix) for prefix in _REFUSAL_TEXT_PREFIXES)


def _is_retryable_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in RETRYABLE_STATUS_CODES:
        return True
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("429", "rate limit", "ratelimit", "too many requests", "timeout", "timed out")
    )


def _extract_openai_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    choice = choices[0]
    if str(choice.get("finish_reason") or "").strip().lower() == "content_filter":
        raise VisionRefusalError("视觉接口触发 content_filter，拒绝返回分析结果。")
    message = choice.get("message") or {}
    if message.get("refusal"):
        raise VisionRefusalError(f"视觉模型拒绝分析该图片: {message.get('refusal')}")
    content = message.get("content")
    text = ""
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
        text = "\n".join(parts).strip()
    if _looks_like_refusal_text(text):
        raise VisionRefusalError(f"视觉模型拒答: {text[:120]}")
    return text


class VisionInterrogateClient:
    """统一封装提示词反推所需的云端 VLM 调用。"""

    def __init__(
        self,
        astrbot_context: Any = None,
        *,
        provider_id: str = "",
        fallback_provider_ids: list[str] | tuple[str, ...] = (),
        thinking_effort: str = "low",
        system_prompt_override: str = "",
        api_base: str = "",
        api_key: str = "",
        api_model: str = "",
        image_detail: str = "high",
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> None:
        self.astrbot_context = astrbot_context
        self.provider_id = _clean_text(provider_id)
        self.fallback_provider_ids = [
            _clean_text(pid) for pid in (fallback_provider_ids or []) if _clean_text(pid)
        ]
        effort = _clean_text(thinking_effort).lower()
        self.thinking_effort = effort if effort in {"low", "medium", "high"} else ""
        self.system_prompt_override = _clean_text(system_prompt_override)
        self.api_base = _clean_text(api_base).rstrip("/")
        self.api_key = _clean_text(api_key)
        self.api_model = _clean_text(api_model)
        detail = _clean_text(image_detail).lower()
        self.image_detail = detail if detail in {"high", "low", "auto"} else "high"
        self.timeout = max(5.0, float(timeout or DEFAULT_TIMEOUT))
        self.max_retries = max(1, int(max_retries or DEFAULT_MAX_RETRIES))
        self.retry_delay = max(0.2, float(retry_delay or DEFAULT_RETRY_DELAY))
        self._http: httpx.AsyncClient | None = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
            )
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def has_direct_api(self) -> bool:
        return bool(self.api_base and self.api_key and self.api_model)

    def resolve_provider_candidates(self) -> list[str]:
        candidates: list[str] = []
        if self.provider_id:
            candidates.append(self.provider_id)
        else:
            framework_vlm_id = ""
            if self.astrbot_context is not None and hasattr(self.astrbot_context, "get_config"):
                try:
                    cfg = self.astrbot_context.get_config() or {}
                    provider_settings = cfg.get("provider_settings") or {}
                    framework_vlm_id = _clean_text(
                        provider_settings.get("default_image_caption_provider_id")
                    )
                except Exception as exc:
                    logger.debug(f"[ComfyUIDirect] 读取框架默认视觉模型配置失败: {exc}")
            if framework_vlm_id:
                candidates.append(framework_vlm_id)
        for fallback_id in self.fallback_provider_ids:
            if fallback_id not in candidates:
                candidates.append(fallback_id)
        return candidates

    def is_configured(self) -> bool:
        return self.has_direct_api() or bool(self.resolve_provider_candidates())

    def _direct_endpoint(self) -> str:
        if self.api_base.endswith("/chat/completions"):
            return self.api_base
        return f"{self.api_base}/chat/completions"

    async def _call_direct_api(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, str]:
        mime = image_media_type(image_bytes, filename)
        if not mime.startswith("image/"):
            mime = "image/png"
        b64 = base64.b64encode(image_bytes).decode("ascii")
        image_url_obj: dict[str, Any] = {"url": f"data:{mime};base64,{b64}"}
        if self.image_detail:
            image_url_obj["detail"] = self.image_detail
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # DeepSeek / OpenAI 规范要求 image_url 只能放在 user role 的 content 数组中
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": image_url_obj},
                ],
            }
        )
        body: dict[str, Any] = {
            "model": self.api_model,
            "messages": messages,
        }
        if self.thinking_effort:
            body["reasoning_effort"] = self.thinking_effort

        endpoint = self._direct_endpoint()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await self.http.post(endpoint, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                text = _extract_openai_content(data)
                if text:
                    return text, f"direct:{self.api_model}"
                last_error = RuntimeError("直连视觉 API 返回空文本")
            except VisionRefusalError:
                raise
            except Exception as exc:
                if _is_moderation_error(exc):
                    raise VisionRefusalError(f"直连视觉 API 触发内容审核拦截（{self.api_model}）: {exc}") from exc
                last_error = exc
                if not _is_retryable_error(exc) or attempt >= self.max_retries - 1:
                    break
            await asyncio.sleep(self.retry_delay * (2**attempt))
        raise RuntimeError(f"直连视觉 API 调用失败（{self.api_model}）: {last_error}") from last_error

    async def _call_astrbot_provider(
        self,
        provider_ids: list[str],
        *,
        image_ref: str,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, str]:
        if self.astrbot_context is None or not hasattr(self.astrbot_context, "llm_generate"):
            raise RuntimeError("当前运行环境缺少 AstrBot llm_generate 接口。")
        extra_kwargs: dict[str, Any] = {}
        if self.thinking_effort:
            extra_kwargs["reasoning_effort"] = self.thinking_effort

        last_error: Exception | None = None
        for idx, provider_id in enumerate(provider_ids):
            if idx > 0:
                logger.info(f"[ComfyUIDirect] 提示词反推切换回落视觉模型: {provider_id}")
            for attempt in range(self.max_retries):
                try:
                    resp = await self.astrbot_context.llm_generate(
                        chat_provider_id=provider_id,
                        prompt=user_prompt,
                        system_prompt=system_prompt or None,
                        image_urls=[image_ref],
                        **extra_kwargs,
                    )
                    text = _clean_text(getattr(resp, "completion_text", "")) if resp else ""
                    if text:
                        if _looks_like_refusal_text(text):
                            last_error = VisionRefusalError(f"视觉模型 {provider_id} 拒答: {text[:120]}")
                            break
                        return text, f"provider:{provider_id}"
                    last_error = RuntimeError(f"视觉模型 {provider_id} 返回空响应")
                except VisionRefusalError as exc:
                    last_error = exc
                    break
                except Exception as exc:
                    if _is_moderation_error(exc):
                        last_error = VisionRefusalError(f"视觉模型 {provider_id} 触发内容审核拦截: {exc}")
                        break
                    last_error = exc
                    if not _is_retryable_error(exc) or attempt >= self.max_retries - 1:
                        break
                await asyncio.sleep(self.retry_delay * (2**attempt))
        if isinstance(last_error, VisionRefusalError):
            raise last_error
        raise RuntimeError(
            f"AstrBot 视觉模型调用失败（已尝试 {len(provider_ids)} 个候选）: {last_error}"
        ) from last_error

    async def describe_image(
        self,
        *,
        image_path: str,
        image_bytes: bytes,
        filename: str = "image.png",
        system_prompt: str = "",
        user_prompt: str = "",
    ) -> tuple[str, str]:
        """调用已配置的 VLM（直连 OpenAI/DeepSeek API 优先，未配置或失败时回落 AstrBot Provider）。"""
        direct_ready = self.has_direct_api()
        provider_ids = self.resolve_provider_candidates()
        if not direct_ready and not provider_ids:
            raise ValueError(
                "未配置反推视觉模型：请在插件配置中设置「反推视觉模型 (interrogate_provider_id)」"
                "或「AstrBot 全局图片描述模型」，或填写直连 Vision API（interrogate_api_base / key / model）。"
            )

        errors: list[str] = []
        last_exc: Exception | None = None
        if direct_ready:
            try:
                return await self._call_direct_api(
                    image_bytes=image_bytes,
                    filename=filename,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception as exc:
                last_exc = exc
                errors.append(str(exc))
                if not provider_ids:
                    if isinstance(exc, VisionRefusalError):
                        raise
                    raise RuntimeError(str(exc)) from exc
                logger.warning(f"[ComfyUIDirect] 直连视觉 API 失败，尝试回落 AstrBot Provider: {exc}")

        if provider_ids:
            image_ref = image_path
            if not image_ref:
                mime = image_media_type(image_bytes, filename)
                if not mime.startswith("image/"):
                    mime = "image/png"
                image_ref = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
            try:
                return await self._call_astrbot_provider(
                    provider_ids,
                    image_ref=image_ref,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception as exc:
                last_exc = exc
                errors.append(str(exc))

        message = "；".join(errors) or "视觉模型调用失败"
        if isinstance(last_exc, VisionRefusalError):
            raise VisionRefusalError(message) from last_exc
        raise RuntimeError(message) from last_exc
