"""图片提示词反推服务：PNG/A1111/ComfyUI 元数据直读 + 云端 VLM 反推。"""

from __future__ import annotations

import json
import re
import struct
import zlib
from pathlib import Path
from typing import Any

from ..clients.vision import VisionInterrogateClient
from ..workflows.slots import detect_slots

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_A1111_PARAM_RE = re.compile(
    r"(?:^|,\s*)(Steps|Sampler|Schedule type|CFG scale|Seed|Size|Model|Model hash|Clip skip|Denoising strength):\s*([^,\n]+)"
)
_PROMPT_NODE_CLASSES = {
    "CLIPTextEncode",
    "CLIPTextEncodeSDXL",
    "TextEncodeQwenImageEdit",
    "TextEncodeQwenImageEditPlus",
    "TextEncodeQwenImage21",
    "CR Prompt Text",
    "DanbooruText",
    "String Literal",
    "PrimitiveStringMultiline",
}

_DANBOORU_SYSTEM = (
    "You are an automated computer-vision art-cataloging and Danbooru taxonomy indexer.\n"
    "Your task is purely technical visual annotation for digital illustration organization and style analysis.\n"
    "Analyze the visual composition and output ONLY concise English Danbooru-style comma-separated tags using neutral, "
    "clinical/art-historical and apparel-taxonomy terminology.\n"
    "Order tags roughly as: subject count & archetype (e.g. 1girl, solo), hair/eyes/features, "
    "garment cuts/accessories/footwear, posture/gesture/expression, framing/camera angle (e.g. upper body, dutch angle), "
    "setting/background, lighting/color palette, artistic medium/rendering style.\n"
    "Rules:\n"
    "- Output pure comma-separated tags in English only (no numbering, no bullet points, no Chinese, no markdown fences, no commentary).\n"
    "- Use objective, matter-of-fact visual taxonomy terms; avoid subjective, sensational, or conversational wording.\n"
    "- Avoid filler quality tags like masterpiece or best quality unless explicitly requested.\n"
    "- Record only clearly visible visual elements without inventing unseen details."
)

_NATURAL_SYSTEM = (
    "You are an automated visual-composition and art-direction analyzer for digital asset cataloging.\n"
    "Produce a clear, objective, high-signal English visual specification suitable for image-generation conditioning.\n"
    "Cover: artistic medium and rendering technique, subject appearance, garment design/materials, posture and framing, "
    "camera perspective, environment details, lighting, and color palette.\n"
    "Rules:\n"
    "- Output ONLY the objective visual description paragraph in English (no preamble, no bullet list, no markdown fences, no meta commentary).\n"
    "- Use neutral, technical art, photography, and fashion terminology; keep the tone strictly analytical and matter-of-fact.\n"
    "- Stay faithful to visible composition and styling details."
)

_AUTO_SYSTEM = (
    "You are an automated computer-vision visual indexer producing both structured descriptive captions "
    "and standardized Danbooru taxonomy tags for art and design workflows.\n"
    "Keep wording strictly neutral, objective, and technical (art medium, apparel design, posture, framing, lighting, setting).\n"
    "Output EXACTLY two labeled sections in English, with no extra commentary:\n"
    "[Natural Prompt]\n"
    "<One concise, objective English descriptive paragraph covering medium/style, subject, outfit, posture, framing, lighting, and setting>\n\n"
    "[Danbooru Tags]\n"
    "<Comma-separated English Danbooru tags ordered from subject count, appearance, garment taxonomy, posture, framing, background, lighting, to style>"
)


def _decode_png_text_chunks(content: bytes) -> dict[str, str]:
    """Extract PNG tEXt, zTXt, and iTXt key/value pairs without external dependencies."""
    if not content.startswith(_PNG_SIGNATURE):
        return {}
    chunks: dict[str, str] = {}
    offset = len(_PNG_SIGNATURE)
    total = len(content)
    while offset + 12 <= total:
        length = struct.unpack_from(">I", content, offset)[0]
        chunk_type = content[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        if data_end + 4 > total:
            break
        raw = content[data_start:data_end]
        offset = data_end + 4

        if chunk_type == b"tEXt":
            sep = raw.find(b"\x00")
            if sep > 0:
                key = raw[:sep].decode("latin-1", errors="replace")
                val = raw[sep + 1 :].decode("utf-8", errors="replace")
                chunks[key] = val
        elif chunk_type == b"zTXt":
            sep = raw.find(b"\x00")
            if sep > 0 and sep + 2 <= len(raw) and raw[sep + 1] == 0:
                key = raw[:sep].decode("latin-1", errors="replace")
                try:
                    val = zlib.decompress(raw[sep + 2 :]).decode("utf-8", errors="replace")
                    chunks[key] = val
                except zlib.error:
                    pass
        elif chunk_type == b"iTXt":
            sep = raw.find(b"\x00")
            if sep > 0 and sep + 3 <= len(raw):
                key = raw[:sep].decode("latin-1", errors="replace")
                comp_flag = raw[sep + 1]
                comp_method = raw[sep + 2]
                rest = raw[sep + 3 :]
                lang_sep = rest.find(b"\x00")
                if lang_sep >= 0:
                    rest = rest[lang_sep + 1 :]
                    trans_sep = rest.find(b"\x00")
                    if trans_sep >= 0:
                        text_bytes = rest[trans_sep + 1 :]
                        if comp_flag == 1 and comp_method == 0:
                            try:
                                text_bytes = zlib.decompress(text_bytes)
                            except zlib.error:
                                continue
                        chunks[key] = text_bytes.decode("utf-8", errors="replace")
        elif chunk_type == b"IEND":
            break
    return chunks


def _extract_node_text(node: dict[str, Any]) -> str:
    inputs = node.get("inputs") if isinstance(node, dict) else None
    if not isinstance(inputs, dict):
        return ""
    for field in ("text", "prompt", "value", "string", "positive"):
        val = inputs.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _resolve_text_input(wf: dict[str, Any], value: Any, seen: set[str] | None = None) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and len(value) == 2:
        node_id = str(value[0])
        if seen is None:
            seen = set()
        if node_id in seen:
            return ""
        seen.add(node_id)
        node = wf.get(node_id)
        if not isinstance(node, dict):
            return ""
        direct = _extract_node_text(node)
        if direct:
            return direct
        inputs = node.get("inputs") or {}
        for key in ("text", "prompt", "string", "text_a", "text_b"):
            if key in inputs:
                resolved = _resolve_text_input(wf, inputs[key], seen)
                if resolved:
                    return resolved
    return ""


def _parse_comfyui_prompt_graph(raw_prompt: str) -> dict[str, Any] | None:
    try:
        wf = json.loads(raw_prompt)
    except (TypeError, ValueError):
        return None
    if not isinstance(wf, dict) or not wf:
        return None

    slots = detect_slots(wf)
    positive_parts: list[str] = []
    negative_parts: list[str] = []

    for role in ("artist", "quality", "prompt", "trigger_words"):
        spec = slots.get(role) or {}
        node_id = str(spec.get("node") or "")
        field = str(spec.get("field") or "")
        node = wf.get(node_id)
        if isinstance(node, dict):
            inputs = node.get("inputs") or {}
            txt = _resolve_text_input(wf, inputs.get(field)) if field in inputs else _extract_node_text(node)
            if txt and txt not in positive_parts:
                positive_parts.append(txt)

    neg_spec = slots.get("negative") or {}
    neg_node = wf.get(str(neg_spec.get("node") or ""))
    if isinstance(neg_node, dict):
        neg_field = str(neg_spec.get("field") or "text")
        neg_inputs = neg_node.get("inputs") or {}
        neg_txt = (
            _resolve_text_input(wf, neg_inputs.get(neg_field))
            if neg_field in neg_inputs
            else _extract_node_text(neg_node)
        )
        if neg_txt:
            negative_parts.append(neg_txt)

    # Fallback: if slot detection missed prompts on custom graphs, collect non-empty prompt nodes
    if not positive_parts:
        for node in wf.values():
            if not isinstance(node, dict):
                continue
            if str(node.get("class_type") or "") in _PROMPT_NODE_CLASSES:
                txt = _extract_node_text(node)
                if txt and txt not in positive_parts and txt not in negative_parts:
                    positive_parts.append(txt)

    params: dict[str, Any] = {}
    model_spec = slots.get("model") or {}
    model_node = wf.get(str(model_spec.get("node") or ""))
    if isinstance(model_node, dict):
        model_field = str(model_spec.get("field") or "")
        model_val = (model_node.get("inputs") or {}).get(model_field)
        if isinstance(model_val, str) and model_val.strip():
            params["model"] = model_val.strip()

    sampler_spec = slots.get("sampler") or {}
    sampler_node = wf.get(str(sampler_spec.get("node") or ""))
    if isinstance(sampler_node, dict):
        s_inputs = sampler_node.get("inputs") or {}
        for key in ("steps", "cfg", "sampler_name", "scheduler", "seed", "noise_seed", "denoise"):
            val = s_inputs.get(key)
            if isinstance(val, (int, float, str)) and val != "":
                params[key] = val

    loras: list[str] = []
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type") or "")
        inputs = node.get("inputs") or {}
        if cls in {"LoraLoader", "LoraLoaderModelOnly", "LoraLoaderModelOnly (rgthree)"}:
            name = inputs.get("lora_name")
            if isinstance(name, str) and name.strip():
                loras.append(name.strip())
        elif cls == "Power Lora Loader (rgthree)":
            for key, val in inputs.items():
                if key.startswith("lora_") and isinstance(val, dict) and val.get("on") and val.get("lora"):
                    loras.append(str(val["lora"]))
    if loras:
        params["loras"] = loras

    if not positive_parts and not negative_parts and not params:
        return None
    return {
        "source": "metadata",
        "format": "comfyui",
        "prompt": ", ".join(positive_parts),
        "negative_prompt": ", ".join(negative_parts),
        "params": params,
    }


def _parse_a1111_parameters(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None
    lines = text.splitlines()
    neg_idx = next((i for i, line in enumerate(lines) if line.startswith("Negative prompt:")), None)
    param_idx = next(
        (i for i, line in enumerate(lines) if "Steps:" in line and ("Sampler:" in line or "CFG scale:" in line)),
        None,
    )
    end_pos = neg_idx if neg_idx is not None else (param_idx if param_idx is not None else len(lines))
    prompt = "\n".join(lines[:end_pos]).strip()
    negative = ""
    if neg_idx is not None:
        end_neg = param_idx if param_idx is not None and param_idx > neg_idx else len(lines)
        neg_lines = [lines[neg_idx][len("Negative prompt:") :].strip(), *lines[neg_idx + 1 : end_neg]]
        negative = "\n".join(line for line in neg_lines if line).strip()
    params: dict[str, Any] = {}
    if param_idx is not None:
        param_line = "\n".join(lines[param_idx:])
        for match in _A1111_PARAM_RE.finditer(param_line):
            params[match.group(1)] = match.group(2).strip()
    if not prompt and not negative and not params:
        return None
    return {
        "source": "metadata",
        "format": "a1111",
        "prompt": prompt,
        "negative_prompt": negative,
        "params": params,
    }


def extract_image_metadata(content: bytes) -> dict[str, Any] | None:
    """从 PNG/内嵌块读取 ComfyUI、A1111 或 NovelAI 的生成提示词元数据。"""
    chunks = _decode_png_text_chunks(content)
    if not chunks:
        return None

    if "prompt" in chunks:
        parsed = _parse_comfyui_prompt_graph(chunks["prompt"])
        if parsed is not None:
            return parsed

    if "parameters" in chunks:
        parsed = _parse_a1111_parameters(chunks["parameters"])
        if parsed is not None:
            return parsed

    # NovelAI / SD metadata fallback
    if chunks.get("Software") == "NovelAI" or "Comment" in chunks:
        comment_raw = chunks.get("Comment") or ""
        prompt = (chunks.get("Description") or "").strip()
        negative = ""
        params: dict[str, Any] = {}
        if comment_raw:
            try:
                comment = json.loads(comment_raw)
                if isinstance(comment, dict):
                    prompt = str(comment.get("prompt") or prompt).strip()
                    negative = str(comment.get("uc") or "").strip()
                    for key in ("steps", "scale", "sampler", "seed"):
                        if key in comment:
                            params[key] = comment[key]
            except (TypeError, ValueError):
                pass
        if prompt or negative:
            return {
                "source": "metadata",
                "format": "novelai",
                "prompt": prompt,
                "negative_prompt": negative,
                "params": params,
            }

    return None


def resolve_interrogate_style(style: str, model_family: str = "", families: Any = None) -> str:
    norm = str(style or "auto").strip().casefold()
    if norm in {"danbooru", "natural"}:
        return norm
    if model_family and families is not None and hasattr(families, "get"):
        family_obj = families.get(model_family)
        if family_obj is not None:
            family_style = str(getattr(family_obj, "prompt_style", "") or "").strip().casefold()
            if family_style in {"danbooru", "natural"}:
                return family_style
    return "auto"


def build_vision_prompts(
    style: str,
    focus: str = "",
    custom_system_prompt: str = "",
) -> tuple[str, str]:
    if style == "danbooru":
        system_prompt = _DANBOORU_SYSTEM
        user_prompt = "Index this artwork with standardized English Danbooru visual taxonomy tags."
    elif style == "natural":
        system_prompt = _NATURAL_SYSTEM
        user_prompt = "Produce an objective English visual-composition prompt for this artwork."
    else:
        system_prompt = _AUTO_SYSTEM
        user_prompt = "Produce both an objective English visual-composition prompt and standardized Danbooru tags for this artwork."
    override = str(custom_system_prompt or "").strip()
    if override:
        system_prompt = override
    focus_text = str(focus or "").strip()
    if focus_text:
        user_prompt += f"\nFocus specifically on: {focus_text}"
    return system_prompt, user_prompt


async def reverse_prompt(
    *,
    image_path: str | Path,
    image_bytes: bytes,
    style: str = "auto",
    model_family: str = "",
    focus: str = "",
    read_metadata: bool = True,
    families: Any = None,
    vision: VisionInterrogateClient | None = None,
) -> dict[str, Any]:
    """先尝试读取图片内嵌生成元数据；未命中则走云端 VLM 反推提示词。"""
    resolved_style = resolve_interrogate_style(style, model_family, families)
    if read_metadata:
        meta = extract_image_metadata(image_bytes)
        if meta is not None and (meta.get("prompt") or meta.get("negative_prompt")):
            return {
                **meta,
                "style": resolved_style,
            }

    if vision is None or not vision.is_configured():
        raise ValueError(
            "图片不包含可识别的生成元数据，且尚未配置反推视觉模型。"
            "请在插件配置中设置「反推视觉模型」或填写直连 Vision API。"
        )

    custom_override = getattr(vision, "system_prompt_override", "") if vision is not None else ""
    system_prompt, user_prompt = build_vision_prompts(
        resolved_style,
        focus=focus,
        custom_system_prompt=custom_override,
    )
    path_str = str(Path(image_path).resolve()) if image_path else ""
    text, backend = await vision.describe_image(
        image_path=path_str,
        image_bytes=image_bytes,
        filename=Path(path_str).name if path_str else "image.png",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return {
        "source": "vision",
        "backend": backend,
        "style": resolved_style,
        "prompt": text,
        "negative_prompt": "",
        "params": {},
    }


def format_reverse_prompt_result(result: dict[str, Any]) -> str:
    """将反推结果格式化为供主 Agent 使用的结构化文本。"""
    source = result.get("source")
    if source == "metadata":
        fmt = result.get("format") or "unknown"
        lines = [f"【提示词反推：图片内嵌元数据 ({fmt})】"]
        if result.get("prompt"):
            lines.append(f"正向提示词：{result['prompt']}")
        if result.get("negative_prompt"):
            lines.append(f"负向提示词：{result['negative_prompt']}")
        params = result.get("params") or {}
        if params:
            summary_items = []
            for key, label in (
                ("model", "底模"),
                ("Model", "底模"),
                ("loras", "LoRA"),
                ("steps", "步数"),
                ("Steps", "步数"),
                ("cfg", "CFG"),
                ("CFG scale", "CFG"),
                ("sampler_name", "采样器"),
                ("Sampler", "采样器"),
                ("scheduler", "调度器"),
                ("seed", "种子"),
                ("Seed", "种子"),
                ("Size", "尺寸"),
            ):
                val = params.get(key)
                if val not in (None, "", []):
                    formatted = ", ".join(str(x) for x in val) if isinstance(val, list) else str(val)
                    summary_items.append(f"{label}={formatted}")
            if summary_items:
                lines.append("生成参数：" + "；".join(summary_items))
        return "\n".join(lines)

    style = result.get("style") or "auto"
    backend = result.get("backend") or "vlm"
    lines = [f"【提示词反推：视觉模型 ({style} | {backend})】", str(result.get("prompt") or "").strip()]
    if style in {"danbooru", "auto"}:
        lines.append("提示：若画面包含特定角色/画师或需匹配本地 LoRA 触发词，可继续调用 comfyui_lookup 核对规范词。")
    return "\n".join(line for line in lines if line)
