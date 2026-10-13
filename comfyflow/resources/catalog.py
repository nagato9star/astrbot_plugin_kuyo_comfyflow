"""Shared family classification and bounded resource discovery; no network requests."""

from __future__ import annotations

import fnmatch
import re
from collections import Counter
from difflib import SequenceMatcher


def canonical_family(value: str) -> str:
    text = str(value or "").strip().casefold()
    aliases = {"krea": "krea2", "krea 2": "krea2", "sd_xl": "sdxl",
               "sd xl": "sdxl", "sdxl 1.0": "sdxl", "sd 1.5": "sd15"}
    return aliases.get(text, text)


def infer_family(value: str) -> str:
    text = str(value or "").casefold().replace("\\", "/")
    # Specific architectures come first; FLUX.1 Krea remains a Flux model.
    patterns = (
        (r"krea[ _.-]*2|^krea$", "krea2"), (r"flux", "flux"),
        (r"anima(?![a-z])", "anima"), (r"illustrious|noobai", "illustrious"),
        (r"pony", "pony"), (r"sd[ _.-]*xl|stable.diffusion.xl", "sdxl"),
        (r"qwen", "qwen"), (r"chroma", "chroma"),
        (r"sd[ _.-]*1[._-]?5|stable.diffusion.v1|v1-5", "sd15"),
        (r"sd[ _.-]*3", "sd3"),
        (r"(?:^|[/_. -])wan(?:[0-9/_. -]|$)", "wan"),
        (r"hunyuan", "hunyuan"), (r"hidream", "hidream"), (r"ltx", "ltxv"),
    )
    for pattern, family in patterns:
        if re.search(pattern, text):
            return family
    return ""


def resource_family(name: str, metadata: dict | None = None, rules=None,
                    kind: str = "lora") -> tuple[str, str]:
    """Return family + evidence. Rules override metadata, then path/name hints."""
    normalized = str(name).replace("\\", "/").casefold()
    for rule in rules or []:
        if not isinstance(rule, dict) or rule.get("kind", "all") not in {kind, "all"}:
            continue
        pattern = str(rule.get("pattern") or "").replace("\\", "/").casefold()
        family = canonical_family(rule.get("family"))
        if pattern and family and fnmatch.fnmatchcase(normalized, pattern):
            return family, "配置"
    metadata = metadata or {}
    if metadata.get("model_family"):
        return canonical_family(metadata["model_family"]), "元数据"
    for key in ("base_model", "baseModel", "ss_base_model_version", "modelspec.architecture"):
        if metadata.get(key):
            family = infer_family(str(metadata[key]))
            # Unrecognized explicit metadata must not be overwritten by a guess.
            return family, "元数据" if family else "未知元数据"
    return infer_family(name), "目录/文件名推断"


def metadata_for(resources: dict, kind: str) -> dict:
    return resources.get("lora_meta" if kind == "lora" else "model_meta") or {}


def family_summary(names: list[str], metadata: dict, rules=None, kind="lora") -> str:
    counts = Counter(resource_family(n, metadata.get(n), rules, kind)[0] or "unknown" for n in names)
    return "，".join(f"{f}={count}" for f, count in sorted(counts.items())) or "无"


def filter_family(names: list[str], metadata: dict, family: str, rules=None,
                  kind="lora", include_unknown=False) -> list[str]:
    target = canonical_family(family)
    result = []
    for name in sorted(set(names), key=str.casefold):
        found, _ = resource_family(name, metadata.get(name), rules, kind)
        if found == target or (not found and (include_unknown or target == "unknown")):
            result.append(name)
    return result


def exact_matches(names: list[str], query: str) -> list[str]:
    q = str(query).replace("\\", "/").casefold()
    full = [n for n in names if n.replace("\\", "/").casefold() == q]
    if full:
        return full
    return [n for n in names if n.replace("\\", "/").rsplit("/", 1)[-1].casefold() == q]


def _name_key(value: str) -> str:
    text = str(value).replace("\\", "/").casefold()
    text = re.sub(r"\.(?:safetensors|ckpt|pt|pth|bin)$", "", text)
    return re.sub(r"[\W_]+", "", text)


def explicitly_named(name: str, query: str) -> bool:
    """Allow an unclassified file only when its full name or stem was named."""
    key = _name_key(query)
    basename = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    return bool(key) and key in {_name_key(name), _name_key(basename)}


def _near_name_score(query: str, name: str) -> float:
    """Compare a possibly abbreviated query with the closest part of a filename."""
    if len(query) < 5:
        return 0.0
    base = _name_key(name.replace("\\", "/").rsplit("/", 1)[-1])
    if not base:
        return 0.0
    matcher = SequenceMatcher(None, query, base)
    block = max(matcher.get_matching_blocks(), key=lambda item: item.size)
    if block.size < max(3, len(query) // 2):
        return 0.0
    center = max(0, block.b - block.a)
    scores = [matcher.ratio()]
    for start in range(max(0, center - 2), min(len(base), center + 3)):
        for length in range(max(1, len(query) - 2), len(query) + 3):
            scores.append(SequenceMatcher(None, query, base[start:start + length]).ratio())
    return max(scores)


def filename_matches(names: list[str], query: str) -> list[tuple[str, str, float]]:
    """Rank exact, normalized-substring, then conservative approximate filename hits."""
    names = sorted(set(names), key=str.casefold)
    exact = exact_matches(names, query)
    if exact:
        return [(name, "exact", 1.0) for name in exact]
    normalized_query = _name_key(query)
    if not normalized_query:
        return []
    raw_query = str(query).replace("\\", "/").casefold()
    contained = [name for name in names if raw_query in name.replace("\\", "/").casefold()
                 or normalized_query in _name_key(name)]
    if contained:
        return [(name, "name", 1.0) for name in sorted(
            contained, key=lambda item: (len(_name_key(item)), item.casefold())
        )]
    near = [(name, score) for name in names
            if (score := _near_name_score(normalized_query, name)) >= 0.82]
    near.sort(key=lambda item: (-item[1], item[0].casefold()))
    return [(name, "near", score) for name, score in near[:20]]


def page_number(value, default=0, maximum=100000) -> int:
    try:
        return max(0, min(maximum, int(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def selection_error(resources: dict, values: dict, family: str, rules=None) -> str | None:
    """Reject confirmed cross-family overrides, including saved recipe overrides."""
    target = canonical_family(family)
    if not target:
        return None
    selected = [("model", values.get("model"))]
    loras = values.get("loras") or []
    if isinstance(loras, list):
        selected.extend(("lora", item.get("name")) for item in loras if isinstance(item, dict))
    for kind, name in selected:
        if not name:
            continue
        names = resources.get("lora_name" if kind == "lora" else "unet_name") or []
        matches = exact_matches(names, name)
        if len(matches) > 1:
            return f"资源名 {name} 有歧义，请使用含目录的完整文件名。"
        if matches:
            name = matches[0]
        found, source = resource_family(name, metadata_for(resources, kind).get(name), rules, kind)
        if found and found != target:
            return f"家族不匹配：{name} 属于 {found}（{source}），当前工作流家族为 {target}。请按 model_family 查询。"
    return None
