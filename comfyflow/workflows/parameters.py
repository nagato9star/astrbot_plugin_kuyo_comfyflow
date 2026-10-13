"""Shared preparation rules for Bot and Workflow Studio generation."""

from .slots import collect_trigger_words


def fill_trigger_words(values: dict, resources: dict, enabled: bool) -> dict:
    values = dict(values)
    if not enabled or values.get("trigger_words") is not None:
        return values
    # Training-frequency guesses caused unwanted prompts in the old automatic path.
    metadata = {}
    for name, info in (resources.get("lora_meta") or {}).items():
        if not isinstance(info, dict):
            continue
        if "trusted_trigger_words" in info:
            metadata[name] = {**info, "trigger_words": info["trusted_trigger_words"]}
        elif info.get("source") not in {
            "tag_frequency",
            "dataset",
            "civitai",
            "lora_manager",
        }:
            metadata[name] = info
    words = collect_trigger_words(metadata, values.get("loras"))
    if words:
        values["trigger_words"] = words
    return values
