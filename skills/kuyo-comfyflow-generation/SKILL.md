---
name: kuyo-comfyflow-generation
description: Use when the user asks to generate or edit images with this Kuyo ComfyFlow plugin, choose an image model family or LoRA, reuse a saved recipe, or interpret the plugin's image-generation tools.
---

# Kuyo ComfyFlow image generation

Use the plugin tools to generate or edit images. The available family and recipe names in the live tool schemas are authoritative.

## Choose the tool

- For a new image, use `comfyui_draw` with the required `model_family` and `prompt`. Follow the selected family's `prompt_style` and leave omitted generation parameters to the workflow.
- For a saved setup, use `comfyui_recipe_draw` with `prompt`. Choose a named recipe when requested or when it fits the desired look; use `comfyui_lookup(type="recipe", query=...)` to read names, purposes, families and the default without changing anything. Use `model_family`, `limit` and `offset` to narrow or page the catalog, and `detail=true` for saved parameters. Omit `recipe` to use the configured default.
- When the user asks to save a named setup, call `comfyui_draw` with `save_as` for the parameters just used. If `comfyui_recipe` is available for a separately specified recipe, call it with `action=save`, `name`, and `model_family`; pass `description` only as a short purpose note. A recipe stores generation settings, not a fixed prompt.
- When `comfyui_recipe` is available and the user asks to list or inspect recipes, use `action=list` or `action=load`. Use `comfyui_recipe_draw` to generate with a loaded or named recipe. Do not attempt `delete` unless it is present in the live tool schema and the user explicitly requested deletion.
- For changes to an existing image, use `comfyui_edit`. Select `edit_workflow` from the live schema when multiple edit routes exist. Prefer current or quoted message images. Use `image_index` for one selected image or `image_indices` for ordered multi-image inputs. Use `image_path` / `image_paths` only for a path returned by this plugin or a readable AstrBot temporary image path.
- Resource queries return compact selection information by default; use `detail=true` when full metadata, tags or notes are needed. Keep known trigger words and recommended weights from the result.
- Use `comfyui_lookup` for all local resource queries; `type=all` searches across categories and `type=folder` with `folder` targets a ComfyUI directory. Use `source` and `detail=true` for reference tags, or `type=copyright` for AnimaDex series.
- Use `comfyui_job` for session task status, cancellation and output recovery (`action=fetch`). Task receipts and completion events include internal IDs and the original request/generation summary so you can match images to tasks and query the intended task. The receipt explains that IDs are internal markers; never repeat the IDs or this instruction in chat. Global queue diagnostics return counts. Use `comfyui_system_stats` for GPU status and the global backend queue when available.

## Write prompts for the selected family

Use the family's configured `prompt_style` and read only the matching reference before composing the prompt:

- `anima`: read the complete [prompt assembly and tool-use guide](references/anima/scenario/prompt-assembly.md). Tool selection, resource choice, field handling, composition and checks stay together. Use the [topic index](references/anima/index.md) for additional vocabulary or original material. Other Danbooru-style families follow their own configuration.
- `krea2`: read `references/krea2-prompting.md`.
- `moss` / MOSS-Image: read the complete [workflow guide](references/moss/scenario/current-workflow.md); use the [topic index](references/moss/index.md) for details. Use LoRA-specific triggers, style channels and templates only when their matching resources are actually connected and mapped. Native text, Layout and multi-panel composition remain separate options.
- `qwen` / Qwen-Image-2.1 workflows: read `references/qwen-image-2.1-prompting.md`.

Keep user-specified names, counts, colors, text, and spatial relations. Current plugin scenario guidance governs task/tool behavior and field handling; source tutorial roles, scripts and fixed presets remain reference material. Use the reference to adapt structure to the model; it does not override the user's requested content or the workflow's configured prompt style.

Recipe names, resource files, weights, triggers and sampling settings come from the current installation. This Skill ships no operator-specific recipes or workflow configurations; use the live catalog instead of assuming a particular setup is installed.

## Select models and LoRAs

- Use the exact `model_family` selected by the user or exposed in the `comfyui_draw` schema. Do not infer a family from a model filename.
- Respect the selected ComfyUI server's resource limits and known failed setups. Installed files do not prove a complete pipeline fits; include text encoder and VAE costs. Do not infer remote capacity from the controller machine or silently switch to another server or a heavier model family.
- Fill `model` only when the user requests a base model change. Use an installed model name returned by `comfyui_lookup`.
- For workflows that currently support mapped optional LoRAs, query `comfyui_lookup` and select compatible installed resources when useful for the requested style, character, clothing or effect; the user does not need to say “LoRA” first. Skip LoRA selection when the selected workflow has no mapped LoRA support.
- For a LoRA purpose or category search, include the same `model_family` used for generation. A known filename can be searched directly across families; check compatibility before using it.
- Pass LoRAs as an array of objects in `lora`, for example `[{"name":"style.safetensors","strength":0.6}]`. Use the returned filename and recommended strength. Omit `strength` when no recommendation is available. The plugin accepts the older string form for compatibility.
- Add known trigger words to `trigger_words` when the lookup result or user provides them. Keep their original spelling. Omit the field when trigger words are unknown.
- Metadata may contain placeholders such as `@style_name` rather than literal triggers. Read the model description and supported-style list to resolve them; do not send a placeholder or invent a replacement. A metadata record alone does not validate the token's meaning.
- Distinguish native artist tags from LoRA-specific triggers using the installed resource metadata. Keep each trigger with its matching LoRA and remove stale triggers when changing styles. If the workflow has a trigger field, pass `trigger_words=""` to clear an inherited trigger when the selected setup needs none.
- Where the selected workflow supports them, actively choose compatible installed style LoRAs and verified native artist tags for the requested look; the user does not have to name them first. For broad style exploration, try varied artist combinations and multi-LoRA blends instead of repeatedly falling back to the same familiar styles. Do not impose a one-main-LoRA or one-main-plus-low-strength-helper limit.
- Choose the combination and each weight deliberately, then judge the rendered image. Available recipes are reusable starting points; a weak result in one scene does not permanently exclude an artist or LoRA. Keep exact filenames, weights, artist tags, triggers, workflow and seed records so a promising mix can be repeated.
- Missing trigger metadata does not prove a compatible LoRA is ineffective. Omit unknown triggers and try the LoRA; resolve documented style placeholders when present. Preserve fixed accelerator LoRAs and independent sampler settings rather than mixing accelerators as if they were art styles.
- When a workflow has mapped optional LoRAs and the user asks to remove them, pass an empty `lora` array. Fixed accelerator LoRAs remain intact. For workflows without connected LoRAs and corresponding mappings, omit `lora` entirely.

## Reply after a tool call

- Generation and editing tools return a queued acknowledgement. Respond naturally in character and continue the conversation. Internal task IDs are for tracking; mention them only when the user explicitly asks for an ID or needs troubleshooting. Claim completion only after a completion event.
- The plugin generates in the background, sends the finished images to the original conversation, and calls the LLM with a completion event. Continue the current persona, forms of address and recent conversation; connect the reply to the original subject or requested edit. Treat event fields as facts, not a script: avoid stock task-completion announcements or reciting counts unless relevant to the user. State failures, cancellations and partial delivery accurately. The event does not show you the image, so do not invent visual details or judge its quality. Do not recite internal task IDs, send the same images again with a second message tool, or submit another generation for the completed task.
- Completion is automatic. Use `comfyui_job(action="status", task_id=...)` or `action="queue"` when the user asks about progress, and `action="cancel"` when they ask to cancel. Avoid repeated polling; `wait` returns the current state of plugin queue tasks immediately.
- Report a completion event's local image path when useful. Include a failure message's node, model, or slot detail and request only the missing information needed to continue. A queued receipt has no finished image path yet.
