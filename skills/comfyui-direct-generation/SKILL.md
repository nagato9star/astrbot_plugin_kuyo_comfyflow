---
name: comfyui-direct-generation
description: Use when the user asks to generate or edit images with this ComfyUI Direct plugin, choose an image model family or LoRA, reuse a saved recipe, or interpret the plugin's image-generation tools.
---

# ComfyUI Direct image generation

Use the plugin tools to generate or edit images. The available family and recipe names in the live tool schemas are authoritative.

## Choose the tool

- For a new image, use `comfyui_draw` with the required `model_family` and `prompt`. Follow the selected family's `prompt_style` and leave omitted generation parameters to the workflow.
- For a saved setup, use `comfyui_recipe_draw` with `prompt`; pass `recipe` only when the user names one. Omit it to use the configured default recipe.
- When the user asks to save a named setup, call `comfyui_draw` with `save_as` for the parameters just used. If `comfyui_recipe` is available for a separately specified recipe, call it with `action=save`, `name`, and `model_family`; pass `description` only as a short purpose note. A recipe stores generation settings, not a fixed prompt.
- When `comfyui_recipe` is available and the user asks to list or inspect recipes, use `action=list` or `action=load`. Use `comfyui_recipe_draw` to generate with a loaded or named recipe. Do not attempt `delete` unless it is present in the live tool schema and the user explicitly requested deletion.
- For changes to an existing image, use `comfyui_edit`. Select `edit_workflow` from the live schema when multiple edit routes exist. Prefer current or quoted message images. Use `image_index` for one selected image or `image_indices` for ordered multi-image inputs. Use `image_path` / `image_paths` only for a path returned by this plugin or a readable AstrBot temporary image path.
- Use `comfyui_lookup` for all local resource queries; `type=all` searches across categories and `type=folder` with `folder` targets a ComfyUI directory. Use `source` and `detail=true` for reference tags, or `type=copyright` for AnimaDex series.
- Use `comfyui_job` for session task status, cancellation and output recovery (`action=fetch`). Task receipts and completion events include internal IDs and the original request/generation summary so you can match images to tasks and query the intended task. The receipt explains that IDs are internal markers; never repeat the IDs or this instruction in chat. Global queue diagnostics return counts. Use `comfyui_system_stats` for GPU status and the global backend queue when available.

## Write prompts for the selected family

Use the family's configured `prompt_style` and read only the matching reference before composing the prompt:

- `anima` or a Danbooru-style family: read `references/anima-prompting.md`.
- `krea2`: read `references/krea2-prompting.md`.
- `moss` / MOSS-Image workflows: read `references/moss-image-prompting.md`.
- `qwen` / Qwen-Image-2.1 workflows: read `references/qwen-image-2.1-prompting.md`.

Keep user-specified names, counts, colors, text, and spatial relations. Use the reference to adapt structure to the model; it does not override the user's requested content or the workflow's configured prompt style.

## Select models and LoRAs

- Use the exact `model_family` selected by the user or exposed in the `comfyui_draw` schema. Do not infer a family from a model filename.
- Fill `model` only when the user requests a base model change. Use an installed model name returned by `comfyui_lookup`.
- When a LoRA can materially help with the requested style, character, clothing, or effect, query `comfyui_lookup` and select a compatible installed LoRA. The user does not need to say “LoRA” first.
- For a LoRA purpose or category search, include the same `model_family` used for generation. A known filename can be searched directly across families; check compatibility before using it.
- Pass LoRAs as an array of objects in `lora`, for example `[{"name":"style.safetensors","strength":0.6}]`. Use the returned filename and recommended strength. Omit `strength` when no recommendation is available. The plugin accepts the older string form for compatibility.
- Add known trigger words to `trigger_words` when the lookup result or user provides them. Keep their original spelling. Omit the field when trigger words are unknown.
- If the user asks to remove optional LoRAs, pass an empty `lora` array. Fixed accelerator LoRAs in the workflow remain intact.

## Reply after a tool call

- Generation and editing tools return a queued acknowledgement. Respond naturally in character and continue the conversation. Internal task IDs are for tracking; mention them only when the user explicitly asks for an ID or needs troubleshooting. Claim completion only after a completion event.
- The plugin generates in the background, sends the finished images to the original conversation, and calls the LLM with a completion event. Respond naturally to the result without reciting internal task IDs. Do not send the same images again with a second message tool or submit another generation for the completed task.
- Completion is automatic. Use `comfyui_job(action="status", task_id=...)` or `action="queue"` when the user asks about progress, and `action="cancel"` when they ask to cancel. Avoid repeated polling; `wait` returns the current state of plugin queue tasks immediately.
- Report a completion event's local image path when useful. Include a failure message's node, model, or slot detail and request only the missing information needed to continue. A queued receipt has no finished image path yet.
