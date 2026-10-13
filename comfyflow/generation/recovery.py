"""按已有记录复用图片，必要时从 ComfyUI 补回缺失输出。"""

import asyncio

from ..clients.comfyui import execution_error_message
from ..storage.images import reuse_images, save_image


async def recover_outputs(client, output_dir, prompt_id, *, images=None, paths=()):
    """路径按图片顺序排列，缺失项保留为 None，便于下一次只补缺失图片。"""
    images = list(images or [])
    paths = list(paths or [])
    if paths and (not images or len(paths) == len(images)) and all(paths) and await asyncio.to_thread(reuse_images, paths):
        return images, paths, None
    if not images:
        entry = await client.get_history_entry(prompt_id)
        if entry is None:
            return [], [], f"任务 {prompt_id} 未找到（可能未完成或不存在）。"
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            return [], [], execution_error_message(status, "详见 ComfyUI 日志")
        images = [image for output in entry.get("outputs", {}).values()
                  for image in output.get("images", [])]
        finals = [image for image in images if image.get("type", "output") == "output"]
        images = finals or images
    if not images:
        return [], [], f"任务 {prompt_id} 无图片输出。"

    recovered = paths + [None] * max(0, len(images) - len(paths))
    for index, image in enumerate(images):
        if recovered[index] and await asyncio.to_thread(reuse_images, [recovered[index]]):
            continue
        filename = str(image.get("filename") or "")
        content = await client.download_image(
            filename, image.get("subfolder", ""), image_type=image.get("type", "output")
        )
        if not content:
            return images, recovered, f"图片下载失败（{filename}）。"
        try:
            path = await asyncio.to_thread(save_image, output_dir, filename, content)
        except (OSError, ValueError) as error:
            return images, recovered, f"图片保存失败（{filename}）：{error}"
        recovered[index] = str(path)
    return images, recovered, None
