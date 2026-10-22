"""为完成回复提供任务事实和对话指引，避免模型照读兜底播报。"""

from .receipts import hide_task_ids, task_reference


def completion_summary(job: dict, state_label: str) -> str:
    """事实与内部关联信息供模型使用，不作为用户回复模板。"""
    facts = [
        f"任务状态：{state_label}",
        f"已发送图片：{job['images_sent']} 张",
    ]
    for key, label in (("error", "异常"), ("note", "补充")):
        if job.get(key):
            facts.append(f"{label}：{job[key]}")
    return hide_task_ids("\n".join(facts), [job.get("prompt_id", "")]) + "\n" + task_reference(job)


def completion_prompt(summary: str) -> str:
    """沿用原人格和历史，把后台结果接回原来的聊天。"""
    return (
        "[ComfyUI 后台任务事件]\n" + summary
        + "\n以上是后台事实，不是要照读的回复文案。"
        "请延续当前人格、称呼和聊天语气，结合原请求以及最近的对话自然接话，简短回应这次具体的生图或编辑需求。"
        "不要逐项汇报状态、数量或内部编号，也不要输出固定的任务完成播报；只有用户关心数量或需要区分多张结果时才提数量。"
        "图片是否发送及任务是否成功以本事件为准；失败、取消或仅部分图片交付时如实说明，不能说全部完成。"
        "你没有看到生成图片，不要编造画面细节或评价成图效果；可以提及原请求中的主体或修改要求。"
        "图片发送由后台负责，无需再次生图、轮询或发送图片。直接输出发给用户的话。"
    )
