# Qwen-Image-2.1 提示词写法

适用：Qwen-Image-2.1 文生图或编辑工作流。先按用户目标区分“生成新画面”和“修改来源图”，再遵循所选工作流的字段映射。

## 文生图

- 将请求写成清楚的自然语言画面描述：主体与动作、背景、构图和空间位置、风格/媒介、光线与重要材质。复杂海报、界面或多区域画面要说明各元素的位置和层级。
- 保留用户明确给出的名称、数量、颜色、位置、宽高比和文案。图中需要呈现的文字按原文逐字引用；用户没要求文字时，不要擅自添加招牌、标签或水印。
- 短请求需要适度补足画面关系；已经具体的提示词只做轻度整理。扩写只补完成画面所需的信息，保留请求中的主体与意图。
- 光线要具体到来源、方向或软硬特征；同时让阴影、反光和材质表现彼此一致。
- 官方 Qwen-Image-2.1 Prompt Enhancer 的 T2I 模板会生成英文长段画面描述，并把比例放在单独字段。本插件直接接收 `prompt` 字符串，不接收 Enhancer 的 JSON 包装；可用用户语言写提示词，且不要把宽高比 JSON 或模型输出包装塞进 `prompt`。
- 尺寸通过工具参数控制：用户只说横/竖/方时用 `size`；明确指定像素尺寸时填写 `width`、`height`。

## 编辑来源图

- 以编辑指令开头，明确哪些元素要改变、改成什么样，以及哪些部分必须保持来源图原状。
- 用户只要求局部改动时，把改动范围和目标说清楚，保留未提及的身份、物体数量、构图、色彩关系和媒介。要求整张图重新设计或把主体放进新场景时，才扩展设计新画面所需的背景、光线和构图。
- 要呈现的文字逐字引用；不确定的文案不要自行补写。
- 本插件的 `comfyui_edit` 可按工作流映射读取一张或多张来源图。多图任务用 `image_indices` 按 `source_images` 映射顺序选择消息图片；单图选择使用 `image_index`。提示词可用“第一张图”“第二张图”等清楚指代来源图，并说明要保留的角色、物体和背景关系。只有工作流明确要求特殊引用标记时才使用该标记。
- 尺寸与宽高比由工具参数控制。编辑时可按任务使用 `resolution`、`custom_size`、`width`、`height`；不要在提示词末尾附加尺寸 JSON。

## 来源与适用范围

- 官方 Qwen-Image-2.1 [模型仓库](https://github.com/QwenLM/Qwen-Image-2.1)介绍文生图、编辑、多参考图和文字渲染能力。
- 官方 [Qwen-Image-2.1 Demo](https://huggingface.co/spaces/Qwen/Qwen-Image-2.1)说明中文和英文提示词都可用。
- [T2I prompt rewrite 模板](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_t2i.txt)与 [Edit prompt rewrite 模板](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_edit.txt)属于官方 Prompt Enhancer 的改写指引，描述的是改写器的输出约定。本插件直接把生成提示词送入工作流，因此只采用其中关于保留用户约束、明确空间关系、精确文字和聚焦编辑范围的写法，不照搬其 JSON 输出协议或固定英文长文长度。
