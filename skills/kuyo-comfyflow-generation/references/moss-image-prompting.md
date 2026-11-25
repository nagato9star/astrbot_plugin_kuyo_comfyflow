# MOSS-Image 速查与补充参考

适用：生图家族 `moss`（MOSS-Image-Base / SFT 及其量化、LoRA 工作流）。

> **首选实战入口**：日常组织提示词、工具槽位分工、`<|...|>` 分块模板及成年向/NSFW 写法，请先完整阅读 [MOSS 工作流与提示词组装指南](moss/scenario/current-workflow.md)。本页仅作表达方式、原生 Layout 换算与官方参考的精简速查。

## 三种输入形式

MOSS-Image 同时支持 **自然语言**、**Danbooru 风格 tag 串**、**原生 Layout JSON**：
- 默认优先用**自然语言**（或 `<|...|>` 分块自然语言）；仅在用户明确要求 tag 式提示词或已给出 tag 串时，按 tag 顺序组织。
- **不要往正文塞个人 `@画师名`**：MOSS 经 Gemma 4 重打标，原生不识别绝大多数个人 Danbooru 画师名（仅 `Studio Ghibli / Hayao Miyazaki`、`JoJo / Hirohiko Araki`、`Akira Toriyama`、`Ilya Kuvshinov`、`WLOP` 等少数大牌原生有效）。需要个人画师风时，必须实际挂载兼容的画师 LoRA 并使用其真实触发词。
- tag 串常用顺序：`Quality/Style → Meta → Character → (Series, 按写实/二次元取舍) → General`。质量词点到为止（如 1–2 个），不要整串堆砌；文本编码上限 4096 token，长而不精反而稀释注意力。

## 图中文字

- 画面内要显示的文案用英文双引号原样括起（如 `"EXPLORE"`、`"老街早餐"`），保留原文语言、标点、大小写与数字。
- 精确换行直接写进提示词（真实换行或 `\n`）；只有一句文字或简单对话框时直接用自然语言即可，不必上 Layout JSON。

## 原生 Layout JSON（精确排版 / 多区块控位）

插件直接把 `prompt` 字符串送进工作流，传一行合规 JSON 即可（不要外包 `{"caption":...}`）：

```json
{"summary":{"meta":["poster"],"canvas":"single","caption":"整体画面与风格描述"},"items":[{"label":"text","caption":"顶部居中的深蓝大写标题","text":"EXPLORE","bbox":[-416,-384,-288,384]},{"label":"object","caption":"下方居中的扁平蓝色山峦插画","bbox":[-192,-288,384,288]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}
```

- `summary.canvas` 取值：`single` / `multi_panel` / `collage`；`items[].label` 取值：`object` / `character` / `text`；可选 `summary.palette` / `items[].palette`（HEX 数组，配合删掉正文自然语言颜色词）。
- `bbox` 为 `[y_min, x_min, y_max, x_max]`，以**画布中心为原点**、绝对像素、x 向右 y 向下；常规左上原点坐标减去 `W/2`、`H/2` 即得（左上 `floor`、右下 `ceil`）。
- **尺寸对齐**：常规生图可直接用 `aspect_ratio` / `megapixels` / `size`（插件会自动换算并对齐 16 的倍数）；但**使用 Layout JSON 时请同时在 `comfyui_draw` 中显式传入 16 倍数的 `width` 与 `height`**，确保画布宽高与 `bbox` 中心坐标严格一致。

## 官方参数参考（不覆盖当前工作流默认配置）

- **分辨率**：官方建议总像素 1M–2M（可外推 0.25M–4M），宽高必须是 **16 的正整数倍**；部分快速方案用 0.6MP / 16 步 / CFG 3.0 / 关闭 SLG。
- **采样**：官方 Base/SFT 高清推荐 Steps ≥ 50、CFG 3.0（3.0–6.0，长提示词用 3.0 即可）、Shift 5.0、采样器 Euler；SLG（`single_layers` 22/24/26）用于收敛高频几何与密集小字。
- **工具限制**：当前插件未公开 Shift / SLG 参数，由工作流本身决定，勿在工具调用或正文里自行编造。

## 上游参考来源

- [资料索引（按章节号查找细节）](moss/index.md)
- 官方 [USERGUIDE](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md)（[中文版](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE_ZH.md)）
- [MOSS-Image-SFT 模型卡](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) · [ComfyUI 说明](https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/README.md)
- 官方 [Prompt Enhancer skill](https://github.com/OpenMOSS/MOSS-Image/blob/main/skills/moss-image-prompt-enhancer/SKILL.md)（本插件直接接收 `prompt` 字符串，不采用其本地脚本与外层 JSON 包装）
