# MOSS-Image 提示词写法

适用：生图家族 `moss`（MOSS-Image-Base / SFT 权重，含 int8 量化工作流）。家族配置和具体工作流字段优先。

先按 [工作流使用入口](moss/index.md) 使用工具和组织提示词。原生文字与 Layout 按需求使用。

## 选择表达方式

MOSS-Image 同时支持三种输入：自然语言、Danbooru 风格 tag 串、JSON Layout。默认用自然语言写整句画面描述；用户点名要 tag 式提示词、或已给出 tag 串时，按 tag 顺序写。Layout 只在用户明确需要精确文字排版、海报、多面板、受控物体关系时使用。

- 语言：英文训练量最大，效果最稳；中文同样可用。用户用中文描述画面、且要渲染中文字时，直接用中文写提示词，画面文字原样保留中文。
- 长度：模型能吃长提示词（文本编码上限 4096 token），但长不等于好。按画面需要补足，不堆同义形容词和质量词。

## 自然语言写法

按“看得见的东西”组织，可参考顺序：主体与动作 → 外观细节 → 场景与环境 → 物体之间的空间关系 → 构图与镜头 → 光线与材质 → 风格与媒介 → 需要呈现的文字。

- 保留用户给出的数量、颜色、位置、材质、媒介和排除项。已经具体的请求只做轻度整理。
- 空间关系用明确方位词写：“冰箱在料理台左侧，台面上有一个水果盘”。模型对文字化的空间关系响应良好，这是它的强项之一。
- 光线写到来源、方向、软硬；材质写到反光质感，避免只写“电影感”“高级感”这类空词。

## tag 串写法

推荐顺序：Quality → Score → Meta → Year → Character → Series → Artist → General。

`best quality, highres, hatsune miku, vocaloid, @foo bar, 1girl, blue hair`

- 画师标签以 `@` 开头，与 Anima 习惯一致；用户点名画师时先用 `comfyui_lookup(type="artist")` 查规范名，查不到就写可观察的画面风格，不要编 tag。
- 质量词和分数词可以只用一两个，不必整串堆。

## 图中文字

这是模型的主要卖点，值得认真写。

- 需要出现的文案用引号原样括起，保留原文语言、标点、大小写和数字。不要改写、翻译或“顺便润色”。
- 精确换行直接写进提示词：要么真实换行，要么写 `\n`。不写换行时模型会自己排版，结果不一定符合预期。
- 需要多块文字各自定位时，才考虑 Layout；只是“画面上有一句话”用普通自然语言就够。
- 用户没要求文字时，不要自行加招牌、水印、标语。

## 负面提示词

通过 `negative` 填写本次确需新增的排除项。先沿用当前工作流或配方的负向设置，合并方式以实际槽位的 append/replace 为准；不假定每套工作流都预置同一串词。原教程的整套写实负向词不能直接覆盖插画需求，也不要通过传空来假定撤销了追加模式的既有词。

## Layout JSON（实验性）

只有要做精确排版时用。插件直接把 `prompt` 字符串送进工作流，所以 Layout 就是一行 JSON：

```json
{"summary":{"meta":["poster"],"canvas":"single","caption":"整体画面与风格描述"},"items":[{"label":"text","caption":"顶部居中的深蓝大写标题","text":"EXPLORE","bbox":[-416,-384,-288,384]},{"label":"object","caption":"下方居中的扁平蓝色山峦插画","bbox":[-192,-288,384,288]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}
```

- `summary.caption` 写整体场景和风格；`items` 逐条写元素类别（`label`）、描述（`caption`）、文字内容（`text`）和包围盒（`bbox`）。
- `bbox` 是 `[y_min, x_min, y_max, x_max]`，坐标系为画布中心原点、绝对像素、x 向右 y 向下。1024×1024 画布的两轴范围是 −512 到 +512。把常规左上原点坐标减去 W/2、H/2 即得。
- 尺寸必须和生图参数一致：Layout 里的坐标按你实际用的宽高算，`prompt` 本身不携带尺寸，画幅用 `width` / `height` 或 `size` 传。
- 包围盒是摆放标注，不是“画个边框”的指令；元素外观写在 caption 里。

## 采样与尺寸（官方 Base 参考）

本节说明官方 Base 示例，不代表当前 SFT、量化或其他工作流的实际值。当前任务默认沿用已配置参数；原教程的快采样参数也不自动替换这些设置。

- 总像素推荐 1M、1.5M、2M；可外推 0.25M（512×512）到 4M。宽高必须是 16 的正整数倍，否则报错。
- 步数 ≥ 50 效果最佳，CFG 默认 3.0（3.0–6.0；长提示词 3.0 就够，短提示词可适当调高），Shift 5.0，采样器 Euler。
- SLG 是否启用、采用哪种方式及层号，以实际工作流为准。当前生图工具没有直接公开 Shift 或 SLG 参数，不能在工具调用里自行添加这些字段，也不把它们写进画面正文。

## 这个插件里的限制

- 图中文字的长提示词会占用文本编码预算（4096 token）；中英混排的大段文案要精简到真正需要渲染的部分。

## 来源与适用范围

- 官方 [USERGUIDE](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md)（[中文版](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE_ZH.md)）：推理参数、Prompt 规范、tag 顺序、文字换行、Layout JSON 字段与负向标签。
- [MOSS-Image-SFT 模型卡](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)：Base/SFT 差异、默认采样设置、文本编码上限 4096 token。
- [ComfyUI 说明](https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/README.md)：节点、SLG 配置、ModelSamplingSD3 与 shift 的关系。
- 官方 [Prompt Enhancer skill](https://github.com/OpenMOSS/MOSS-Image/blob/main/skills/moss-image-prompt-enhancer/SKILL.md)：caption / Layout 的取舍、文字原文保留、Layout 草稿坐标转换。该 skill 面向文件系统和脚本工具，输出 `{"caption":"..."}` 之类的包装；本插件只接收提示词字符串，因此只采用其中关于描述取舍和坐标的写法，不照搬其文件与 JSON 包装流程。
