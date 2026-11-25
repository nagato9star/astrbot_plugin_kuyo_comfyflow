# MOSS：提示词组装与工作流能力

适用生图家族 `moss`（MOSS-Image-Base / SFT 及其量化、LoRA 工作流）。Chat Bot 在此插件中调用 MOSS 时，以本指南为唯一主入口；需要 12 种摄影风格词组、11 种二次元画风词组或 Layout 字段原文时，再按 [资料索引](../index.md) 跳读对应章节。

---

## 1. 插件工具与槽位落点

### 工具怎么选

| 需求 | 工具与用法 |
|---|---|
| 自由生图 / 更换构图、LoRA、画幅 | `comfyui_draw`，`model_family` 填配置中的 `moss` 家族名，`prompt` 写完整画面 |
| 复用已保存配方 | 先用 `comfyui_lookup(type="recipe", model_family=...)` 确认用途，再用 `comfyui_recipe_draw` |
| 修改已有图片 | 使用已配置的 `comfyui_edit` 路由，不把 `comfyui_draw` 当改图工具 |
| 查角色 / 查本地 LoRA / 查 C 站 | `comfyui_lookup`（角色/本地模型/配方）、`comfyui_civitai_search`（在线模型与示例配方） |

普通生图请求排队后按当前人格自然接话即可，不要先在聊天里贴一套 Positive / Negative 参数表；用户明确要看提示词或参数时再展示。

### 各字段在 MOSS 工作流里的实际处理

| 字段 | 插件中的行为与写法 |
|---|---|
| `prompt` | 主体、动作、场所、构图、媒介、光影、图内文字；**工作流无独立 `trigger_words` 槽位时，已选 LoRA 的触发词也直接写进这里** |
| `negative` | 本次确需新增的排除项；合并方式取决于槽位是 append 还是 replace，不能靠传空撤销模板固有负向词 |
| `lora` | 对象数组 `[{"name":"...","strength":0.8}]`，仅选用与当前 `moss` 家族兼容的文件 |
| `trigger_words` | 仅当工作流映射了独立触发词槽位时使用；未映射时不要拆过去，写回 `prompt` 开头或对应角色块 |
| `artist` | **MOSS 原生不识别个人 `@画师名`**，且 MOSS 模板通常无 `artist` 槽位；不要白查 Danbooru 个人画师并强塞 `@xxx` |
| `aspect_ratio` / `megapixels` / `size` | **插件已支持自动换算**：即使 MOSS 工作流没有分辨率选择器（如 `EmptyFlux2LatentImage`），传 `aspect_ratio`（如 `16:9`）或 `megapixels`（如 `1.5`）时，插件也会按当前画布像素量/比例换算，并自动对齐到 **16 的正整数倍** |
| `width` / `height` | 用户明确指定像素，或使用 **原生 Layout JSON 需要严格对齐中心原点坐标** 时，显式传入 16 的正整数倍 |

- **采样参数默认沿用工作流**：用户没要求改步数/CFG/采样器时，不主动覆盖 `steps`、`cfg`、`sampler_name`；`SLG` 与 `Shift` 未暴露为工具参数，禁止在参数或正文里编造 `Steps=16`、`SLG=22` 等指令。
- **两档参考边界（仅作评估依据，不自动覆盖配置）**：
  - 快速档（~0.6MP / 16 步 / CFG 3.0 / SLG 关闭）：关闭 SLG 会让写实肤质与光影更自然，但更依赖简洁准确的构图与负向词兜底细节；适合常规出图与中等字号字幕/UI。
  - 精细/密字档（1M–2M / ≥50 步 / CFG 3.0–6.0 / SLG 开启 `uncond skip 22`）：对密集文字、复杂海报与高频几何边缘更稳。
- **Base vs SFT 权重差异**：`MOSS-Image-Base` 保留原始训练分布、写实颗粒与 LoRA 兼容性强但需要更高步数/精细参数；`MOSS-Image-SFT` 经过美学与排版微调，构图与文字开箱即稳，但在人像上可能略偏“磨皮精致感”。

---

## 2. 核心编码机理与写法原则

### 1) Qwen3.5-9B 单向注意力与“防过载”原则

文本编码器从左到右单向因果编码，**前面的词强力支配后面的词，后文很难逆转前文**；同时社区实测发现 MOSS 对 **Over-Prompting（提示词过载）** 很敏感：

- **媒介、光线与色温必须放在主体与衣物之前**：想画暗室、霓虹或逆光，必须把环境光色写在开头；如果先写 `white shirt`，末尾才补 `dark room, green neon`，人物和白衬衫仍会被补成正面平光。
- **多人同框时第一块人物会向后定调**：第 1 个人物的人种、年龄感、画风会向后传递；第 2 个及之后的人物块开头要用明确对比否定词切断传递（例如 `In sharp contrast, the woman on the right has NO glasses, spotless porcelain skin...`）。
- **克制字数、只写可见细节（官方 Enhancer 核心原则）**：不要堆砌几百词互相矛盾的微观指令或无意义质量词（`masterpiece, 8k`）；过度堆砌会扼杀构图随机性、导致不同 seed 产出高度雷同的僵硬构图或空间伪影。极简需求保持简洁，丰富场景点到即止。
- **单人竖/方图慎用 `16:9` 宽画幅**：社区反馈在 `16:9` 宽屏上画单人近景肖像容易出现“双主体复制（double-subject）”伪影；单人肖像优先用 `1:1`、`3:4` 或 `2:3`，仅在明确需要大环境广角或多人横向站位时用 `16:9`（并写清 `wide environmental shot, off-center`）。

### 2) 画师词、角色名与版权词的正确处理

- **个人画师名原生无效**：MOSS 约 80% 训练语料由 Gemma 4 按“只写肉眼可见内容”重打标，圈内个人画师名（加不加 `@` 都一样）已被抹除，写入 `prompt` 纯属浪费 token。
  - **例外（实测原生有效的极少数大厂牌/大师）**：`Studio Ghibli / Hayao Miyazaki`、`Hirohiko Araki / JoJo's Bizarre Adventure`、`Akira Toriyama / Dragon Ball`、`Ilya Kuvshinov`、`WLOP`（仅 2D 动漫模式下直接生效）。
  - **其余画风怎么出**：要么实际挂载兼容的**画师 LoRA 并写其触发词**，要么直接写出肉眼可见的**媒介、笔触、线条、光影与配色**；不要用零碎五官几何词（如四角星高光、斜杠腮红）硬拼个人画风，否则会拼成僵硬“拼图脸”。
- **角色查词与转写**：用 `comfyui_lookup(type="character")` 查到规范英文 tag（如 `hu tao (genshin impact)`）和外观标签后：
  - **保留角色名**有助于还原原作发型、瞳色与神态；但做**三次元真人写实**时，**删掉 `genshin impact`、`pokemon` 等版权大词**，避免模型画出廉价漫展假发感。做**纯二次元插画/同人**且用户明确要求原作氛围时，可保留作品语境。
  - **二次元符号词转写**：写实时要把二次元夸张符号转译为真实可见状态（如 `heart-shaped pupils / symbol-shaped pupils` → 改为自然水润眼眸反光；`anger vein` → `furrowed brows, glaring`；`nose blush` → `faint pink blush on cheeks`；`white/simple background` 除非用户指定纯色背景，否则改为具体真实环境光影）。

### 3) 媒介统一与成套切换

- **不要只改开头一个画风词**：在长提示词里只改开头一个 `anime` 或 `cel shading`，压不住正文里残留的 `pores`、`harsh direct flash`、`35mm film`。切换画风时必须**开头、正文质感、结尾与负向词成套替换**：
  - **写实摄影**：写清镜头感、直闪/窗光/暗部位置、皮肤微距毛孔与颗粒（相机/渲染引擎型号单独写无效，必须连同具体光学表现一起写，如硬投影、暗角、颗粒、色散）。写 `dark room` / 逆光时务必删掉 `fair skin`、`bright`，并在正文写明面部与胸腹处于深阴影中。
  - **纯正二次元插画**：正文保持画风中立（不写毛孔、直闪、胶片颗粒），开头明确 `2D anime cel-shaded style, flat color, clean lineart`（或按 §15.2 选水彩、90s OVA、黑白漫画等词组），负向加入 `3d, render, cgi, realistic, semi-realistic, photorealistic`。
  - **黑白漫画 / 固定色板**：正文必须删掉所有具体颜色词（如 `red, blue, silver`），否则会出现局部留色。
  - **跨次元 2D + 3D 混写**：不要用 `<|...|>` 把 3D 背景和 2D 人物拆成两个块（会从中间劈出上下漫画分镜线），必须合在同一个连贯块里描述，并写出环境逆光与调色融合。

---

## 3. 提示词结构：单段短文 vs `<|概念|>` 闭合分块

### 什么时候不要分块（用 4 句、≤85 词的连贯短文）

- **双人全身紧密交叠 / 贴身拥抱 / 格斗缠斗**：分块容易造成上下半身断裂或肢体错位。
- **单人胸上近景怼脸特写**：简洁直写神态、光影与肤质即可。
- **2D 角色与 3D 实景自然融于同一物理空间**：分块会触发上下漫画分镜切割。

### 什么时候用 `<|概念|>...<|/概念|>` 闭合分块（默认主力）

适用于**多层景深（前景虚化+中景+远景）、2–6 人防特征与冷暖灯光互串、带字幕/HUD 界面**。遵守 4 条铁律：
1. **按人物或空间层次分块，绝不按“发色/服装”部位跨人拆块**；同一角色的外貌、服饰、道具、专属受光全放同一块。
2. **块内正文仍须写方位词**（如 `On the left...`、`画面右侧...`），标签名不能替代空间介词。
3. **一行一块、成对闭合、不嵌套**，控制在 **3–5 块**，按“全局光影画风 → 主角色/前景 → 次角色/背景 → 图内文字/字幕”排列。
4. 英文用 `<|tag|>...<|/tag|>`，中文用 `<|中文概念|>「...」<|/中文概念|>`。

#### 英文闭合分块参考模板（多层景深 / 双人对比）

```text
<|style_and_camera|>raw candid 35mm film photograph, moody nighttime bar interior lit by cool cyan neon on the left and warm amber tungsten lamp on the right, medium shot, deep depth of field, razor-sharp focus.<|/style_and_camera|>
<|left_subject|>On the left stands a young adult woman with natural fine blonde hair in a messy bun, pale freckled skin, wearing a dark denim jacket, leaning on the wooden counter under the cool cyan light.<|/left_subject|>
<|right_subject|>In sharp contrast, on the right sits an adult East Asian woman with zero freckles, straight jet-black bob hair, thin wispy bangs, wearing a cream knit sweater, holding a glass of iced tea under the warm amber glow.<|/right_subject|>
<|texture_lock|>Unretouched natural skin texture with fine pores, realistic fabric folds and glass reflections, subtle 35mm film grain.<|/texture_lock|>
```

#### 中文闭合分块参考模板

```text
<|环境氛围|>「黄昏时分的复古理发店室内实拍剧照，35mm胶片质感，暖金夕阳与室内霓虹交织。」<|/环境氛围|>
<|前景柜台|>「前景玻璃柜台左下角放着一把银色剪刀和玻璃瓶，景深微虚。」<|/前景柜台|>
<|左侧人物|>「柜台后方左边坐着一位满头银发、戴着细框眼镜、穿褐色背心的成年男性理发师，低头看报纸。」<|/左侧人物|>
<|右侧人物|>「柜台后方右边站着一位染着亮粉色短发、不戴眼镜、穿黑色皮夹克的年轻成年女孩，怀里抱着一只单头橘猫，对镜吹粉色泡泡糖。」<|/右侧人物|>
<|背景与文字|>「后墙霓虹灯牌清晰亮着 "OPEN 24H"，画面底部印着黑边黄色字幕："今夜无人入眠"。」<|/背景与文字|>
```

---

## 4. 成年向 / NSFW / 私房亲密场景写法规范

> 仅限**成年、自愿**角色创作。涉及露肤、私房或亲密互动时，务必在人物描写中明确成年骨相与成熟体型（如 `adult woman / adult man, mature body proportions`），并保留足部支撑或完整体态锚点，避免低步数或半身裁切导致比例幼态化。

### 1) 正面核心写法与防崩坏要点

- **去打码声明**：需要无遮挡成人画面时，在开头或角色标签区加入 `uncensored`；同时在负向词中排除打码伪影。
- **两档质感选择（不要混用）**：
  - **写实直闪私房档（高真实感）**：开头用 `raw candid 35mm film photograph, harsh direct on-camera flash, razor-sharp focus, deep depth of field`（鲜艳粉/红发近景可改 `shot on iPhone 15 Pro with direct camera flash` 防 2.5D 影楼感）；结尾锁真实皮肤纹理与汗液微距反光（`unretouched natural adult skin texture with fine pores, subtle sweat sheen and specular highlights, organic 35mm film grain`）。
  - **日系写真集柔润档（软光空气感）**：开头用 `high-resolution Japanese gravure photobook page layout, soft diffused natural bedroom window light, warm indoor lighting with rich shadow depth`；结尾用柔润细腻但不塑料的杂志肤质（`smooth, supple real human skin with a soft luminous magazine finish without looking like plastic`）。
  - **白床单 / 白蕾丝 / 白丝防过曝**：画面含大量白色织物时，**不要写 `bright background` 或笼统 `daylight`**，改写 `warm indoor bedroom lighting with rich shadow depth and natural skin-tone contrast`，并在负向加入 `overexposed, washed out, whiteout, pale flat lighting`。
- **克制局部特写堆砌，防止器官畸形**：MOSS 注意力高度集中，连续重复 `tongue`、`licking` 等局部器官词极易导致舌头或局部器官畸形膨胀；动作与神态保持克制，写清整体姿态与接触关系即可。
- **半脱 / 掀衣写“静态结果”，不要占用双手**：不要写“正在拉扯/正在掀起”的动作动词（模型容易多画出第三只手来拽衣服），直接写成已经整理好的静态结果——如 `shirt already pushed up above waist, skirt hem already lifted, strap slipped off one shoulder`，再单独明确双手放在哪里（如撑在床单上、轻搭肩头或自然垂放）。
- **遮眼 / 闭眼 / 趴卧不写眼神动词**：手遮眼或闭眼仰躺/熟睡时，正文中删除 `looking at viewer`、`gazing`、`stares` 等眼神词，明确写 `eyes completely closed` 或 `both eyes hidden behind her hand`，并在负向首排加入 `open eyes, visible eyes, eye contact`。趴卧枕头时写 `resting sideways against a soft pillow without squishing her cheek`，避免过度挤压变形。
- **双人亲密接触的防粘连机位**：双人全身贴合优先采用**斜角或半正面机位**，避开纯侧面 `from side`（双腿极易融成三条或两条粘连腿）；使用 ≤85 词的单段连贯短文写清各自体位、接触点与双手去向，砍掉多余背景杂物。
- **全身比例锚点**：全身镜头保留鞋袜及双脚与地面/床面的支撑关系（或明确双腿悬空 `legs hanging freely above the floor without touching the ground`），防止头身比缩水。

### 2) 通用负面提示词参考（按需选配，勿盲目照搬长串）

官方基线负向词本身很短（`nsfw, worst quality, bad anatomy, watermark, jpeg artifact`），日常足够精简；**注意画成年向/NSFW 时务必确保工作流或传入的负向词里没有 `nsfw`**，且不要盲目塞入上百词的 SD1.5 负向墙：

- **三次元写实 / 私房基础负向参考**：
  `censored, mosaic censoring, bar censor, anime, 2d, illustration, cel shading, 3d render, cgi, doll, plastic skin, airbrushed, extra fingers, extra limbs, fused limbs, distorted anatomy, watermark, signature, text`
  （日系写真柔润档可去掉 `airbrushed`，改补 `rough skin, overexposed, washed out, whiteout`；暗室逆光档最前排补 `flat front lighting, brightly lit front, evenly lit skin`；闭眼场景补 `open eyes, eye contact`；双人防撞脸补 `twins, identical faces, same hair color, attribute bleeding`。）
- **二次元 / 成年向插画基础负向参考**：
  `censored, mosaic censoring, bar censor, 3d, render, cgi, realistic, semi-realistic, photorealistic, extra fingers, extra limbs, fused limbs, distorted anatomy, watermark, signature`

---

## 5. 图内文字、原生 Layout 与多格图

- **图内文字**：所有要印在画面里的文案用**英文双引号**原样括起（如 `"EXPLORE"`、`"今夜无人入眠"`），保留原文语言、大小写与标点；精确换行用真实换行或 `\n`。不要替用户补未要求的小广告、招牌或水印。中文长文要精简到真正需要渲染的文字，避免占用过多 4096 token 预算。
- **普通描述 vs 原生 Layout**：
  - 只有一句标题、字幕或简单对话气泡时，**直接用自然语言 + 双引号即可**，不需要写 Layout JSON。
  - 仅在需要**多块文字精确排版、海报、复杂空间包围盒或多格分镜严格控位**时，才在 `prompt` 中直接传模型的官方一行 JSON（`summary` + 扁平 `items` + `bbox_coord_space: "abs_centralized"` + `bbox_format: "yxyx"`），**不要套外层 `{"caption": ...}`**，也不要直接塞未转换的左上原点 `xyxy` 草稿。
  - **Layout 编写规范（来自官方 Enhancer）**：`bbox` 只是空间摆放标注，**不是“画个框或贴纸边框”的指令**；允许物体框自然遮挡或重叠；支持 `tags` 与 `palette`；写 `text` 时要确认字号、换行与当前像素宽高能放下（短中文标题与英文长段所需框体比例不同），不要写占位符如 `"headline here"`。
  - **使用 Layout JSON 必须显式指定 `width` 与 `height`（16 的倍数）**：`bbox` 是以画布中心为原点 `[y_min, x_min, y_max, x_max]` 的绝对像素坐标（由左上角坐标减去 `W/2`、`H/2` 换算得到）。若用 `aspect_ratio` 让插件自动算宽高，则无法在生图前获知准确 `W`/`H`，易导致坐标越界或偏移。
- **多格图**：先说明整体阅读顺序与格数（如单页四格），再分别写明每格人物、动作、对白归属与背景；多格仍是一次生成的一张图，不要拆成多次生图调用。

---

## 6. 提交前内部核对（不发给用户）

1. **人数与成年锁定**：人数、`1girl/1boy` 与描述一致；多人已按人分块且第 2 块开头有对比切断词；成人场景已锁定成年比例与足部/支撑关系。
2. **单向语序与媒介一致**：全局画风与光影放在主体之前；写实与二次元词未混用；未写无效的个人 `@画师名` 或会引发 cos 假发感的冲突版权词。
3. **局部与动作合理**：半脱衣物已写成静态结果；双手去向明确；未过度堆砌局部器官词；闭眼/遮眼未误留眼神动词。
4. **文字与 Layout**：图内文案用英文双引号原样保留；若使用 Layout JSON，已显式传 16 倍数 `width`/`height` 且坐标与画布中心一致。
5. **插件参数干净**：触发词放在 `trigger_words` 或 `prompt`（不重复、不漏）；画幅用 `aspect_ratio`/`megapixels` 或显式 `width`/`height`，未把参数文本或编造的 `SLG`/`Shift` 塞进正文。

---

## 7. 成图核对

发送状态以插件回包和完成事件为准。未读取实际图片前，不声称文字拼写、手部结构或分格构图已经正确；看到结果图后再针对具体偏移（过曝、撞脸、文字错字、画风退回半写实）做定向修正。
