## 14. 按官方 Layout JSON schema 生成结构化提示词
官方将这一接口标注为 `Experimental`，本节各条照搬官方 `USERGUIDE > Layout Prompts` 的原文与字段定义，未作改写。官方同时说明：`The model supports combining natural language and structured input.`，即自然语言与结构化输入可以混用。

### 14.1 照搬官方示例的 JSON 结构并按需替换字段值
以下是官方给出的示例，生成提示词时保持这一 JSON 结构，按实际需求替换各字段的值：
```json
{
  "summary": {
    "meta": ["poster"],
    "canvas": "single",
    "caption": "A minimalist travel poster with a cream background, a blue mountain illustration, and the title EXPLORE at the top."
  },
  "items": [
    {
      "label": "text",
      "caption": "Large dark-blue uppercase title centered at the top.",
      "text": "EXPLORE",
      "bbox": [-416, -384, -288, 384]
    },
    {
      "label": "object",
      "caption": "A flat blue mountain illustration in the lower center.",
      "bbox": [-192, -288, 384, 288]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

### 14.2 按官方定义填写各字段并换算中心原点坐标
以下是官方文档里的字段定义原文：
- `summary.caption`：`A description of the overall scene and style.`，即整体场景与风格描述。
- `items`：每个元素的类别（`label`）、描述（`caption`）、边界框（`bbox`）；渲染文字时用 `text` 指定内容。
- `bbox_format: "yxyx"`：坐标顺序为 `[y_min, x_min, y_max, x_max]`。
- `bbox_coord_space: "abs_centralized"`：绝对像素坐标，以画布中心为原点，`x` 向右、`y` 向下。例如 1024×1024 画布在两个坐标轴上的范围均为 −512…+512。
- 从常规的左上角原点坐标换算时，官方原句为 `subtract W/2 and H/2 from conventional top-left-origin x and y coordinates, respectively`，换算公式为 `[y_min, x_min, y_max, x_max] = [y′_min − H/2, x′_min − W/2, y′_max − H/2, x′_max − W/2]`。
- 官方明确说明选用绝对像素坐标的原因：`We use absolute pixel coordinates, which we believe are better suited to layout control across different resolutions and aspect ratios.`

### 14.3 按官方规定限制各字段的填写范围
以下是官方 skill 文档给出的各字段取值范围：
- `summary.canvas`：填 `single` / `multi_panel` / `collage` 三值之一。
- `items[].label`：填 `object` / `character` / `text` 三值之一。训练语料里还包含 `panels`、`characters`、`watermarks` 等类别，但公开编译器只接受上述三个；论文推理增强器的 `elements` 属另一套 schema，官方 skill 原文要求不要写入：`Do not add ... scene-schema elements`。
- 官方支持 `summary.palette` 与 `items[].palette`（HEX 字符串数组），官方原文说明为 `palette on summary is also supported`。

### 14.4 处理坐标取整、画布尺寸与负面提示词参数
- 分数边缘按官方规则对左上坐标作 `floor`、对右下坐标作 `ceil`，最终坐标为整数且宽高为正。
- 编译产物中删去 `width/height` 等画布尺寸字段，尺寸在生成参数中另外设置，与本文件 §13 的采样参数一致；`W`、`H` 取正的 16 倍数。
- 负面提示词填入 CFG 的 `negative_prompt` 参数，官方 `Prompt Guidelines` 给出的 5 条已由本文件 §10 的负面提示词覆盖并扩充。论文 `planner` 的 `anchors` / `must_not` 是推理增强器内部字段，公开 schema 中不存在，不要写入。

### 14.5 将本文件其余规则填入对应的 JSON 字段
> 本段留空。



---

