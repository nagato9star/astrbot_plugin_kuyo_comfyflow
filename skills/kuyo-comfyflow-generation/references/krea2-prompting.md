# Krea 2 提示词写法

适用：生图家族 `krea2` 或使用 Krea 2 权重的工作流。此家族通常使用自然语言提示词，不套用 Anima 的标签顺序。

## 写作方法

- 用自然语言描述画面。Krea 官方指南认为短提示词也能出图，细节充分的提示词通常更能明确控制结果。
- 根据用户给出的内容逐步补足：主体与动作、发生场景、主体关系和位置、构图/镜头、光线、媒介或风格、背景和颜色。选择对当前画面有用的项目即可。
- 保留用户指定的主体、动作、颜色和空间关系。提示词已经具体时，只整理表达；短提示词可以适度明确主体、场景和构图，避免凭空增加道具、人物或剧情。
- 用户指定照片、插画、3D、复古印刷等媒介时保留该媒介。风格选择应服务用户描述，不要自动追加一串互相冲突的审美词。
- 图片中需要出现文字时，把精确文案放在引号中。不要改写用户给出的品牌名、拼写或大小写。
- 可用清楚的场景 brief 组织长提示词，例如主体、环境、机位、光线和材质。不要为了显得详细而重复同义形容词。

## 示例

用户：「红色背景前的蓝发女孩，电影感。」

可整理为：`A blue-haired young woman in a dark jacket, standing against a vivid red studio backdrop, three-quarter portrait, direct gaze, cinematic side lighting with soft shadow falloff, detailed digital illustration`

实际调用时保留用户指定的年龄/角色身份等细节；示例中的具体服装只用于展示句式，不代表默认添加。

## 来源

- Krea 官方模型仓库的 [Prompting guidelines](https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md)：建议自然语言；长而具体的描述有帮助，短提示词同样可用；渲染文字时使用引号。
- Krea 官方 [prompt expansion guidance](https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt)：优先忠实保留用户内容，按主体、情境、构图、光线和风格扩写；细节不足时避免发明与请求无关的对象；已充分描述时轻度润色。
