# ✨ Kuyo ComfyFlow

<div align="center">

**局域网直连 ComfyUI。模型家族负责选择工作流，配方负责复用实验好的 LoRA 与采样参数。**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AstrBot](https://img.shields.io/badge/AstrBot-%E2%89%A54.16-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
[![Last Commit](https://img.shields.io/github/last-commit/nagato9star/astrbot_plugin_kuyo_comfyflow)](https://github.com/nagato9star/astrbot_plugin_kuyo_comfyflow/commits/main)

</div>

---

## 📢 简介

Kuyo ComfyFlow 是一款基于 [AstrBot](https://github.com/AstrBotDevs/AstrBot) 的生图插件。

先在 Workflow Studio 工作流页导入 ComfyUI 工作流，再为已有模型家族绑定生图或编辑图用途。LLM 调用 `comfyui_draw` 时填写模型家族，插件便会走对应工作流自由生图。

调试出满意的底模、LoRA、画幅和采样参数后，可以另存为配方。`comfyui_recipe_draw` 只需配方名和本次提示词，即可快捷复用整套参数。配方引用模型家族；工作流选择和节点槽位由家族及工作流档案统一管理，因此更换家族工作流无需逐份修改配方。

本插件完全开源免费，欢迎 Issue 和 PR。

## ✨ 核心功能

| 功能 | 说明 |
|:---|:---|
| **按模型家族自由生图** | LLM 填写家族名，插件自动选择工作流；底模、LoRA、画幅和采样参数可自由调整 |
| **配方快捷生图** | 输入配方名和本次提示词，复用实验好的模型、LoRA 和采样参数 |
| **后台生成队列** | 工具返回排队确认，等待期间继续聊天；完成后自动发送图片并由 LLM 回复 |
| **Workflow Studio** | 分别管理工作流 JSON、家族绑定与节点映射，以及静态配方和试画 |
| **查一下再画** | 角色、画师、底模、LoRA 不确定时先查短列表；LoRA 支持 style / character 等分类 |

## 开发目录

根目录的 `main.py` 是 AstrBot 插件入口，内部实现统一放在 `comfyflow/`：

| 目录 | 职责 |
|:---|:---|
| `clients/` | ComfyUI HTTP/WebSocket、Civitai、Booru 和 AnimaDex 客户端 |
| `workflows/` | 模板构建、节点映射、图校验、模型家族及编辑路由 |
| `resources/` | 本地资源分类、角色规范词和参考来源查询 |
| `storage/` | 图片缓存与配方、生成历史存储 |
| `generation/` | 后台队列、完成回复生命周期、发送隔离及输出恢复 |
| `studio/` | 工作台 API 与配置选项刷新 |
| `tools/` | 生图、编辑、配方、查询、任务和诊断工具；共用参数解析单独维护 |

模块使用包内相对导入；热重载只更新当前插件的内部包。工作流、图片、配方与模型缓存继续使用原插件数据目录，安装目录由 `comfyflow/paths.py` 定位。开发测试脚本仍位于 `scripts/`，工作台静态资源仍位于 `pages/workflow-editor/`。

## 图片缓存与清理

Bot 生图、输出下载和工作台试画统一将原图缓存到 AstrBot 的 `data/plugin_data/astrbot_plugin_kuyo_comfyflow/output/`（由 `StarTools.get_data_dir` 定位）。缓存按图片内容哈希命名，保留原始字节，同内容复用并刷新保留时间，同名不同内容分别保存。旧版 output 图片也纳入清理。

管理员可发送：

- `/comfyui_cache` 或 `/comfyui_cache status`：查看缓存数量、容量和目录。
- `/comfyui_cache expired`：立即按保留期限和容量上限清理。
- `/comfyui_cache clear`：清理图片缓存。

配置项 `image_cache_auto_clean` 默认开启，启动及每小时执行清理；`image_cache_days` 默认 7 天；`image_cache_max_mb` 默认 1024 MiB。天数或容量设为 0 可关闭对应限制。清理只处理 output 直属图片，保留配方、工作流和生成历史。待通知、发送中和送达待确认任务引用的图片始终保留；自动清理及 `expired` 也保留各会话最近输出和工作台仍缓存的结果，供继续编辑或重复查询。手动 `clear` 可清理最近输出，但仍保护待交付任务和最近 5 分钟保存或复用的文件，因此容量可能暂时超限。历史里已清理图片的本地路径会失效。

生成、配方与工作台试画保留模板输出节点的前缀和节点 ID；图片编辑按来源文件内容哈希命名远端输入，重复使用同一图片会得到相同输入文件名。仅修改映射参数及种子，保留其余节点、连线和固定加速 LoRA，让 ComfyUI 可以复用未变化的上游计算。实际缓存命中由 ComfyUI 的缓存策略和自定义节点行为决定。模型查询使用快速资源清单，资源刷新缺少某类定义时保留已有清单及元数据；端点明确返回空选项时才移除该类资源。轻量 LoRA 目录补全缺少字段时沿用已有元数据与可信触发词，正常过期刷新仍更新目录；强制刷新可重新读取完整元数据。取回已有本地图片时不重新下载或改写字节，只刷新保留时间，与自动清理共享文件锁。

## 生图上下文与工作流适配

默认 `llm_tool_mode=basic` 启用自由生图、配方生图、查询和任务管理；配置编辑工作流后也启用图片编辑。家族目录提供 `prompt_style`。生成与编辑工具先返回排队确认；完成事件包含发送状态、原请求、生成摘要和内部任务标记，完整参数继续保存到生成历史。本地路径可通过任务管理取回。配方生图只需提示词及可选配方名。

## 后台生成队列

自由生图、配方生图和图片编辑统一进入插件队列。校验并准备好工作流后，工具返回排队确认，当前 LLM 可以自然回应并继续聊天。后台按队列顺序提交 ComfyUI，通过 WebSocket 和 history 跟踪完成，下载最终输出并发送到原会话，再携带原对话历史及人格调用一次 LLM 告知结果。任务编号仅用于内部追溯，聊天回复不播报编号。

完成通知使用原平台事件的独立副本，保留平台能力和插件状态，重置旧回复、流式结束和停止标记；完成事件从这次后台回复开始重新计时，避免将 GPU 排队计入清理插件的 LLM 超时。原消息时间保留在完成事件的元数据中，恢复后的任务使用独立完成事件。回复依次经过 `on_waiting_llm_request`、`on_llm_request`、`on_agent_begin`、`on_llm_response`、`on_agent_done`、`on_decorating_result` 及 AstrBot 的结果装饰和发送阶段，遵循本会话启用的插件、分段回复、内容清理及消息格式设置。插件对响应、运行上下文和事件结果的替换或清空都会保留，自行发送的内容不重复发送。通过共享 Context 直接发送的分段回复按异步任务和会话隔离捕获，历史只记录实际送达的完整文本。

- `comfyui_job(action="status", task_id="C000001")` 查询状态、图片数量和通知状态；也兼容 `prompt_id`，省略编号查询本会话最近任务。本地图片路径通过 `action="fetch"` 按需取回。
- 排队回包和后台完成事件向模型提供内部任务编号，并标注“编号仅为内部标记，不用向用户声明或复述。”原请求、生成指令、模型家族、配方和工作流摘要用于关联具体任务；原请求最多 160 字符、生成指令最多 240 字符，内容相同时只展示一次，完整信息仍保存在任务记录中。队列列表只附一次内部使用说明；状态回包只返回图片数量，路径由 `fetch` 按需取回。后台请求保留原历史消息；开始 hook 只修改本次提示词时不会重新序列化历史，修改历史时也保留未变的前缀。发给用户的普通回复和后台完成回复过滤编号及说明，种子、节点 ID 和图片数量保留。全局队列诊断仅返回数量。
- 完成通知日志按任务编号记录图片发送数量、LLM 实际调用、会话/对话/provider、返回耗时与文本长度，以及回复处理结果；关闭通知、会话切换或 hook 拦截有明确的跳过原因，调用和发送失败继续记录错误。
- `comfyui_job(action="queue")` 查看本会话正在排队或执行的任务；`comfyui_system_stats` 的全局队列包含其他客户端提交的任务。
- `comfyui_job(action="fetch", task_id="C000001")` 优先返回已有本地图片，缺失时补下载；也可填写 `prompt_id` 取回后端历史输出。已登记到其他会话的任务不会回退为外部任务，无论使用哪个编号字段都不能跨会话取图。补下载返回本地路径，不重复触发完成通知。
- `comfyui_job(action="cancel", task_id="C000001")` 取消待提交任务或请求 ComfyUI 中断已提交任务。队列任务的 `wait` 操作立即返回状态，完成时自动通知。

任务登记保存在插件数据目录的 `generation_jobs.json`。重载后继续处理待提交任务、跟踪已有 prompt_id、回收尚未下载的图片及发送待通知结果。等待超时会继续跟踪原任务。提交响应丢失或插件在提交期间退出时标记“提交结果待确认”，避免自动重复生成。发送期间退出时通知状态标记为 `unknown`，可查询结果并人工核实送达情况，避免自动重复发图。图片下载失败和通知失败分别保留状态及原 prompt_id。最近 200 个终态且通知已处理的任务保留在登记文件中。

默认队列容量 `generation_queue_limit=32`，同时跟踪任务数 `generation_queue_concurrency=1`（最多 4）；GPU 调度继续由 ComfyUI 负责。`generation_notify_llm=true` 开启完成后的 LLM 回复，`generation_notify_timeout=60` 设置通知超时。关闭 LLM 通知或 LLM 失败时，简短结果通知仍经过相同的装饰及发送阶段，本地路径可在任务管理中查询。原对话已切换或删除时通知原会话，并保留任务归属。完成通知通过 AstrBot 会话锁协调正常聊天；较旧 SDK 缺少该锁时仅保证插件通知之间串行。原平台须支持主动消息，平台送达失败会记录到通知状态。

Workflow Studio 的提交和浏览器轮询流程继续使用现有接口。

已配置或保存的工作流映射直接使用，自动检测仅作为无映射时的回退。主提示词节点失效时会提示重新确认映射，避免消耗一次无效生成。图片返回优先选择最终 output；仅有 PreviewImage 等预览输出时使用其真实 type 与 subfolder 下载。

## 🚀 快速开始

### 1. 安装

将插件目录放入 AstrBot 的 `data/plugins/` 下，在 WebUI 重载插件列表。

依赖：`httpx>=0.27`（见 `requirements.txt`，AstrBot 自带）。

### 2. 配置 ComfyUI 地址

在插件配置中确认 `comfyui_host`（默认 `127.0.0.1:8188`，本机）。

### 3. 添加模型家族

先在 Workflow Studio 导入工作流 JSON。随后在插件配置的“生图家族与工作流”中添加家族：

- `name`：暴露给 LLM 的家族名，例如 `anima`、`krea2`、`flux`
- `workflow`：这个家族使用的工作流
- `prompt_style`：`danbooru`、`natural` 或 `auto`
- `description`：适用画面或用途，会随家族清单提供给 LLM

需要图片编辑时，在“独立编辑工作流路由”配置编辑路由名并选择已导入的工作流，也可以在 Workflow Studio 的工作流页直接创建或修改编辑路由。编辑路由与生图家族相互独立；LLM 调用 `comfyui_edit` 时使用 `edit_workflow` 选择路由。工作流页还可编辑 API JSON、独立保存节点映射。工作流、家族和默认配方等选择框由本地插件数据填充；配置页保留按工作流的节点下拉与通用手填条目。导入新工作流后重新打开配置页即可看到新选项。

Qwen Image 2.1 可分别配置 T2I 与编辑工作流：生图家族 `qwen` 绑定 T2I 模板，独立编辑路由 `qwen-edit` 绑定编辑模板。LLM 调 `comfyui_draw(model_family="qwen")` 生成新图，调 `comfyui_edit(edit_workflow="qwen-edit")` 修改已有图片；两条入口各走自己的工作流，不需要把编辑路由登记为生图家族。T2I 分辨率选择器映射 `aspect_ratio` 与 `megapixels`，例如方图 2K 约为 4.0 MP。编辑时省略 `resolution` 会保留来源图原始尺寸，适合在原图上局部修改；传 `resolution` 会按来源图比例缩放画布。也可以传 `custom_size=true` 启用工作流中的分辨率选择器，或直接传 `width`/`height` 指定新画布；只指定一边时另一边按来源图比例计算。画布宽高写入映射的 `size` 节点（如 `EmptyLatentImage.width/height`）；工作流有直接 `resolution` 输入时也会同步写入。Qwen 多图编辑工作流里的 `images.image_1`、`images.image_2` 会自动识别并按消息附件顺序填入；附件多于工作流输入时用 `image_indices` 选择。无附件时可通过 `image_path` / `image_paths` 指定本插件输出目录或 AstrBot `data/temp` 中的图片。请在 Workflow Studio 为尺寸输入保存节点映射；显式传参但缺少所需映射时，工具会提示补齐。配置编辑工作流时，先在 Workflow Studio 从 ComfyUI 历史导入 API 工作流，再创建独立路由并绑定模板。智能识别会沿最终编辑分支选择提示词、采样、模型及来源图片；无法唯一判定时留空。点「重新识别」查看建议，核对后点「保存映射」；配置页手动映射始终优先。确认「用户要画的内容」指向 `TextEncodeQwenImageEdit`，「编辑来源图片」指向该分支的 `LoadImage`。当前消息或引用消息附图时，`comfyui_edit` 会自动上传并填入对应图片节点；也可以使用本插件此前回执中的本地图片路径。只有一条编辑路由时可省略 `edit_workflow`。旧版 `edit_families` 和 `model_families[].edit_workflow` 配置会作为兼容来源读取。

### 4. 分别保存工作流与配方

打开插件页面 Workflow Studio：

1. 在**工作流**页选择模板和生图/编辑图用途。生图工作流绑定到生图家族；编辑工作流绑定到独立编辑路由。需要时修改 API JSON，确认 **用户要画的内容**、**编辑来源图片**、**出图采样** 等槽位后点「保存映射」。槽位归工作流所有。
2. 在**静态配方**页选择家族，填写底模、LoRA、画幅和采样参数，起名并保存。配方文件只记录家族与参数；导入、编辑、绑定工作流及保存节点映射均不会改写配方。只有显式旧版默认参数的首次迁移可能自动创建一条兼容配方。

自动识别优先从最终输出沿连线回溯，区分正向与负向条件，忽略闲置节点及独立预览分支，并按上下游顺序建议双采样节点。支持服务端节点定义声明的自定义输出节点，以及 Qwen 同一编码节点的 `prompt` / `negative_prompt`。多个独立出图分支或多个候选节点无法唯一确定时留给手工映射；配置映射和已保存档案继续优先。单独的 Model-only LoRA 加载器需明确映射后才归入可选 LoRA 槽位。
3. 在右侧使用已保存的配方试画。表单有新修改时先保存配方，再试画已保存的参数。

然后就可以对机器人说：

| 你说 | 机器人做什么 |
|:---|:---|
| 用 anima 家族画一个站在街上的女孩 | 自由生图，使用 anima 对应工作流 |
| 用立绘配方画一个全身女孩 | 快捷复用立绘配方，只写入新提示词 |
| 用 anima 家族，换成喵喵底模并加上 zoda | 在家族工作流中按本次参数自由生成 |
| 画成水彩风格 | 可查询风格 LoRA，按用途和模型适用信息选用，并填写已知触发词 |
| 画师用 xxx | 只改画师 |
| 精细一点 | 才动步数 |
| 把这次参数记住，叫日常 | 将自由生图的实际底模、LoRA、画幅与采样参数存成新配方 |

配置项 `llm_tool_mode=full` 才会把旧的调试工具暴露给模型。

## ⚙️ 配置（`_conf_schema.json`）

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| `comfyui_host` | `127.0.0.1` | ComfyUI 主机 IP（自填） |
| `comfyui_port` | `8188` | ComfyUI 端口 |
| `comfyui_timeout` | `300` | 单次结果等待窗口（秒），后台超时后继续跟踪原任务 |
| `generation_queue_limit` / `generation_queue_concurrency` | `32` / `1` | 含待通知任务的队列容量 / 同时跟踪的任务数（最多 4） |
| `generation_notify_llm` / `generation_notify_timeout` | `true` / `60` | 完成后调用 LLM 回复 / 通知 LLM 超时（秒） |
| `model_cache_ttl` | `600` | 模型清单缓存刷新间隔（秒），0 表示每次强制同步 |
| `lora_manager_enabled` | `true` | 复用 ComfyUI LoRA Manager 的分类、标签、用途说明、推荐权重和触发词；未安装时自动跳过 |
| `model_families` | `anima → anima-v3` | 生图家族配置；工作流可从本地已导入模板下拉选择 |
| `edit_workflows` | 空 | 独立编辑路由；每条记录提供路由名并选择本地编辑工作流，配置后启用 `comfyui_edit` |
| `workflow_node_mappings` | 空 | 可重复添加的工作流节点映射；填写各槽位的节点 ID，优先于 Workflow Studio 档案和自动检测 |
| `default_recipe` | `默认` | 用户只说「画一张」时用的配方；在工作台配方列表点「设为默认」自动写入，也可直接填配方名 |
| `llm_tool_mode` | `basic` | `basic` 暴露自由生图、图片编辑（已配置时）、配方生图、查询和任务管理；`full` 打开诊断工具 |
| `allow_llm_unsafe_tools` | `false` | 是否允许 LLM 读取本地图片上传、释放显存和删除配方；默认关闭 |
| `default_workflow` / `node_slots` | 旧版兼容 | `model_families` 为空时使用；新版槽位在 Workflow Studio 工作流页按工作流保存 |
| `danbooru_base_url` | `https://danbooru.donmai.us` | danbooru 接口地址（国内可换镜像） |
| `gelbooru_base_url` | `https://gelbooru.com` | gelbooru DAPI 地址（镜像可换） |
| `civitai_api_key` | 空 | civitai API Key（可选，以你的身份调用） |
| `animadex_mcp_url` / `animadex_timeout` | `http://127.0.0.1:11451/mcp` / `8.0` | AnimaDex 角色库 MCP 端点与查询超时；不用可忽略 |
| `default_*` 生成参数 | 旧版兼容 | 新版配置页隐藏；已有值用于迁移为默认配方，日常参数请保存在配方中 |

自由生图的参数优先级为本次 LLM 参数 > 工作流原值。配方生图的优先级为本次提示词/种子/画幅方向 > 配方参数 > 工作流原值。

## 🖼️ 工作流模板

- 模板源：插件数据目录 `data/plugin_data/astrbot_plugin_kuyo_comfyflow/workflows/`（首次部署需从原环境拷贝，插件包不含模板文件）。anima-v3 使用 rgthree（Power Lora Loader、Image Comparer）、Comfyroll（CR Prompt Text、JoinStringMulti）与 DanbooruText 自定义节点，需自行安装。
- 兼容旧模板：`nagato-anima`、`anima-v2`，从 AstrBot `data/skills/anima-comfyui/references/` 目录加载（不存在时报错提示）。
- 模板即事实来源：模型 / 步数 / cfg / 采样器默认值都在模板里，代码不写死。

## 📂 数据目录

`data/plugin_data/astrbot_plugin_kuyo_comfyflow/`

- `comfyui_models.json`：模型清单缓存
- `workflows/`：自定义工作流模板
- `workflows/_ui/`：工作流画布布局与控件快照，和可执行 API JSON 同名保存
- `workflow_profiles.json`：按工作流保存的共享节点槽位
- `recipes/`：只保存模型家族与生成参数的快捷配方
- `output/`：生成的图片文件
- `generation_jobs.json`：持久化任务编号、归属、生成与通知状态

## 🤖 LLM 工具

插件随包提供 `comfyui-direct-generation` Skill，可在 AstrBot Skills 管理页启用。它按所选家族读取 Anima、Krea 2、MOSS-Image、Qwen-Image-2.1 的提示词参考，并指导模型选择生图、编辑和配方工具。

默认 `llm_tool_mode=basic` 启用以下工具；`comfyui_edit` 仅在至少配置一个可用的独立编辑工作流后启用：

| 工具 | 说明 |
|:---|:---|
| `comfyui_draw` | 自由生图；必填 `model_family` 和 `prompt`，按家族选择工作流，可填写底模、LoRA、尺寸、分辨率选择器比例与目标 MP，并可另存配方 |
| `comfyui_edit` | 用独立 `edit_workflow` 路由修改当前或引用消息中的一张或多张图片；可选分辨率与自定义画布参数 |
| `comfyui_recipe_draw` | 快捷配方生图；填写本次 `prompt`，可选 `recipe`、`size` 和 `seed`，其余参数来自配方 |
| `comfyui_lookup` | 统一查询角色、画师、作品系列与本地资源；支持家族筛选、分页、刷新、目录搜索和详细 tag 统计 |
| `comfyui_job` | 本会话任务状态、取消、队列，以及 `action=fetch` 复用或补下载输出图片 |

`llm_tool_mode=full` 额外启用以下工具，其中标注为需额外开关的工具还要求 `allow_llm_unsafe_tools=true`：

| 工具 | 说明 |
|:---|:---|
| `comfyui_civitai_search` | 搜参考图并返回生成配方（模型/prompt/负向/sampler/steps/cfg/seed） |
| `comfyui_model_info` | 查模型/LoRA 元数据与触发词（`source`=local / civitai） |
| `comfyui_system_stats` | 查询设备、显存、系统内存及全局后端队列；`include_queue=false` 仅查系统 |
| `comfyui_nodes` | 搜索节点类或查询节点输入输出结构 |
| `comfyui_validate_workflow` | 提交前检查工作流节点和必填输入 |
| `comfyui_recipe` | 保存、列出、加载配方；`action` 必填，保存时用 `description` 写用途说明并引用模型家族；删除动作需额外开关 |
| `comfyui_upload_file` | 上传本地图片至 ComfyUI 的 input 目录，需额外开关 |
| `comfyui_free_memory` | 请求卸载模型、释放显存，需额外开关 |

默认不会把 `comfyui_upload_file`、`comfyui_free_memory` 交给 LLM，且 `comfyui_recipe` 的删除动作也要求在 Workflow Studio 手动完成。确有需要时，先开启 `allow_llm_unsafe_tools`。

底模与 LoRA 按 `model_family` 查询，支持 `anima`、`krea2`、`sdxl`、`flux`、`illustrious`、`pony` 等家族，以及配置规则中的自定义家族。例如：

```text
comfyui_lookup(type="lora", model_family="anima", query="style")
comfyui_lookup(type="lora", model_family="krea2", query="水彩")
comfyui_lookup(type="model", model_family="sdxl")
comfyui_lookup(type="model", query="miao miao harem")
comfyui_lookup(type="lora", model_family="anima", limit=5, offset=0)
comfyui_lookup(type="model", model_family="sdxl", limit=5)
comfyui_lookup(type="all", query="miaomao")
comfyui_lookup(type="clip", query="clip")
comfyui_lookup(type="folder", folder="checkpoints", query="base")
comfyui_lookup(type="all", refresh=true)
comfyui_lookup(type="artist", query="artist_name", source="danbooru", detail=true)
comfyui_lookup(type="copyright", query="作品名", source="animadex", page=2)
```

默认每页 5 项，最多 10 项；有后续结果时返回 `next_offset`。省略家族和 `query` 时给出家族数量，`type=all` 给资源数量摘要。填写 `query` 后可省略家族按文件名跨家族搜索；`type=all` 会跨资源类别搜索。名称搜索忽略大小写与常见分隔符，找不到直接命中时返回少量近似名称并提示核对。搜索结果标明家族；选用底模或 LoRA 时仍需匹配生图工作流家族。`type=model` 表示底模。查询底模/LoRA 时可以省略 `query`，角色/画师查询仍需关键词。`comfyui_model_info` 也支持家族参数；底模元数据用 `types="Checkpoint"`，LoRA 用 `types="LORA"`。

归类优先级：插件配置中的 `resource_family_rules` → 基础模型元数据 → 目录/文件名推断。配置规则支持大小写及斜杠归一化，例如 `kind=lora, family=anima, pattern=Anima/*`；无特征的底模可用完整文件名配置。家族名应与生图路由一致，自定义路由需添加对应规则。SDXL、Pony、Illustrious 分别列出；FLUX Krea 与 Krea2 分开归类。

默认浏览结果排除未知家族；名称查询准确命中文件名或去扩展名的名称时会显示未知资源，并提示兼容性未确认。使用 `model_family="unknown"` 或 `include_unknown=true` 可查看其他未知项。未知资源不会通过近似名称自动推荐；核实后仍可显式填写完整文件名。生图与配方提交会拦截已知家族冲突，同名文件要求明确目录。规则和文件名推断依赖标注准确性，不构成模型张量结构验证。固定在工作流中的加速 LoRA 保持原设置。

LoRA 名称可跨家族查询；按分类、标签和用途关键词挑选 LoRA 时需指定 `model_family`。在线元数据回退只接受文件名匹配的版本，避免直接套用搜索结果首项。升级后资源缓存会重新同步，原有离线缓存仍可回退。


`comfyui_draw` 与 `comfyui_recipe` 的 `lora` 使用对象数组，例如 `[{"name":"style.safetensors","strength":0.6}]`；文件名要替换成查询结果。传入数组会覆盖工作流映射的可选 LoRA 槽位；独立加速 LoRA 保持工作流原值。省略时沿用可选 LoRA 原值，传空数组关闭映射槽位。旧版字符串输入仍由运行时解析。

选用 LoRA 时，可将查询返回的已知触发词同步填入 `trigger_words`，保留原词格式，无需用户再次提出。查询未提供触发词时可省略该字段并继续使用 LoRA。插件的绘图工具仍由调用方显式填写触发词；工作台选 LoRA 后自动填充文本框的行为保持不变。

## 🖥️ WebUI：Workflow Studio

AstrBot Dashboard → 插件页 → **Workflow Studio**，分为工作流与静态配方两页：

- 工作流页提供可拖拽、选择、连线、缩放和删除节点的画布；右侧检查器编辑节点输入，节点库来自远端 `/object_info`
- 支持导入 ComfyUI 前端工作流 JSON 并保留节点位置、连线与控件布局；执行用 API JSON 与画布快照一起保存
- 画布改动后重新编译可执行工作流；未改动时复用转换缓存。「运行当前画布」直接向配置的远端 ComfyUI 提交
- 工作流页也可编辑 API JSON、绑定生图家族或独立编辑路由，并独立保存共享槽位映射
- 配方页只保存家族与底模、LoRA、画幅、采样等静态参数
- 从已保存配方试跑，并从历史另存新配方
- 配方列表一键「设为默认」，机器人只说「画一张」时即用这套
- 模型选择：UNET / LoRA / CLIP / VAE 下拉来自自动同步清单
- 连接状态灯 + 「配方试画」：按已保存配方生成，结果内联预览，可一键中断

后端接口（`webapi.py`）：`workflows` / `workflow` / `workflow/nodes` / `workflow/import` / `workflow/save` / `workflow/run` / `workflow/profile` / `workflow/bind` / `workflow/delete` / `recipes` / `recipe/save` / `status` / `generate`。

节点画布使用 MIT 许可的 Comfy Org LiteGraph 0.17.2，构建文件随插件打包。修改画布源码后在 `pages/workflow-editor/` 运行 `npm ci && npm run build`。

## ⚠️ 注意事项

- 图片由插件通过 `event.send` 直接发送，不要重复调用 `send_message_to_user`。
- ComfyUI 不可达时生成失败，请确认 ComfyUI 已启动且网络可达。
- 提交失败时会解析 ComfyUI 400 响应，返回真实错误原因（缺节点/模型不存在等）。

## 📄 License

[MIT](LICENSE) © 2026 长门九曜


### 中文角色名 → 角色 tag 与外观参考

```python
comfyui_lookup(type="character", query="初音未来")
comfyui_lookup(type="character", query="伊蕾娜")
comfyui_lookup(type="character", query="elysia_(honkai_impact)")
```

角色查询先使用本地名字索引和配置别名定位规范 tag，再对唯一精确匹配查询 [AnimaDex](https://github.com/zetaneko/AnimaDex) 的结构化角色样例标签。不需要额外 MCP 服务，不拉取图片或帖子样本，不调用 LLM 翻译名字。同名、不同版本或部分匹配返回候选；眼色和发色有多个参考值时分列候选，按版本选用。

首次查询下载约 1 MB 的名字 JSON，来自 [Jannchie/danbooru-tag-index](https://github.com/Jannchie/danbooru-tag-index) 的 Wiki 别名选取、人工校订与错误译名排除列表。名字选取包含 LLM 辅助，覆盖有限，不能当作完整官方角色名库。网络兜底使用 Danbooru 角色 tag / Wiki 别名元数据；外观参考来自 AnimaDex 样例，可能含默认服饰。所有结果标明来源。

- `character_aliases`：可编辑别名到规范 tag 的映射，支持填写自定义外观 tag；同名可配置多行，配置优先于社区索引。
- `character_names_file`：可指定本地 JSON 替代下载。支持 `{"hatsune_miku":{"zh_hans":"初音未来","ja":"初音ミク"}}`，或 `[{"tag":"hatsune_miku","names":["初音未来"],"appearance_tags":["aqua_hair","twintails"]}]`。
- `character_reference_url`：默认 `https://animadex.net`，可换成本地 AnimaDex Web 服务；留空关闭外观联网查询。
- `character_cache_hours`：默认 168 小时，名字与外观缓存保存在插件数据目录的 `character_tags.json`；重启后可复用。
- `character_lookup_timeout`：默认每阶段 6 秒；缓存命中不等待网络。缺少名字时仍可查询英文 tag，缺少外观来源时保留角色 tag。

离线回归：`python scripts/character_lookup_test.py`，覆盖同名候选、错误译名排除、外观角色一致性、配置覆盖、并发请求复用、重启缓存和免帖子取样。


### 生图任务等待与资源准备

- `comfyui_timeout` 是任务等待上限；`comfyui_request_timeout` 单独控制 HTTP 请求超时，默认 15 秒。连接建立失败可重试提交，提交响应超时会保留“可能已提交”的提示，避免重复排队。
- `comfyui_events_enabled` 默认开启，通过一个可重连的 WebSocket 接收完成、错误和进度事件；HTTP 历史是结果依据，事件连接失效时继续轮询。等待超时会返回任务 ID，WebUI 可点「继续等待」，复用原任务。
- 生图准备使用单次资源快照，共用节点定义缓存。全量 LoRA Manager 分类目录在后台同步；选中的 LoRA 按需读取元数据，查询完整资源信息时才等待全量补全。
- 同一个 WebUI 结果的重复／并发查询复用本地原图，生成历史只写一次；结果缓存保存文件路径，不长期保留 base64 图片。
- `auto_lora_trigger_words` 默认关闭，Bot 和 WebUI 规则一致。启用后只补全未显式设置且已映射的触发词槽位；显式留空保留，训练词频与数据集标签推测不会自动注入。

回归验证：`python scripts/generation_pipeline_test.py`，覆盖提交重试、请求与等待期限、执行事件及断线回退、并发结果查询、资源准备、各生图入口的触发词优先级。

画布支持屏幕像素比例适配、节点内多行文本、Ctrl+Z / Ctrl+Y 编辑历史、完整节点搜索、可收起侧栏，以及保存重开时保留参数、布局和视角。
