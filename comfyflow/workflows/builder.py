"""Workflow 模板构建：加载模板并按角色覆盖节点。模板即事实来源（模型/步数/cfg/采样器等默认值都在模板里）。

2026-08-24 v2.1.0：
- 模板文件迁出插件包（插件数据目录 workflows/ 为唯一内置源，skill references 为备份源）
- 插件包不再携带任何工作流 JSON，模板即"秘制蓝图"只存 data 目录，可安全推 GitHub
- 旧模板（nagato-anima / anima-v2）仍可从 skill references 目录加载，走启发式覆盖
- LoRA 覆盖：优先 Power Lora Loader (rgthree) 插槽，其次 LoraLoaderModelOnly 链
- KSampler 参数（steps/cfg/sampler_name/scheduler/denoise/seed）可覆盖
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from astrbot.api import logger

from .slots import (
    ANIMA_DROP_NODES,
    apply_slots,
    detect_slots,
    looks_like_anima,
    normalize_lora_strength,
    parse_lora,
)

# 模板名 -> 文件名；查找顺序：插件数据目录 workflows/（custom）-> skill references 目录
WORKFLOW_TEMPLATES: dict[str, str] = {
    "anima-v3": "anima-workflow-v3.json",
    "nagato-anima": "nagato-anima-workflow.json",
    "anima-v2": "anima-workflow-v2-api.json",
}

# 画师串特征：(@画师名:权重) 括号写法
ARTIST_PATTERN = re.compile(r"\(\s*@[^)]*?:\s*[\d.]+\s*\)")
# 质量词特征
QUALITY_KEYWORDS = ("masterpiece", "best quality", "score_")
# 负向提示词特征
NEGATIVE_MARKERS = ("lowres", "worst quality")

# 提示词五段式的角色名
ROLE_ARTIST = "artist"
ROLE_QUALITY = "quality"
ROLE_MAIN = "main"
ROLE_TRIGGERS = "triggers"


class WorkflowBuilder:
    """模板解析 + 参数覆盖。None 参数一律保留模板原值（不写死）。

    模板查找顺序：自定义目录（插件数据目录 workflows/，WebUI 保存的位置）→
    skill references 目录。同名时自定义优先。插件包本身不含任何模板文件。
    """

    def __init__(
        self,
        plugin_dir: Path,
        default_workflow: str = "anima-v3",
        custom_dir: Path | None = None,
    ) -> None:
        # 插件包不携带工作流模板：唯一模板源为插件数据目录（custom），去掉内置 workflows/
        self._skill_ref_dir = plugin_dir.parent.parent / "skills" / "anima-comfyui" / "references"
        self._custom_dir = custom_dir
        self.default_workflow = default_workflow

    def custom_dir(self) -> Path:
        """自定义模板目录（WebUI 保存的工作流放这里），自动创建。"""
        if self._custom_dir is None:
            raise RuntimeError("WorkflowBuilder 未配置自定义模板目录")
        self._custom_dir.mkdir(parents=True, exist_ok=True)
        return self._custom_dir

    def _template_dirs(self) -> list[tuple[Path, str]]:
        """返回 (目录, 来源名) 列表，按优先级排序。"""
        dirs: list[tuple[Path, str]] = []
        if self._custom_dir is not None:
            dirs.append((self._custom_dir, "custom"))
        dirs.append((self._skill_ref_dir, "skill"))
        return dirs

    def source_of(self, path: Path) -> str | None:
        """返回模板路径的来源（custom/skill），未知返回 None。"""
        for base, src in self._template_dirs():
            if base == path.parent:
                return src
        return None

    # ------------------------------------------------------------------
    # 模板解析
    # ------------------------------------------------------------------

    def _resolve_template(self, workflow: str | None) -> Path:
        name = workflow or self.default_workflow
        # 直接传文件路径
        if "/" in name or "\\" in name or name.endswith(".json"):
            p = Path(name).expanduser()
            if p.is_absolute() or (p.is_file()):
                return p
        fname = WORKFLOW_TEMPLATES.get(name, f"{name}.json")
        # 注册名优先找自定义副本（WebUI 保存时按 name.json 命名）
        if name in WORKFLOW_TEMPLATES and self._custom_dir is not None:
            p = self._custom_dir / f"{name}.json"
            if p.is_file():
                return p
        for base, _src in self._template_dirs():
            p = base / fname
            if p.is_file():
                return p
        searched = "、".join(str(base) for base, _ in self._template_dirs())
        raise FileNotFoundError(f"workflow 模板不存在: {name}（已查找 {searched}）")

    def resolve_workflow_path(self, workflow: str) -> Path:
        """解析模板路径（WebUI 需要知道来源时用）。"""
        return self._resolve_template(workflow)

    def load_template(self, workflow: str | None = None) -> dict:
        """加载完整模板；各工作流的 drop_nodes 只在提交前显式应用。"""
        return self._load(self._resolve_template(workflow))

    # ------------------------------------------------------------------
    # 模板管理（WebUI 使用）
    # ------------------------------------------------------------------

    def list_templates(self) -> list[dict]:
        """列出所有可用模板：name / source(custom|skill) / node_count / title。

        name 优先用注册名（WORKFLOW_TEMPLATES 反查），自定义模板用文件名 stem。
        """
        fname_to_name = {fname: name for name, fname in WORKFLOW_TEMPLATES.items()}
        out: dict[str, dict] = {}
        for base, src in self._template_dirs():
            if not base.is_dir():
                continue
            for p in sorted(base.glob("*.json")):
                name = fname_to_name.get(p.name, p.stem)
                if name in out:
                    continue  # 优先级高的来源已收录
                try:
                    wf = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                title = None
                for node in wf.values():
                    if not isinstance(node, dict):
                        continue
                    meta = node.get("_meta") or {}
                    if meta.get("title"):
                        title = meta["title"]
                        break
                out[name] = {
                    "name": name,
                    "source": src,
                    "node_count": len(wf),
                    "title": title or "",
                }
        return list(out.values())

    def save_template(self, name: str, wf: dict) -> Path:
        """保存/覆盖自定义模板（写入插件数据目录 workflows/）。"""
        if not name or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError(f"模板名不合法: {name}（仅允许字母数字下划线连字符，最长64）")
        path = self.custom_dir() / f"{name}.json"
        path.write_text(
            json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def load_ui_template(self, name: str) -> dict | None:
        """读取画布布局；旧模板只有 API JSON 时返回 None。"""
        if not name or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            return None
        path = self.custom_dir() / "_ui" / f"{name}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) and isinstance(data.get("nodes"), list) else None

    def save_ui_template(self, name: str, ui: dict) -> Path:
        """保存 ComfyUI 前端格式，保持节点位置、分组与控件布局。"""
        if not name or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError(f"模板名不合法: {name}")
        if not isinstance(ui, dict) or not isinstance(ui.get("nodes"), list):
            raise ValueError("需要 ComfyUI 前端格式工作流")
        folder = self.custom_dir() / "_ui"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.json"
        path.write_text(json.dumps(ui, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def delete_ui_template(self, name: str) -> None:
        if name and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            (self.custom_dir() / "_ui" / f"{name}.json").unlink(missing_ok=True)

    def delete_template(self, name: str) -> None:
        """删除自定义模板；内置/技能模板不允许删除。"""
        if not name or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError(f"模板名不合法: {name}")
        path = self._custom_dir / f"{name}.json" if self._custom_dir else None
        if path is not None and path.is_file():
            path.unlink()
            self.delete_ui_template(name)
            return
        raise ValueError(f"{name} 不是自定义模板，无法删除")

    def _load(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def drop_ui_nodes(wf: dict, drop_nodes: list[str] | tuple[str, ...] = ANIMA_DROP_NODES) -> dict:
        """按调用方指定的工作流档案删除 UI/辅助节点。"""
        for nid in drop_nodes:
            wf.pop(nid, None)
        return wf

    # ------------------------------------------------------------------
    # 提示词角色识别
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_prompt_role(text: str) -> str | None:
        """把 CR Prompt Text 内容分类到五段式角色之一。"""
        t = text.strip()
        if not t:
            return ROLE_MAIN  # 空文本视为待填的主提示词
        if ARTIST_PATTERN.search(t):
            return ROLE_ARTIST
        if t.startswith("@"):
            return ROLE_TRIGGERS
        if any(k in t for k in QUALITY_KEYWORDS):
            return ROLE_QUALITY
        return ROLE_MAIN

    def _find_role_nodes(self, wf: dict) -> dict[str, str | None]:
        """返回 role -> 节点ID。优先 JoinStringMulti 拼接链，无则用旧版启发式。

        兜底字段 main_alt：无 CR Prompt Text 可作主提示词时，指向最长的正向 CLIPTextEncode。
        """
        roles: dict[str, str | None] = {
            ROLE_ARTIST: None,
            ROLE_QUALITY: None,
            ROLE_MAIN: None,
            ROLE_TRIGGERS: None,
            "main_alt": None,
        }

        joins = [
            nid
            for nid, node in wf.items()
            if node.get("class_type") == "JoinStringMulti"
        ]
        if joins:
            for jid in joins:
                node = wf[jid]
                entries: list[tuple[int, Any]] = []
                for key, val in node.get("inputs", {}).items():
                    if key.startswith("string_"):
                        try:
                            idx = int(key.split("_")[1])
                        except ValueError:
                            continue
                        entries.append((idx, val))
                pending: list[str] = []  # 模板里留空的提示词节点（画师/触发词默认空）
                seen_at = False  # 画师串可能是裸 @名字 格式，第一个 @ 串视为画师
                for _, link in sorted(entries):
                    if not isinstance(link, list) or not link:
                        continue
                    tid = str(link[0])
                    tn = wf.get(tid, {})
                    # 如果中间隔了文本处理节点（如 PromptCleaningMaid），
                    # 顺着它的 string/text 输入往下追到真正的 CR Prompt Text
                    if tn.get("class_type") != "CR Prompt Text":
                        for _ in range(5):  # 最多追5层
                            sub = tn.get("inputs", {})
                            ref = sub.get("string") or sub.get("text")
                            if isinstance(ref, list) and ref:
                                tid = str(ref[0])
                                tn = wf.get(tid, {})
                                if tn.get("class_type") == "CR Prompt Text":
                                    break
                            else:
                                break
                        if tn.get("class_type") != "CR Prompt Text":
                            continue
                    text = str(tn.get("inputs", {}).get("prompt", ""))
                    if not text.strip():
                        pending.append(tid)
                        continue
                    role = self._classify_prompt_role(text)
                    if (
                        role == ROLE_TRIGGERS
                        and not seen_at
                        and roles[ROLE_ARTIST] is None
                    ):
                        role = ROLE_ARTIST  # 第一个裸 @ 串：画师（@名字 无权重写法）
                    if role == ROLE_ARTIST:
                        seen_at = True
                    if role and roles[role] is None:
                        roles[role] = tid
                # 空文本节点按槽位顺序补位：已知是多段模板（质量/触发词已就位）时
                # 第一个空槽是画师，其余按 主提示词 -> 质量 -> 触发词 顺序补齐。
                # 全空模板（简单单提示词）则全部落到主提示词。
                multi_role = (
                    roles[ROLE_QUALITY] is not None or roles[ROLE_TRIGGERS] is not None
                )
                for tid in pending:
                    if multi_role and roles[ROLE_ARTIST] is None:
                        roles[ROLE_ARTIST] = tid
                    elif roles[ROLE_MAIN] is None:
                        roles[ROLE_MAIN] = tid
                    elif multi_role and roles[ROLE_TRIGGERS] is None:
                        roles[ROLE_TRIGGERS] = tid
            return roles

        # 旧模板（无 JoinStringMulti）：启发式
        candidates: list[tuple[str, str]] = []
        for nid, node in wf.items():
            if node.get("class_type") != "CR Prompt Text":
                continue
            candidates.append((nid, str(node.get("inputs", {}).get("prompt", ""))))
        seen_at = False
        for nid, text in candidates:
            role = self._classify_prompt_role(text)
            if role == ROLE_TRIGGERS and not seen_at and roles[ROLE_ARTIST] is None:
                role = ROLE_ARTIST  # 兼容裸 @名字 画师写法
            if role == ROLE_ARTIST:
                roles[ROLE_ARTIST] = nid
                seen_at = True
                break
        others = [(nid, t) for nid, t in candidates if nid != roles[ROLE_ARTIST]]
        if others:
            roles[ROLE_MAIN] = max(others, key=lambda x: len(x[1]))[0]
        else:
            best = None
            for nid, node in wf.items():
                if node.get("class_type") != "CLIPTextEncode":
                    continue
                text = str(node.get("inputs", {}).get("text", ""))
                if any(m in text for m in NEGATIVE_MARKERS):
                    continue
                if best is None or len(text) > best[1]:
                    best = (nid, len(text))
            if best:
                roles["main_alt"] = best[0]
        return roles

    @staticmethod
    def _find_negative_node(wf: dict) -> str | None:
        """负向提示词节点：含 lowres / worst quality 的 CLIPTextEncode。"""
        for nid, node in wf.items():
            if node.get("class_type") != "CLIPTextEncode":
                continue
            t = str(node.get("inputs", {}).get("text", ""))
            if any(m in t for m in NEGATIVE_MARKERS):
                return nid
        return None

    def _sync_danbooru_text(self, wf: dict, artist_text: str) -> None:
        """anima-v3 里 DanbooruText 镜像画师串，随画师覆盖同步。"""
        for node in wf.values():
            if node.get("class_type") == "DanbooruText":
                t = str(node.get("inputs", {}).get("text", ""))
                if "@" in t:
                    node["inputs"]["text"] = artist_text
                    return

    # ------------------------------------------------------------------
    # LoRA 覆盖
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_lora(lora: Any) -> list[dict]:
        """Use the shared LoRA parser so generate/draw accept identical input."""
        return parse_lora(lora)

    def _collect_lora_chain(self, wf: dict) -> list[str]:
        """按模型链顺序收集模板中的 LoraLoaderModelOnly 节点ID（从底模开始追踪）。"""
        base_id = None
        for nid, node in wf.items():
            if node.get("class_type") == "UNETLoader":
                base_id = nid
                break
        if base_id is None:
            return []
        chain: list[str] = []
        cur = base_id
        for _ in range(50):  # 防死循环
            nxt = None
            for nid, node in wf.items():
                if nid in chain:
                    continue
                ins = node.get("inputs", {})
                if (
                    node.get("class_type") == "LoraLoaderModelOnly"
                    and isinstance(ins.get("model"), list)
                    and ins["model"][0] == cur
                ):
                    chain.append(nid)
                    nxt = nid
                    break
            if nxt is None:
                break
            cur = nxt
        return chain

    def _apply_lora_power_slots(self, wf: dict, parsed: list[dict]) -> None:
        """rgthree Power Lora Loader 插槽覆盖：数组顺序映射 lora_1..lora_N。"""
        power_id = None
        for nid, node in wf.items():
            if node.get("class_type") == "Power Lora Loader (rgthree)":
                power_id = nid
                break
        if power_id is None:
            return False
        # 这个映射只拥有 Power Loader 内的可选插槽。工作流中的其它 LoRA
        # 节点可能是 Anima 双采样所需的固定加速分支，必须保留模板原值。
        ins = wf[power_id]["inputs"]
        slots = sorted(
            (k for k in ins if k.startswith("lora_") and isinstance(ins[k], dict)),
            key=lambda k: int(k.split("_")[1]),
        )
        for i, slot in enumerate(slots):
            if i < len(parsed):
                spec = parsed[i]
                ins[slot] = {
                    "on": True,
                    "lora": str(spec.get("name") or "").strip(),
                    "strength": normalize_lora_strength(spec.get("strength", 0.8)),
                }
            else:
                ins[slot]["on"] = False  # 多余的插槽关掉，保留原 lora 名便于恢复
        if len(parsed) > len(slots):
            logger.warning(
                f"[ComfyUIDirect] 传入 {len(parsed)} 个 LoRA，超过 Power Lora Loader "
                f"的 {len(slots)} 个插槽，多余的已忽略"
            )
        return True

    def _apply_lora_chain(self, wf: dict, parsed: list[dict]) -> None:
        """旧模板的 LoraLoaderModelOnly 链覆盖：数组顺序映射链节点。"""
        chain = self._collect_lora_chain(wf)
        if len(parsed) > len(chain) and chain:
            # 动态扩容：在链尾追加新 LoraLoaderModelOnly 节点
            prev = chain[-1]
            tail_consumers = [
                nid
                for nid, node in wf.items()
                if isinstance(node.get("inputs", {}).get("model"), list)
                and node["inputs"]["model"][0] == prev
            ]
            for j in range(len(parsed) - len(chain)):
                spec = parsed[len(chain) + j]
                max_id = max(int(n) for n in wf if n.isdigit())
                new_id = str(max_id + 1)
                wf[new_id] = {
                    "inputs": {
                        "model": [prev, 0],
                        "lora_name": str(spec["name"]),
                        "strength_model": normalize_lora_strength(spec.get("strength", 0.8)),
                    },
                    "class_type": "LoraLoaderModelOnly",
                }
                chain.append(new_id)
                prev = new_id
            # 下游引用改挂到新的链尾
            for nid in tail_consumers:
                wf[nid]["inputs"]["model"] = [prev, 0]
        for i, nid in enumerate(chain):
            if i < len(parsed):
                spec = parsed[i]
                if "name" in spec:
                    wf[nid]["inputs"]["lora_name"] = str(spec["name"])
                if "strength" in spec:
                    wf[nid]["inputs"]["strength_model"] = normalize_lora_strength(spec["strength"])
            else:
                # 链上多余的 LoRA 权重归零（等效禁用）
                wf[nid]["inputs"]["strength_model"] = 0.0

    # ------------------------------------------------------------------
    # 构建入口
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        workflow: str | None = None,
        prompt: str,
        artist: str | None = None,
        trigger_words: str | None = None,
        quality: str | None = None,
        negative_prompt: str | None = None,
        model: str | None = None,
        lora: Any = None,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        sampler_name: str | None = None,
        scheduler: str | None = None,
        denoise: float | None = None,
        prefix: str | None = None,
    ) -> dict:
        """加载模板并覆盖参数；None 表示保留模板值。

        有配方槽位时请走 apply_slots。本方法用自动检测兼容旧调用/冒烟测试。
        """
        path = self._resolve_template(workflow)
        wf = self._load(path)
        slots = detect_slots(wf)
        values: dict[str, Any] = {"prompt": prompt}
        for key, val in (
            ("artist", artist),
            ("trigger_words", trigger_words),
            ("quality", quality),
            ("negative", negative_prompt),
            ("model", model),
            ("lora", lora),
            ("width", width),
            ("height", height),
            ("seed", seed),
            ("steps", steps),
            ("cfg", cfg),
            ("sampler_name", sampler_name),
            ("scheduler", scheduler),
            ("denoise", denoise),
        ):
            if val is not None:
                values[key] = val
        drop_nodes = list(ANIMA_DROP_NODES) if looks_like_anima(wf) else []
        apply_slots(wf, slots, values, prefix=prefix, drop_nodes=drop_nodes)
        return wf
