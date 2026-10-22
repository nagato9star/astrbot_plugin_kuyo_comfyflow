"""按 Markdown 标题拆分文档，保留原始字节并生成按需读取的目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from bisect import bisect_right
from pathlib import Path


HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scan(lines: list[bytes]) -> tuple[list[tuple[int, int, str]], list[int]]:
    """识别标题和段落边界，跳过前置元数据及围栏代码块。"""
    headings = []
    paragraphs = []
    fence = None
    frontmatter = False
    for index, raw in enumerate(lines):
        line = raw.decode("utf-8").rstrip("\r\n")
        if index == 0:
            line = line.removeprefix("\ufeff")
            frontmatter = line == "---"
            if frontmatter:
                continue
        elif frontmatter:
            if line in {"---", "..."}:
                frontmatter = False
            continue
        if fence is not None:
            char, length = fence
            if re.fullmatch(r" {0,3}" + re.escape(char) + "{" + str(length) + r",}[ \t]*", line):
                fence = None
            continue
        match = FENCE.match(line)
        if match:
            fence = (match[1][0], len(match[1]))
            continue
        match = HEADING.match(line)
        if match:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", match[2] or "").strip()
            headings.append((index, len(match[1]), title or "未命名章节"))
        if not line.strip():
            paragraphs.append(index + 1)
    return headings, paragraphs


def boundaries(lines: list[bytes], headings, paragraphs, level: int, max_lines: int) -> list[int]:
    """优先按指定层级分章，长章再沿内部标题或段落拆开。"""
    count = len(lines)
    primary = sorted({0, count, *(index for index, depth, _ in headings if depth <= level)})
    all_headings = [index for index, _, _ in headings]
    result = {0, count}
    for start, stop in zip(primary, primary[1:]):
        cursor = start
        while max_lines and stop - cursor > max_lines:
            ceiling = cursor + max_lines
            choices = [index for index in all_headings if cursor < index <= ceiling]
            if not choices:
                choices = [index for index in paragraphs if cursor < index <= ceiling]
            if not choices:
                # 完整代码块或长段落可超过软上限，不从中间截断。
                choices = [index for index in [*all_headings, *paragraphs] if ceiling < index < stop]
                if not choices:
                    break
                end = min(choices)
            else:
                end = max(choices)
            result.add(end)
            cursor = end
        result.add(stop)
    return sorted(result)


def split_document(source: Path, output: Path, *, level: int = 2, max_lines: int = 180) -> dict:
    raw = source.read_bytes()
    raw.decode("utf-8-sig")
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise ValueError("输入文档为空。")
    if output.exists():
        raise ValueError("输出目录已存在，请使用新目录，避免覆盖已有文件。")
    headings, paragraphs = scan(lines)
    cuts = boundaries(lines, headings, paragraphs, level, max_lines)
    starts = [index for index, _, _ in headings]
    parts = []
    payloads = []
    for number, (start, stop) in enumerate(zip(cuts, cuts[1:])):
        position = bisect_right(starts, start) - 1
        title = headings[position][2] if position >= 0 else "前言与元数据"
        if position >= 0 and headings[position][0] != start:
            title += "（续）"
        slug = re.sub(r"[^\w-]+", "-", title, flags=re.UNICODE).strip("-_")[:48] or "section"
        filename = f"chapters/{number:03d}-{slug}.md"
        content = b"".join(lines[start:stop])
        parts.append({"file": filename, "title": title, "start_line": start + 1,
                      "end_line": stop, "bytes": len(content), "sha256": digest(content)})
        payloads.append(content)
    assert b"".join(payloads) == raw
    manifest = {"version": 1, "source": source.name, "bytes": len(raw), "sha256": digest(raw),
                "heading_level": level, "max_lines": max_lines, "parts": parts}
    output.mkdir(parents=True)
    (output / "chapters").mkdir()
    for part, content in zip(parts, payloads):
        (output / part["file"]).write_bytes(content)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index = ["# 章节目录", "", f"源文档：{source.name}", "",
             "按本次任务选读相关章节；正文未改写，前言、署名及许可信息保留在对应章节中。", "",
             "| 章节 | 原文行号 |", "|---|---|"]
    for part in parts:
        label = part["title"].replace("|", "\\|").replace("[", "\\[").replace("]", "\\]")
        index.append(f"| [{label}]({part['file']}) | {part['start_line']}–{part['end_line']} |")
    index.extend(["", "正文中的原始相对链接未改写；移动文档时请另行保留其依赖文件。", ""])
    (output / "index.md").write_text("\n".join(index), encoding="utf-8")
    return manifest


def read_bundle(bundle: Path) -> bytes:
    """校验各分段及合并后的摘要，正文修改会明确报错。"""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    chunks = []
    root = bundle.resolve()
    for part in manifest["parts"]:
        path = (bundle / part["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("分段路径超出文档目录。")
        content = path.read_bytes()
        if len(content) != part["bytes"] or digest(content) != part["sha256"]:
            raise ValueError(f"分段校验失败：{part['file']}")
        chunks.append(content)
    result = b"".join(chunks)
    if len(result) != manifest["bytes"] or digest(result) != manifest["sha256"]:
        raise ValueError("合并结果与原始文档摘要不一致。")
    return result


def positive(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("数值不能为负数。")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split", help="拆分 UTF-8 Markdown 并生成目录")
    split.add_argument("source", type=Path)
    split.add_argument("output", type=Path)
    split.add_argument("--level", type=int, choices=range(1, 7), default=2, help="主分章标题层级，默认 2")
    split.add_argument("--max-lines", type=positive, default=180, help="每段软行数上限，0 表示只按标题分章")
    verify = commands.add_parser("verify", help="校验分段内容与原始文档摘要")
    verify.add_argument("bundle", type=Path)
    join = commands.add_parser("join", help="校验后按顺序合并，恢复原始字节")
    join.add_argument("bundle", type=Path)
    join.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "split":
            manifest = split_document(args.source, args.output, level=args.level, max_lines=args.max_lines)
            oversized = sum(part["end_line"] - part["start_line"] + 1 > args.max_lines
                            for part in manifest["parts"]) if args.max_lines else 0
            print(f"已生成 {len(manifest['parts'])} 段：{args.output / 'index.md'}")
            if oversized:
                print(f"其中 {oversized} 段因完整代码块或长段落而超过软上限。")
        else:
            content = read_bundle(args.bundle)
            if args.command == "join":
                with args.output.open("xb") as stream:
                    stream.write(content)
                print(f"已恢复原始文档：{args.output}")
            else:
                print(f"校验通过：{len(content)} 字节，SHA-256={digest(content)}")
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(1, f"处理失败：{error}\n")


if __name__ == "__main__":
    main()
