"""验证 Markdown 分章的正文完整性、代码围栏和命令行往返。"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from split_markdown import read_bundle, scan, split_document


class SplitMarkdownTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "原始文档.md"
        self.bundle = self.root / "章节"

    def test_exact_bytes_and_metadata(self):
        raw = ("\ufeff---\r\nname: example\r\n# 元数据注释\r\n---\r\n"
               "# 文档标题\r\n\r\n署名与许可。\r\n\r\n## 第一节\r\n"
               "正文\n\n## 第二节\r\n最后一行没有换行").encode("utf-8")
        self.source.write_bytes(raw)
        manifest = split_document(self.source, self.bundle, max_lines=0)
        self.assertEqual(read_bundle(self.bundle), raw)
        self.assertEqual(self.source.read_bytes(), raw)
        self.assertEqual([part["title"] for part in manifest["parts"]],
                         ["前言与元数据", "文档标题", "第一节", "第二节"])
        first = (self.bundle / manifest["parts"][0]["file"]).read_bytes()
        self.assertTrue(first.startswith(b"\xef\xbb\xbf---\r\n"))
        title_part = (self.bundle / manifest["parts"][1]["file"]).read_bytes()
        self.assertIn("署名与许可。".encode("utf-8"), title_part)

    def test_fences_and_long_sections(self):
        code = "```python\n## 代码内标题\n" + "print('line')\n" * 15 + "```\n"
        raw = ("# 示例\n\n## 长章节\n\n### 小节 A\n\n短段落。\n\n" + code
               + "\n### 小节 B\n\n下一段。\n\n~~~text\n## 也不是标题\n~~~\n").encode()
        self.source.write_bytes(raw)
        manifest = split_document(self.source, self.bundle, max_lines=10)
        self.assertEqual(read_bundle(self.bundle), raw)
        parts = [(self.bundle / part["file"]).read_text(encoding="utf-8") for part in manifest["parts"]]
        self.assertTrue(any(code in part for part in parts))
        headings, _ = scan(raw.splitlines(keepends=True))
        self.assertFalse(any("代码内标题" in title or "也不是标题" in title for _, _, title in headings))
        self.assertGreater(len(parts), 2)

    def test_duplicate_names_and_output_protection(self):
        self.source.write_text("## 重名 / 标题\nA\n## 重名 / 标题\nB\n", encoding="utf-8")
        manifest = split_document(self.source, self.bundle)
        names = [part["file"] for part in manifest["parts"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all((self.bundle / name).is_file() for name in names))
        before = (self.bundle / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "已存在"):
            split_document(self.source, self.bundle)
        self.assertEqual((self.bundle / "manifest.json").read_bytes(), before)

    def test_tamper_detection(self):
        self.source.write_text("# 标题\n内容。\n", encoding="utf-8")
        manifest = split_document(self.source, self.bundle)
        part = self.bundle / manifest["parts"][0]["file"]
        part.write_bytes(part.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "分段校验失败"):
            read_bundle(self.bundle)

    def test_cli_roundtrip_and_overwrite_protection(self):
        raw = b"# Original\r\n\r\n## First\r\nText\r\n## Second\nMore"
        self.source.write_bytes(raw)
        script = Path(__file__).with_name("split_markdown.py")
        command = [sys.executable, "-X", "utf8", str(script)]
        for args in (["split", str(self.source), str(self.bundle), "--max-lines", "8"],
                     ["verify", str(self.bundle)]):
            result = subprocess.run([*command, *args], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
        restored = self.root / "恢复.md"
        result = subprocess.run([*command, "join", str(self.bundle), str(restored)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
        self.assertEqual(restored.read_bytes(), raw)
        result = subprocess.run([*command, "join", str(self.bundle), str(restored)], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(restored.read_bytes(), raw)
        manifest = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(part["bytes"] for part in manifest["parts"]), len(raw))


if __name__ == "__main__":
    unittest.main()
