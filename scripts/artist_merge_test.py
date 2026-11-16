"""验证没有画师槽位时画师串并入正面提示词的行为。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comfyflow.workflows.slots import _merge_artist_into_prompt, apply_slots


def wf_with_text(text: str = "") -> dict:
    return {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": text}}}


class ArtistMergeTest(unittest.TestCase):
    def test_merges_when_no_artist_slot(self):
        wf = wf_with_text()
        apply_slots(wf, {"prompt": {"node": "1", "field": "text"}},
                    {"prompt": "1girl, seaside", "artist": "(@ask (askzy):1.2)"})
        self.assertEqual(wf["1"]["inputs"]["text"], "1girl, seaside, (@ask (askzy):1.2)")

    def test_does_not_duplicate_existing_artist(self):
        wf = wf_with_text()
        apply_slots(wf, {"prompt": {"node": "1", "field": "text"}},
                    {"prompt": "1girl, @ask (askzy), seaside", "artist": "(@ask (askzy):0.8)"})
        self.assertEqual(wf["1"]["inputs"]["text"], "1girl, @ask (askzy), seaside")

    def test_keeps_artist_slot_when_mapped(self):
        wf = {"1": {"class_type": "CR Prompt Text", "inputs": {"prompt": "1girl"}},
              "2": {"class_type": "CR Prompt Text", "inputs": {"prompt": "old artist"}}}
        slots = {"prompt": {"node": "1", "field": "prompt"},
                 "artist": {"node": "2", "field": "prompt"}}
        apply_slots(wf, slots, {"prompt": "1girl, seaside", "artist": "(@ask (askzy):1.2)"})
        self.assertEqual(wf["1"]["inputs"]["prompt"], "1girl, seaside")
        self.assertEqual(wf["2"]["inputs"]["prompt"], "(@ask (askzy):1.2)")

    def test_empty_artist_leaves_prompt(self):
        wf = wf_with_text()
        apply_slots(wf, {"prompt": {"node": "1", "field": "text"}}, {"prompt": "1girl", "artist": ""})
        self.assertEqual(wf["1"]["inputs"]["text"], "1girl")

    def test_empty_prompt_becomes_artist(self):
        wf = wf_with_text()
        apply_slots(wf, {"prompt": {"node": "1", "field": "text"}}, {"prompt": "", "artist": "(@ask (askzy):1.2)"})
        self.assertEqual(wf["1"]["inputs"]["text"], "(@ask (askzy):1.2)")

    def test_helper_reports_merge_flag(self):
        self.assertEqual(_merge_artist_into_prompt("1girl", "(@a:1.0)"), ("1girl, (@a:1.0)", True))
        self.assertEqual(_merge_artist_into_prompt("1girl, @a", "(@a:1.0)"), ("1girl, @a", False))


if __name__ == "__main__":
    unittest.main()
