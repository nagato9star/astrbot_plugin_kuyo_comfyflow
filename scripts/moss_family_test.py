"""MOSS 家族与比例换算的快速验证：detect_slots → apply_slots 比例折叠 → 家族识别。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comfyflow.workflows.slots import apply_slots, detect_slots, read_current_values
from comfyflow.resources.catalog import infer_family, resource_family
from comfyflow.storage.recipes import model_family
from comfyflow.workflows.registry import WorkflowProfileStore

WFDIR = Path(r"D:\ai\plugin test\comfyui test")
ok = True


def check(name, cond, detail=""):
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{status}] {name} {detail}")


moss = json.loads((WFDIR / "mosslora (3).json").read_text(encoding="utf-8"))
qwen = json.loads((WFDIR / "qwen.json").read_text(encoding="utf-8"))

# 1. moss 槽位检测
slots = detect_slots(moss)
print("moss slots:", json.dumps(slots, ensure_ascii=False))
check("moss prompt", slots.get("prompt", {}).get("node") == "4")
check("moss size=EmptyFlux2LatentImage", slots.get("size", {}).get("node") == "6")
check("moss sampler=KSampler", slots.get("sampler", {}).get("node") == "9")
check("moss model=UNETLoader", slots.get("model", {}).get("node") == "1")
check("moss loras=PowerLoraLoader", slots.get("loras", {}).get("node") == "12")
check("moss negative", slots.get("negative", {}).get("node") == "5")
check("moss no aspect_ratio slot", not slots.get("aspect_ratio"))

# 2. aspect_ratio 无专用槽位 → 换算宽高（面积沿用 800*1424，snap 16）
wf = json.loads(json.dumps(moss))
apply_slots(wf, slots, {"prompt": "test", "aspect_ratio": "16:9"})
ins = wf["6"]["inputs"]
area_before = 800 * 1424
ratio = ins["width"] / ins["height"]
check("moss 16:9 w%16==0", ins["width"] % 16 == 0 and ins["height"] % 16 == 0, f"{ins['width']}x{ins['height']}")
check("moss 16:9 ratio≈1.78", abs(ratio - 16 / 9) < 0.06, f"ratio={ratio:.3f}")
check("moss 16:9 area≈orig", abs(ins["width"] * ins["height"] - area_before) / area_before < 0.08)

# 3. megapixels 单独 → 保持原比例缩放
wf = json.loads(json.dumps(moss))
apply_slots(wf, slots, {"prompt": "t", "megapixels": 4.0})
ins = wf["6"]["inputs"]
check("moss 4MP area≈4e6", abs(ins["width"] * ins["height"] - 4e6) / 4e6 < 0.08, f"{ins['width']}x{ins['height']}")
check("moss 4MP ratio≈orig", abs(ins["width"] / ins["height"] - 800 / 1424) < 0.06)

# 4. aspect_ratio + megapixels 组合
wf = json.loads(json.dumps(moss))
apply_slots(wf, slots, {"prompt": "t", "aspect_ratio": "1:1", "megapixels": 2.0})
ins = wf["6"]["inputs"]
check("moss 1:1@2MP", abs(ins["width"] / ins["height"] - 1.0) < 0.08 and abs(ins["width"] * ins["height"] - 2e6) / 2e6 < 0.1,
      f"{ins['width']}x{ins['height']}")

# 5. aspect_ratio 非法格式报错
try:
    wf = json.loads(json.dumps(moss))
    apply_slots(wf, slots, {"prompt": "t", "aspect_ratio": "横的"})
    check("moss bad ratio raises", False)
except ValueError as e:
    check("moss bad ratio raises", "aspect_ratio" in str(e), str(e))

# 6. qwen 工作流同样享受换算（EmptyLatentImage, snap 8）
qslots = detect_slots(qwen)
print("qwen slots:", json.dumps(qslots, ensure_ascii=False))
check("qwen size", qslots.get("size", {}).get("node") == "7")
wf = json.loads(json.dumps(qwen))
apply_slots(wf, qslots, {"prompt": "t", "aspect_ratio": "3:2"})
ins = wf["7"]["inputs"]
check("qwen 3:2 snap8", ins["width"] % 8 == 0 and ins["height"] % 8 == 0, f"{ins['width']}x{ins['height']}")
check("qwen 3:2 ratio≈1.5", abs(ins["width"] / ins["height"] - 1.5) < 0.06)

# 7. 家族识别
check("infer moss unet", infer_family("MOSS-Image\\moss_image_sft_int8_convrot.safetensors") == "moss")
check("infer moss lora dir", infer_family("Moss\\1999v1_step00001500.safetensors") == "moss")
check("infer moss clip", infer_family("MOSS-Image\\qwen3.5_9b_int8_convrot.safetensors") == "moss")
check("infer flux vae still flux", infer_family("flux2-vae.safetensors") == "flux")
check("infer qwen still qwen", infer_family("Qwen\\qwen_image_2.1_int8_convrot.safetensors") == "qwen")
check("model_family moss", model_family("moss_image_sft_int8_convrot.safetensors") == "moss")
fam, src = resource_family("Moss\\1999v1_step00001500.safetensors")
check("resource_family moss", fam == "moss", f"{fam} via {src}")

# 8. effective() 档案语义：已存档案整体优先（配置/档案映射是人工确认的完整映射）
store = WorkflowProfileStore(Path(sys.argv[1]) / "_tmp_profiles")
wf = json.loads(json.dumps(moss))
saved = store.save("moss-test", {"prompt": {"node": "4"}, "sampler": {"node": "9"}})
assert "size" not in saved["slots"]
eff = store.effective("moss-test", wf)
check("effective keeps wholesale profile", set(eff["slots"]) == {"prompt", "sampler"})
check("effective source=profile", eff["source"] == "profile")

# 9. 启动补齐逻辑（等价于 main.py::_heal_profile_slots）：检测补缺失角色、档案值优先
from comfyflow.workflows.slots import merge_slots
merged = merge_slots(detect_slots(wf), saved["slots"])
check("heal adds size", merged.get("size", {}).get("node") == "6")
check("heal keeps saved prompt", merged.get("prompt", {}).get("node") == "4")
saved2 = store.save("moss-test2", {"prompt": {"node": "5"}})
merged2 = merge_slots(detect_slots(wf), saved2["slots"])
check("heal saved mapping wins", merged2.get("prompt", {}).get("node") == "5")

import shutil
shutil.rmtree(Path(sys.argv[1]) / "_tmp_profiles", ignore_errors=True)
print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
