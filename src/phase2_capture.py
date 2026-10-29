"""Phase 2: シーンを組み立て、上からのカメラ画像と立方体ごとの切り出し画像を保存する.

実行：python src\\phase2_capture.py [--seed 配置の seed]
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from app_config import SIM_APP_CONFIG, parse_script_args

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None, help="立方体の配置を決める seed（省略するとランダム）")
args = parse_script_args(parser)

from isaacsim import SimulationApp

# SimulationApp は他の isaacsim モジュールより先に起動する必要がある
simulation_app = SimulationApp(SIM_APP_CONFIG)

from PIL import Image, ImageDraw

from scene import CROP_SIZE, build_scene, crop_box_for_cube, random_layouts, wait_for_camera_image

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "crops"

seed, layouts = random_layouts(args.seed)
print(f"[phase2] seed={seed} layout={layouts[0]}")
scene = build_scene(layouts[0])
image = Image.fromarray(wait_for_camera_image(scene))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
image.save(OUTPUT_DIR / "overview.png")

annotated = image.copy()
draw = ImageDraw.Draw(annotated)
labels = []
for cube in scene.cubes:
    position, _ = cube["prim"].get_world_pose()
    box = crop_box_for_cube(scene, position)
    crop_file = f"{cube['name']}.png"
    image.crop(box).resize((CROP_SIZE, CROP_SIZE)).save(OUTPUT_DIR / crop_file)
    draw.rectangle(box, outline=(255, 255, 0), width=2)
    draw.text((box[0], max(box[1] - 12, 0)), cube["name"], fill=(255, 255, 0))
    labels.append(
        {
            "name": cube["name"],
            "color": cube["color"],
            "position": [round(float(v), 4) for v in position],
            "box": list(box),
            "crop_file": crop_file,
        }
    )
annotated.save(OUTPUT_DIR / "overview_annotated.png")
(OUTPUT_DIR / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"[phase2] saved {len(labels)} crops to {OUTPUT_DIR}")
for label in labels:
    print(f"[phase2] {label['name']}: color={label['color']} box={label['box']}")

# 確認できるようウィンドウは開いたままにする（閉じると終了）
while simulation_app.is_running():
    scene.world.step(render=True)

simulation_app.close()
