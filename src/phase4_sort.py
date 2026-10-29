"""Phase 4: 立方体ごとに上からのカメラ画像を CLIP で判定し、判定した置き場へ Franka で運ぶ.

1ラウンドで立方体 3個を運び、終わったらシーンをリセットして、ランダムに置き直した次のラウンドを始める
（Isaac Sim は開いたまま）。判定の様子は、別ウィンドウの判定モニタ（judge_viewer.py）に表示する。

実行：python src\\phase4_sort.py [--seed 配置の seed] [--rounds ラウンド数] [--rules 設定ファイル] [--device cuda|cpu] [--no-viewer]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from app_config import SIM_APP_CONFIG, parse_script_args

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None, help="立方体の配置を決める seed（省略するとランダム）")
parser.add_argument("--rounds", type=int, default=3, help="ラウンド数（既定: 3）")
parser.add_argument("--rules", default=None, help="分別ルールの YAML（既定: config/sorting_rules.yaml）")
parser.add_argument("--device", default=None, help="CLIP を動かすデバイス（cuda / cpu）。設定ファイルより優先")
parser.add_argument("--no-viewer", action="store_true", help="判定モニタのウィンドウを開かない")
args = parse_script_args(parser)

from isaacsim import SimulationApp

# SimulationApp は他の isaacsim モジュールより先に起動する必要がある
simulation_app = SimulationApp(SIM_APP_CONFIG)

import numpy as np
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from PIL import Image

from classifier import DEFAULT_CONFIG, load_classifier
from judge_status import RUN_DIR, write_status
from scene import (
    BIN_SIZE,
    CROP_SIZE,
    NUM_CUBES,
    apply_layout,
    bin_slot_position,
    build_scene,
    crop_box_for_cube,
    random_layouts,
    wait_for_camera_image,
)

# ラウンドが終わってから次のラウンドを始めるまでの待ち（物理ステップ数。60Hz で約2秒）。置き終わった様子を見せる
ROUND_PAUSE_STEPS = 120

rules_path = Path(args.rules) if args.rules else DEFAULT_CONFIG
seed, layouts = random_layouts(args.seed, rounds=args.rounds)
print(f"[phase4] seed={seed} rounds={args.rounds}")
for r, layout in enumerate(layouts, start=1):
    print(f"[phase4] round {r} layout: " + ", ".join(f"{item['color']}@{item['position']}" for item in layout))
scene = build_scene(layouts[0])
classifier = load_classifier(rules_path, device=args.device)
print(f"[phase4] CLIP loaded on {classifier.device}")

# 判定モニタに渡す状態。判定・配置のたびに書き換えて状態ファイルに書き出す
status = {
    "seed": seed,
    "rules": rules_path.name,
    "round": 1,
    "rounds": args.rounds,
    "per_round": NUM_CUBES,
    "total": args.rounds * NUM_CUBES,
    "done": False,
    "classes": [
        {"label": c["label"], "name": c.get("display_name", c["label"]), "color": c.get("display_color", "#cccccc")}
        for c in classifier.classes
    ],
    "current": None,
    "history": [],
}
write_status(status)
if not args.no_viewer:
    subprocess.Popen([sys.executable, str(Path(__file__).with_name("judge_viewer.py"))])

franka = scene.franka
controller = PickPlaceController(
    name="pick_place_controller",
    gripper=franka.gripper,
    robot_articulation=franka,
)
franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)


def classify_cube(cube):
    """最新のカメラ画像から立方体を切り出し、CLIP で判定する. 判定結果・切り出し範囲・保存した画像名を返す."""
    image = Image.fromarray(wait_for_camera_image(scene))
    position, _ = cube["prim"].get_world_pose()
    box = crop_box_for_cube(scene, position)
    crop = image.crop(box).resize((CROP_SIZE, CROP_SIZE))
    prefix = f"r{round_index + 1}_{cube['name']}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    image.save(RUN_DIR / f"{prefix}_scene.png")
    crop.save(RUN_DIR / f"{prefix}_crop.png")
    return classifier.classify(crop), box, prefix


def start_round(index):
    """index 番目（0 始まり）のラウンドを始める. 2ラウンド目以降はシーンを置き直す."""
    global round_index, pending
    round_index = index
    if index > 0:
        apply_layout(scene, layouts[index])
        controller.reset()
        franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)
    for bin_name in slots_used:
        slots_used[bin_name] = 0
    pending = list(scene.cubes)
    print(f"[phase4] ===== round {index + 1}/{len(layouts)} =====")
    status["round"] = index + 1
    write_status(status)


def start_task(cube):
    """判定して置き場と置く位置を決め、運ぶ準備をする."""
    prediction, box, prefix = classify_cube(cube)
    slot = slots_used[prediction.bin]
    slots_used[prediction.bin] += 1
    place_position = bin_slot_position(scene, prediction.bin, slot)
    print(
        f"[phase4] round {round_index + 1} {cube['name']}: CLIP -> {prediction.label} ({prediction.confidence:.3f}) "
        f"-> {prediction.bin} slot {slot}"
    )
    status["current"] = {
        "cube": cube["name"],
        "scene_image": f"{prefix}_scene.png",
        "crop_image": f"{prefix}_crop.png",
        "box": list(box),
        "label": prediction.label,
        "confidence": prediction.confidence,
        "scores": prediction.scores,
        "bin": prediction.bin,
        "state": "placing",
        "in_bin": None,
    }
    status["history"].append(
        {
            "round": round_index + 1,
            "cube": cube["name"],
            "label": prediction.label,
            "confidence": prediction.confidence,
            "bin": prediction.bin,
            "state": "placing",
            "in_bin": None,
        }
    )
    write_status(status)
    controller.reset()
    return {"cube": cube, "prediction": prediction, "place_position": place_position}


def finish_task(task):
    """置いた後の位置を正解値で確かめ、結果を記録する."""
    cube, prediction = task["cube"], task["prediction"]
    final_position, _ = cube["prim"].get_world_pose()
    bin_xy = scene.bins[prediction.bin]["position"]
    in_bin = bool(np.all(np.abs(final_position[:2] - bin_xy) <= BIN_SIZE[:2] / 2))
    print(
        f"[phase4] round {round_index + 1} {cube['name']}: placed at {np.round(final_position, 3)} "
        f"in {prediction.bin}: {'OK' if in_bin else 'NG'}"
    )
    results.append(
        {
            "round": round_index + 1,
            "name": cube["name"],
            "color": cube["color"],
            "label": prediction.label,
            "confidence": prediction.confidence,
            "bin": prediction.bin,
            "in_bin": in_bin,
        }
    )
    for entry in (status["current"], status["history"][-1]):
        entry["state"] = "placed"
        entry["in_bin"] = in_bin
    write_status(status)


def print_summary():
    print(f"[phase4] ===== summary (seed={seed}) =====")
    for r in results:
        print(
            f"[phase4] round {r['round']} {r['name']} ({r['color']}): {r['label']} ({r['confidence']:.3f}) "
            f"-> {r['bin']} {'OK' if r['in_bin'] else 'NG'}"
        )
    for r in range(1, len(layouts) + 1):
        in_round = [x for x in results if x["round"] == r]
        print(f"[phase4] round {r}: placed in bin {sum(x['in_bin'] for x in in_round)}/{len(in_round)}")
    print(f"[phase4] placed in bin: {sum(r['in_bin'] for r in results)}/{len(results)}")
    status["done"] = True
    write_status(status)


slots_used = {bin_name: 0 for bin_name in scene.bins}
results = []
round_index = 0
pending = []
start_round(0)
task = None
pause_steps = 0  # ラウンドの終わりから次のラウンドまでの残りステップ数
finished = False

while simulation_app.is_running():
    if task is None and pending:
        task = start_task(pending.pop(0))
    scene.world.step(render=True)
    if not scene.world.is_playing():
        continue
    if task is None:
        if pause_steps > 0:
            pause_steps -= 1
            if pause_steps == 0:
                start_round(round_index + 1)
        continue
    cube_position, _ = task["cube"]["prim"].get_world_pose()
    actions = controller.forward(
        picking_position=cube_position,
        placing_position=task["place_position"],
        current_joint_positions=franka.get_joint_positions(),
        end_effector_offset=np.array([0.0, 0.005, 0.0]),
    )
    franka.apply_action(actions)
    if controller.is_done():
        finish_task(task)
        task = None
        if not pending:
            if round_index + 1 < len(layouts):
                pause_steps = ROUND_PAUSE_STEPS
            elif not finished:
                print_summary()
                finished = True

simulation_app.close()
