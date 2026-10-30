"""言語指示タスク：チャットの指示（例：「赤の箱を緑へ」）に合う物体を CLIP で探し、Franka で指定の置き場へ運ぶ.

机の上には赤・青の箱と円柱が1個ずつランダムな位置にあり、置き場は赤・青・緑・黄の4か所（固定の位置）。
指示は別ウィンドウのチャット（instruct_ui.py）から受け取り、LLM（OpenAI API）か簡単なルールで
「どの物体を・どの置き場へ」に変換したものが届く。物体ごとに、指示された種類である確率（信頼度）を CLIP で求め、
いちばん高い物体を運ぶ。ただしその物体自身の判定が指示と違うときは「見つからない」と返す。

画面は 1920×1080 を想定し、左に Isaac Sim、右上に判定モニタ、右下にチャットを並べる（screen_layout.py）。

実行：python src\\instruct_sort.py [--seed 配置の seed] [--rules 設定ファイル] [--device cuda|cpu] [--no-ui]
"""
import argparse
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from app_config import SIM_APP_CONFIG, parse_script_args
from screen_layout import ISAAC_WINDOW

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None, help="物体の配置を決める seed（省略するとランダム）")
parser.add_argument("--rules", default=None, help="設定ファイル（既定: config/instruct_rules.yaml）")
parser.add_argument("--device", default=None, help="CLIP を動かすデバイス（cuda / cpu）。設定ファイルより優先")
parser.add_argument("--no-ui", action="store_true", help="判定モニタとチャットのウィンドウを開かない")
args = parse_script_args(parser)

from isaacsim import SimulationApp

window_x, window_y, window_w, window_h = ISAAC_WINDOW
simulation_app = SimulationApp(
    {
        **SIM_APP_CONFIG,
        "window_width": window_w,
        "window_height": window_h,
        "extra_args": SIM_APP_CONFIG["extra_args"] + [f"--/app/window/x={window_x}", f"--/app/window/y={window_y}"],
    }
)

# 起動時の設定が効かない場合に備えて、起動後にも位置と大きさを合わせる
try:
    import omni.appwindow

    app_window = omni.appwindow.get_default_app_window()
    app_window.move(window_x, window_y)
    app_window.resize(window_w, window_h)
except Exception as e:  # ウィンドウの調整に失敗しても、タスク自体は続ける
    print(f"[instruct] could not move/resize the Isaac Sim window: {e}")

import numpy as np
import yaml
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from PIL import Image

from classifier import PROJECT_ROOT, TrashClassifier
from instruct_io import RUN_DIR, CommandReader, clear_commands, write_status
from instruct_scene import (
    BIN_SIZE,
    apply_instruct_layout,
    bin_slot_position,
    build_instruct_scene,
    random_layout,
)
from scene import CROP_SIZE, crop_box_for_cube, wait_for_camera_image

# 指示ファイルを読みに行く間隔（物理ステップ数）
COMMAND_POLL_STEPS = 10

rules_path = Path(args.rules) if args.rules else PROJECT_ROOT / "config" / "instruct_rules.yaml"
config = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
object_names = {o["label"]: o["display_name"] for o in config["objects"]}
bin_names = {b["label"]: b["display_name"] for b in config["bins"]}

seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(10000))
rng = np.random.default_rng(seed)
layout = random_layout(rng)
print(f"[instruct] seed={seed} layout={layout}")
scene = build_instruct_scene(layout)

# 物体の4種類で判定する。置き場はチャットの指示で決まるので、分類器のクラスには置き場を持たせない
classifier = TrashClassifier(
    model_name=config["model"],
    classes=[dict(o, bin=None) for o in config["objects"]],
    cache_dir=str(PROJECT_ROOT / config["cache_dir"]) if config.get("cache_dir") else None,
    device=args.device or config.get("device", "auto"),
)
print(f"[instruct] CLIP loaded on {classifier.device}")

franka = scene.franka
controller = PickPlaceController(name="pick_place_controller", gripper=franka.gripper, robot_articulation=franka)
franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)

status = {
    "seed": seed,
    "objects": [{"label": o["label"], "name": o["display_name"], "color": o["display_color"]} for o in config["objects"]],
    "bins": [{"label": b["label"], "name": b["display_name"], "color": b["display_color"]} for b in config["bins"]],
    "total": len(scene.cubes),
    "remaining": len(scene.cubes),
    "busy": False,
    "current": None,
    "events": [],
}


def add_event(text: str, kind: str = "info") -> None:
    """チャットに出す、ロボットからの返事を追加する."""
    status["events"].append({"id": len(status["events"]) + 1, "text": text, "kind": kind})
    print(f"[instruct] {text}")


def reset_objects(new_layout=None) -> None:
    """物体を机の上に戻す. new_layout を渡すとその位置に置き直す."""
    for obj in scene.cubes:
        obj["active"] = True
    for bin_name in slots_used:
        slots_used[bin_name] = 0
    if new_layout is not None:
        apply_instruct_layout(scene, new_layout)
        controller.reset()
        franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)
    # 物体の番号は、画像の左から順に振り、次に並べ直すまで変えない（チャットの「物体N」と画面の番号を合わせる）
    lefts = []
    for obj in scene.cubes:
        position, _ = obj["prim"].get_world_pose()
        lefts.append((crop_box_for_cube(scene, position)[0], obj["name"], obj))
    for number, (_, _, obj) in enumerate(sorted(lefts, key=lambda item: item[:2]), start=1):
        obj["number"] = number
    status["remaining"] = len(scene.cubes)
    status["current"] = None


def capture_scene() -> Image.Image:
    """最新のカメラ画像を撮り、判定モニタ用に保存する."""
    image = Image.fromarray(wait_for_camera_image(scene))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    image.save(RUN_DIR / "scene.png")
    return image


def ground(target: str) -> list:
    """机の上の物体を1個ずつ切り出して CLIP で判定し、候補の一覧を物体の番号順に返す."""
    image = capture_scene()
    candidates = []
    for obj in scene.cubes:
        if not obj["active"]:
            continue
        position, _ = obj["prim"].get_world_pose()
        box = crop_box_for_cube(scene, position)
        crop = image.crop(box).resize((CROP_SIZE, CROP_SIZE))
        crop.save(RUN_DIR / f"{obj['name']}_crop.png")
        prediction = classifier.classify(crop)
        candidates.append(
            {
                "obj": obj,
                "crop_image": f"{obj['name']}_crop.png",
                "box": list(box),
                "target_score": prediction.scores[target],
                "top_label": prediction.label,
                "top_score": prediction.confidence,
                "number": obj["number"],
            }
        )
    candidates.sort(key=lambda c: c["number"])
    return candidates


def start_command(command: dict):
    """指示を1つ処理する. 運ぶ物体が決まったら運ぶ準備をして task を返し、決まらなければ None を返す."""
    if command["type"] == "reset":
        new_layout = random_layout(rng)
        print(f"[instruct] new layout={new_layout}")
        reset_objects(new_layout)
        # 判定モニタの画像を、並べ直した後の配置に更新する（状態ファイルより先に保存する）
        capture_scene()
        add_event("物体を新しい位置に並べ直しました。")
        write_status(status)
        return None

    target, bin_name = command["object"], command["bin"]
    target_name, bin_display = object_names[target], bin_names[bin_name]
    if status["remaining"] == 0:
        add_event("机の上に物体が残っていません。「リセット」と送ると並べ直します。", "ng")
        write_status(status)
        return None

    status["busy"] = True
    status["current"] = {
        "command_id": command["id"],
        "text": command.get("text", ""),
        "object": target,
        "bin": bin_name,
        "state": "grounding",
        "candidates": [],
        "selected": None,
    }
    write_status(status)

    candidates = ground(target)
    best = max(candidates, key=lambda c: c["target_score"])
    found = best["top_label"] == target
    current = status["current"]
    current["candidates"] = [{k: v for k, v in c.items() if k != "obj"} for c in candidates]
    current["selected"] = best["number"] if found else None
    truth = best["obj"]["kind"]
    print(
        f"[instruct] command {command['id']}: {target} -> {bin_name} | best: {best['obj']['name']} "
        f"(target {best['target_score']:.3f}, top {best['top_label']} {best['top_score']:.3f}, truth {truth})"
    )

    if not found:
        current["state"] = "not_found"
        status["busy"] = False
        nearest = object_names[best["top_label"]]
        add_event(
            f"{target_name}が見つかりませんでした（いちばん近い物体{best['number']}は{nearest}と判定、"
            f"{target_name}である信頼度 {best['target_score'] * 100:.1f}%）。",
            "ng",
        )
        write_status(status)
        return None

    slot = slots_used[bin_name]
    slots_used[bin_name] += 1
    current["state"] = "placing"
    add_event(
        f"物体{best['number']}を{target_name}と判断しました（信頼度 {best['target_score'] * 100:.1f}%）。"
        f"{bin_display}へ運びます。"
    )
    write_status(status)
    controller.reset()
    return {"obj": best["obj"], "number": best["number"], "target": target, "bin": bin_name,
            "place_position": bin_slot_position(bin_name, slot)}


def finish_task(task: dict) -> None:
    """置いた後の位置を正解値で確かめ、チャットに結果を返す."""
    obj, bin_name = task["obj"], task["bin"]
    final_position, _ = obj["prim"].get_world_pose()
    bin_xy = scene.bins[bin_name]["position"]
    in_bin = bool(np.all(np.abs(final_position[:2] - bin_xy) <= BIN_SIZE[:2] / 2))
    obj["active"] = False
    status["remaining"] = sum(o["active"] for o in scene.cubes)
    status["busy"] = False
    status["current"]["state"] = "placed" if in_bin else "missed"
    print(
        f"[instruct] {obj['name']} placed at {np.round(final_position, 3)} in {bin_name}: {'OK' if in_bin else 'NG'} "
        f"(truth {obj['kind']}, instructed {task['target']})"
    )
    if in_bin:
        add_event(f"物体{task['number']}を{bin_names[bin_name]}に置きました。", "ok")
    else:
        add_event(f"物体{task['number']}を{bin_names[bin_name]}に置けませんでした。", "ng")
    if status["remaining"] == 0:
        add_event("机の上の物体をすべて運びました。「リセット」と送ると並べ直します。")
    # 判定モニタの画像を、置き終わった後の机の様子に更新する（状態ファイルより先に保存する）
    capture_scene()
    write_status(status)


def self_check() -> None:
    """起動時に4個すべてを判定し、正解（シミュレータが知っている種類）と比べてターミナルに出す（確認用）."""
    candidates = ground(scene.cubes[0]["kind"])
    correct = 0
    for c in candidates:
        ok = c["top_label"] == c["obj"]["kind"]
        correct += ok
        print(
            f"[instruct] self-check {c['obj']['name']}: truth {c['obj']['kind']} -> CLIP {c['top_label']} "
            f"({c['top_score']:.3f}) {'OK' if ok else 'NG'}"
        )
    print(f"[instruct] self-check: {correct}/{len(candidates)}")


slots_used = {bin_name: 0 for bin_name in scene.bins}
reset_objects()
self_check()
add_event("準備ができました。「赤の箱を緑へ」のように指示してください。")
clear_commands()
write_status(status)
if not args.no_ui:
    subprocess.Popen([sys.executable, str(Path(__file__).with_name("instruct_ui.py"))])

reader = CommandReader()
queue = deque()
task = None
step = 0

while simulation_app.is_running():
    step += 1
    if step % COMMAND_POLL_STEPS == 0:
        queue.extend(reader.read_new())
    if task is None and queue:
        task = start_command(queue.popleft())
    scene.world.step(render=True)
    if not scene.world.is_playing() or task is None:
        continue
    position, _ = task["obj"]["prim"].get_world_pose()
    actions = controller.forward(
        picking_position=position,
        placing_position=task["place_position"],
        current_joint_positions=franka.get_joint_positions(),
        end_effector_offset=np.array([0.0, 0.005, 0.0]),
    )
    franka.apply_action(actions)
    if controller.is_done():
        finish_task(task)
        task = None

simulation_app.close()
