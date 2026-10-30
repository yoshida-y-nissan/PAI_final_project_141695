"""言語指示タスクのシーン：赤・青の箱と円柱を1個ずつランダムな位置に置き、4色の置き場を固定の位置に置く.

床・照明・カメラ・画像の取得・切り出しは、フェーズ4のシーン（scene.py）と同じものを使う。
SimulationApp を起動した後に import すること。
"""
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, FixedCuboid
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera
from pxr import UsdLux

from scene import (
    CAMERA_POSITION,
    CAMERA_RESOLUTION,
    COLORS,
    CUBE_SIZE,
    DOME_LIGHT_INTENSITY,
    FLOOR_COLOR,
    SceneObjects,
    _add_floor_grid,
)

# 物体の種類（ラベル, 色, 形）。円柱は箱と同じ大きさ（直径・高さ = CUBE_SIZE）にする
OBJECT_KINDS = [
    ("red_box", "red", "box"),
    ("red_cylinder", "red", "cylinder"),
    ("blue_box", "blue", "box"),
    ("blue_cylinder", "blue", "cylinder"),
]

# 物体を置く範囲。カメラに写り、切り出し範囲（物体の中心から約 0.053m）が画像からはみ出さない範囲にする
WORK_AREA_X = (0.36, 0.60)
WORK_AREA_Y = (-0.17, 0.17)
# 物体どうしの最小間隔。開いたグリッパの指（中心から約 0.06m）が隣の物体に当たらないようにする
MIN_SPACING = 0.12

# 置き場は固定の位置。2列×2個で最大4個置ける。カメラの範囲（y ±0.27m）の外に置く
BIN_SIZE = np.array([0.17, 0.22, 0.01])
BIN_SPECS = {
    "red": {"position": (0.25, 0.40), "color": np.array([0.95, 0.65, 0.65])},
    "green": {"position": (0.47, 0.40), "color": np.array([0.65, 0.90, 0.65])},
    "blue": {"position": (0.25, -0.40), "color": np.array([0.65, 0.78, 0.95])},
    "yellow": {"position": (0.47, -0.40), "color": np.array([0.95, 0.90, 0.55])},
}
# 置き場の中の置き場所（置き場の中心からのずれ dx, dy）。dy は「ロボットの正面側（y=0 側）が負」の向きで、手前の列から埋める
# x 方向の間隔 0.07m・y 方向の間隔 0.12m は、フェーズ4b で隣に当たらないことを確かめた間隔と同じ
BIN_SLOT_OFFSETS = [(-0.035, -0.06), (0.035, -0.06), (-0.035, 0.06), (0.035, 0.06)]


def random_layout(rng: np.random.Generator) -> list:
    """物体ごとの xy 位置をランダムに決める（OBJECT_KINDS と同じ順）. 物体どうしは MIN_SPACING 以上離す."""
    while True:
        positions = []
        for _ in range(1000):
            p = np.array([rng.uniform(*WORK_AREA_X), rng.uniform(*WORK_AREA_Y)])
            if all(np.linalg.norm(p - q) >= MIN_SPACING for q in positions):
                positions.append(p)
                if len(positions) == len(OBJECT_KINDS):
                    return [tuple(round(float(v), 3) for v in p) for p in positions]


def bin_slot_position(bin_name: str, slot: int) -> np.ndarray:
    """置き場の slot 番目の置き場所（物体の中心の高さ）を返す."""
    bin_x, bin_y = BIN_SPECS[bin_name]["position"]
    dx, dy = BIN_SLOT_OFFSETS[slot % len(BIN_SLOT_OFFSETS)]
    dy *= np.sign(bin_y)
    return np.array([bin_x + dx, bin_y + dy, BIN_SIZE[2] + CUBE_SIZE / 2])


def build_instruct_scene(layout: list) -> SceneObjects:
    """シーンを組み立てる. 物体は layout（random_layout の戻り値）の位置に置く.

    物体の一覧は SceneObjects.cubes に入れる（{"name", "kind", "color", "shape", "prim"}）。
    """
    world = World(stage_units_in_meters=1.0)
    world.scene.add_ground_plane(size=10.0, color=FLOOR_COLOR)
    _add_floor_grid()
    UsdLux.DomeLight.Define(get_current_stage(), "/World/dome_light").CreateIntensityAttr(DOME_LIGHT_INTENSITY)

    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

    objects = []
    for i, ((kind, color, shape), (x, y)) in enumerate(zip(OBJECT_KINDS, layout)):
        name = f"obj_{i}"
        position = np.array([x, y, CUBE_SIZE / 2])
        if shape == "box":
            prim = DynamicCuboid(
                prim_path=f"/World/{name}", name=name, position=position, scale=np.array([CUBE_SIZE] * 3), color=COLORS[color]
            )
        else:
            prim = DynamicCylinder(
                prim_path=f"/World/{name}",
                name=name,
                position=position,
                radius=CUBE_SIZE / 2,
                height=CUBE_SIZE,
                color=COLORS[color],
            )
        world.scene.add(prim)
        objects.append({"name": name, "kind": kind, "color": color, "shape": shape, "prim": prim})

    bins = {}
    for bin_name, spec in BIN_SPECS.items():
        x, y = spec["position"]
        prim = world.scene.add(
            FixedCuboid(
                prim_path=f"/World/bin_{bin_name}",
                name=f"bin_{bin_name}",
                position=np.array([x, y, BIN_SIZE[2] / 2]),
                scale=BIN_SIZE,
                color=spec["color"],
            )
        )
        bins[bin_name] = {"position": np.array([x, y]), "prim": prim}

    # 机の上を真上から見下ろすカメラ（scene.py と同じ）
    camera = Camera(
        prim_path="/World/top_camera",
        position=CAMERA_POSITION,
        frequency=20,
        resolution=CAMERA_RESOLUTION,
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
    )

    world.reset()
    camera.initialize()
    return SceneObjects(world=world, franka=franka, cubes=objects, bins=bins, camera=camera)


def apply_instruct_layout(scene: SceneObjects, layout: list) -> None:
    """物体を layout の位置に置き直し、ロボットも含めてシーンを初期状態に戻す（「リセット」の指示用）."""
    for obj, (x, y) in zip(scene.cubes, layout):
        obj["prim"].set_default_state(position=np.array([x, y, CUBE_SIZE / 2]))
    scene.world.reset()
