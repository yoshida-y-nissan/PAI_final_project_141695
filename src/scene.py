"""ゴミ分別タスクのシーン（Franka・立方体・置き場・上からのカメラ）を組み立てる.

SimulationApp を起動した後に import すること。
"""
from dataclasses import dataclass

import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera
from pxr import UsdLux

CUBE_SIZE = 0.0515

# 床は無彩色にする。既定の床（青いグリッド）だと、背景の青に引っ張られて赤い立方体の判定が揺らぐ
FLOOR_COLOR = np.array([0.6, 0.6, 0.6])
# 光源はドームライトのみ。全方向から照らすので、アームの影が立方体に落ちにくい
DOME_LIGHT_INTENSITY = 1000.0

# 床の薄い灰色のグリッド（見た目用。当たり判定なし）。立方体がマスの中央に来るように線を置き、
# 切り出し画像に線が入らないようにする
GRID_PITCH = 0.15
GRID_ORIGIN = (0.475, 0.075)  # x 方向・y 方向の線が通る位置の1つ
GRID_HALF_EXTENT = 1.5  # カメラの真下から ±1.5m の範囲に引く
GRID_LINE_WIDTH = 0.003
GRID_COLOR = np.array([0.45, 0.45, 0.45])

COLORS = {
    "red": np.array([0.85, 0.1, 0.1]),
    "blue": np.array([0.1, 0.25, 0.85]),
}

# 立方体を置けるマス（xy 位置）。Franka の根元が原点で、届く範囲は半径およそ 0.8m。グリッドの各マスの中央
# ラウンドごとに、この6マスから NUM_CUBES 個を選び、色もランダムに決める（random_layouts）
GRID_POSITIONS = [
    (0.40, -0.15),
    (0.40, 0.00),
    (0.40, 0.15),
    (0.55, -0.15),
    (0.55, 0.00),
    (0.55, 0.15),
]
NUM_CUBES = 3

# 立方体が全部同じ置き場に行っても置けるよう、2列×3個（6個）の置き場所を持つ大きさにする
BIN_SIZE = np.array([0.26, 0.26, 0.01])

# 赤を入れる置き場は淡い赤、青を入れる置き場は淡い青
BIN_SPECS = {
    "burnable_bin": {"position": (0.30, 0.45), "color": np.array([0.95, 0.65, 0.65])},
    "recyclable_bin": {"position": (0.30, -0.45), "color": np.array([0.65, 0.78, 0.95])},
}

# 置き場の中の置き場所（置き場の中心からのずれ dx, dy）。置く順に並べる
# dy は「ロボットの正面側（y=0 側）が負」の向き。手前の列から埋める
# - x 方向の間隔 0.07m：グリッパの指は y 方向に開くので、x 方向の隣には当たらない
# - y 方向の間隔 0.12m：開いた指の外側（中心から約 0.06m）が隣の列の立方体（中心から 0.094m〜）に当たらない
BIN_SLOT_OFFSETS = [
    (0.0, -0.06), (-0.07, -0.06), (0.07, -0.06),
    (0.0, 0.06), (-0.07, 0.06), (0.07, 0.06),
]

# x は立方体2列（0.40 と 0.55）の中間。ずらすと奥の列が画像の端で切れる
CAMERA_POSITION = np.array([0.475, 0.0, 1.3])
CAMERA_RESOLUTION = (640, 480)
CROP_SIZE = 224  # 切り出し画像は CLIP の入力サイズに合わせて拡大する


@dataclass
class SceneObjects:
    world: World
    franka: Franka
    cubes: list  # [{"name", "color", "prim"}]
    bins: dict  # {bin_name: {"position", "prim"}}
    camera: Camera


def random_layouts(seed=None, rounds: int = 1) -> tuple:
    """ラウンドごとの立方体の配置をランダムに決める. 戻り値は (使った seed, 配置のリスト).

    配置は [{"position": (x, y), "color": "red" | "blue"}, ...] で、GRID_POSITIONS から重ならないように
    NUM_CUBES マスを選び、色は赤・青からランダムに選ぶ（偏りもあり得る）。運ぶ順に合わせて手前のマスから並べる。
    seed を省略すると seed 自体をランダムに決める。表示された seed を渡せば同じ配置を再現できる。
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(10000))
    rng = np.random.default_rng(seed)
    layouts = []
    for _ in range(rounds):
        cells = sorted(int(i) for i in rng.choice(len(GRID_POSITIONS), size=NUM_CUBES, replace=False))
        colors = [str(c) for c in rng.choice(list(COLORS), size=NUM_CUBES)]
        layouts.append([{"position": GRID_POSITIONS[i], "color": c} for i, c in zip(cells, colors)])
    return seed, layouts


def apply_layout(scene: "SceneObjects", layout: list) -> None:
    """立方体を layout の位置と色に置き直し、ロボットも含めてシーンを初期状態に戻す.

    Isaac Sim を開いたまま、次のラウンドを始めるときに使う。
    """
    for cube, item in zip(scene.cubes, layout):
        x, y = item["position"]
        cube["prim"].set_default_state(position=np.array([x, y, CUBE_SIZE / 2]))
        cube["prim"].get_applied_visual_material().set_color(COLORS[item["color"]])
        cube["color"] = item["color"]
    # 立方体は既定の位置（上で変えた位置）へ、ロボットは初期姿勢へ戻る
    scene.world.reset()


def bin_slot_position(scene: "SceneObjects", bin_name: str, slot: int) -> np.ndarray:
    """置き場の slot 番目の置き場所（立方体の中心の高さ）を返す."""
    bin_x, bin_y = scene.bins[bin_name]["position"]
    dx, dy = BIN_SLOT_OFFSETS[slot % len(BIN_SLOT_OFFSETS)]
    # ロボットの正面（y=0）から遠ざかる向きを dy の正にする
    dy *= np.sign(bin_y)
    return np.array([bin_x + dx, bin_y + dy, BIN_SIZE[2] + CUBE_SIZE / 2])


def build_scene(layout: list) -> SceneObjects:
    """シーンを組み立てる. 立方体は layout（random_layouts の1ラウンド分）の位置と色で置く."""
    world = World(stage_units_in_meters=1.0)
    # add_default_ground_plane() は青いグリッドの床と光源を読み込むので使わず、単色の床と光源を自分で置く
    world.scene.add_ground_plane(size=10.0, color=FLOOR_COLOR)
    _add_floor_grid()
    UsdLux.DomeLight.Define(get_current_stage(), "/World/dome_light").CreateIntensityAttr(DOME_LIGHT_INTENSITY)

    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

    cubes = []
    for i, item in enumerate(layout):
        name, color = f"cube_{i}", item["color"]
        x, y = item["position"]
        prim = world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/{name}",
                name=name,
                position=np.array([x, y, CUBE_SIZE / 2]),
                scale=np.array([CUBE_SIZE] * 3),
                color=COLORS[color],
            )
        )
        cubes.append({"name": name, "color": color, "prim": prim})

    bins = {}
    for bin_name, spec in BIN_SPECS.items():
        x, y = spec["position"]
        prim = world.scene.add(
            FixedCuboid(
                prim_path=f"/World/{bin_name}",
                name=bin_name,
                position=np.array([x, y, BIN_SIZE[2] / 2]),
                scale=BIN_SIZE,
                color=spec["color"],
            )
        )
        bins[bin_name] = {"position": np.array([x, y]), "prim": prim}

    # 机の上を真上から見下ろすカメラ（world 軸で y 軸まわりに 90 度回すと下向きになる）
    camera = Camera(
        prim_path="/World/top_camera",
        position=CAMERA_POSITION,
        frequency=20,
        resolution=CAMERA_RESOLUTION,
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
    )

    world.reset()
    camera.initialize()
    return SceneObjects(world=world, franka=franka, cubes=cubes, bins=bins, camera=camera)


def _add_floor_grid() -> None:
    """床に薄い灰色の線を格子状に並べる. 線は厚さ 1mm の VisualCuboid（当たり判定なし）."""
    center_x, center_y = CAMERA_POSITION[0], CAMERA_POSITION[1]
    length = 2 * GRID_HALF_EXTENT
    n = int(GRID_HALF_EXTENT // GRID_PITCH)
    for axis, origin, center in (("x", GRID_ORIGIN[0], center_x), ("y", GRID_ORIGIN[1], center_y)):
        # center に最も近い線の位置から、両側に GRID_PITCH ごとに並べる
        first = origin + round((center - origin) / GRID_PITCH) * GRID_PITCH
        for i in range(-n, n + 1):
            offset = first + i * GRID_PITCH
            if axis == "x":  # x が一定の線（y 方向に伸びる）
                position = [offset, center_y, 0.0005]
                scale = [GRID_LINE_WIDTH, length, 0.001]
            else:  # y が一定の線（x 方向に伸びる）
                position = [center_x, offset, 0.0005]
                scale = [length, GRID_LINE_WIDTH, 0.001]
            VisualCuboid(
                prim_path=f"/World/floor_grid/{axis}_{i + n}",
                position=np.array(position),
                scale=np.array(scale),
                color=GRID_COLOR,
            )


def wait_for_camera_image(scene: SceneObjects, min_steps: int = 30, max_steps: int = 200) -> np.ndarray:
    """少なくとも min_steps 進めてからカメラ画像を取り、RGB 画像 (H, W, 3) を返す.

    最初の数フレームは材質の読み込みが終わっておらず、実際と違う色で写ることがあるため待つ。
    """
    for i in range(max_steps):
        scene.world.step(render=True)
        if i + 1 < min_steps:
            continue
        rgba = scene.camera.get_rgba()
        if rgba is not None and rgba.size > 0 and rgba.ndim == 3:
            return rgba[..., :3].astype(np.uint8)
    raise RuntimeError("カメラ画像を取得できませんでした")


def crop_box_for_cube(scene: SceneObjects, position: np.ndarray, margin: float = 0.5) -> tuple:
    """立方体の中心を画像に投影し、立方体の周囲を切り出す矩形 (left, top, right, bottom) を返す."""
    points = np.array([position, position + np.array([0.0, CUBE_SIZE, 0.0])])
    uv = scene.camera.get_image_coords_from_world_points(points)
    center = uv[0]
    cube_px = np.linalg.norm(uv[1] - uv[0])
    half = cube_px * (0.5 + margin)
    width, height = CAMERA_RESOLUTION
    left = int(max(center[0] - half, 0))
    top = int(max(center[1] - half, 0))
    right = int(min(center[0] + half, width))
    bottom = int(min(center[1] + half, height))
    return left, top, right, bottom
