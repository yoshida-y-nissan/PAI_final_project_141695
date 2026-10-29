"""Phase 1: Franka が赤い立方体を1個掴み、指定位置に置くことを確認する."""
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

# SimulationApp は他の isaacsim モジュールより先に起動する必要がある
# simulation_app = SimulationApp({"headless": False})
# simulation_app = SimulationApp({"headless": False, "multi_gpu": False, "active_gpu": 0})
simulation_app = SimulationApp(
    {
        "headless": False,
        "multi_gpu": False,
        "active_gpu": 0,
        # Isaac Sim 4.5 の Vulkan 描画が新しいドライバでレイトレーシング確認に失敗するため、DirectX 12 を使う
        "extra_args": ["--/app/vulkan=false"],
    }
)

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

CUBE_SIZE = 0.0515
PICK_POSITION = np.array([0.5, 0.0, CUBE_SIZE / 2])
PLACE_POSITION = np.array([0.3, 0.3, CUBE_SIZE / 2])

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/cube_0",
        name="cube_0",
        position=PICK_POSITION,
        scale=np.array([CUBE_SIZE] * 3),
        color=np.array([1.0, 0.0, 0.0]),
    )
)
world.reset()

controller = PickPlaceController(
    name="pick_place_controller",
    gripper=franka.gripper,
    robot_articulation=franka,
)
franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)

done = False
while simulation_app.is_running():
    world.step(render=True)
    if not world.is_playing() or done:
        continue
    cube_position, _ = cube.get_world_pose()
    actions = controller.forward(
        picking_position=cube_position,
        placing_position=PLACE_POSITION,
        current_joint_positions=franka.get_joint_positions(),
        end_effector_offset=np.array([0.0, 0.005, 0.0]),
    )
    franka.apply_action(actions)
    if controller.is_done():
        final_position, _ = cube.get_world_pose()
        error = np.linalg.norm(final_position[:2] - PLACE_POSITION[:2])
        print(f"[phase1] done: cube={np.round(final_position, 3)} target={PLACE_POSITION} xy_error={error:.3f} m")
        done = True

simulation_app.close()
