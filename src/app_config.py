"""SimulationApp の起動設定（isaacsim を import する前に読み込めるよう、このファイルでは isaacsim を import しない）."""
import argparse
import sys

SIM_APP_CONFIG = {
    "headless": False,
    # Microsoft の Vulkan 互換レイヤーが作る仮想 GPU を使わず、実 GPU 0 番だけで描画する
    "multi_gpu": False,
    "active_gpu": 0,
    # Isaac Sim 4.5 の Vulkan 描画が新しいドライバでレイトレーシング確認に失敗するため、DirectX 12 を使う
    "extra_args": ["--/app/vulkan=false"],
}


def parse_script_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """スクリプト用の引数を読み、sys.argv から取り除く.

    SimulationApp は sys.argv の残りを Kit に渡すので、SimulationApp を作る前に呼ぶ。
    """
    args, kit_args = parser.parse_known_args()
    sys.argv = sys.argv[:1] + kit_args
    return args
