"""判定の様子を判定モニタ（judge_viewer.py）に伝える状態ファイル.

phase4_sort.py が書き、judge_viewer.py が読む。isaacsim に依存しない。
"""
import json
import os
import time
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[1] / "assets" / "run"
STATUS_FILE = RUN_DIR / "status.json"


def write_status(status: dict) -> None:
    """途中まで書かれたファイルを読まれないよう、一時ファイルに書いてから置き換える."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows では読み込み中のファイルを置き換えられないことがあるので、少し待って再試行する
    for _ in range(10):
        try:
            os.replace(tmp, STATUS_FILE)
            return
        except PermissionError:
            time.sleep(0.02)
    os.replace(tmp, STATUS_FILE)


def read_status():
    """状態ファイルを読む. まだ無い・書き換え中のときは None を返す."""
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
