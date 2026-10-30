"""言語指示タスクで、シミュレーション（instruct_sort.py）と画面（instruct_ui.py）がやり取りするファイル.

- 状態ファイル（JSON）：シミュレーションが書き、画面が読む（判定の様子、ロボットからの返事）
- 指示ファイル（1行1指示の JSON Lines）：画面が追記し、シミュレーションが新しい行だけを読む
isaacsim に依存しない。
"""
import json
import os
import time
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[1] / "assets" / "run" / "instruct"
STATUS_FILE = RUN_DIR / "status.json"
COMMANDS_FILE = RUN_DIR / "commands.jsonl"


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


def clear_commands() -> None:
    """指示ファイルを空にする（シミュレーションの起動時）."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    COMMANDS_FILE.write_text("", encoding="utf-8")


def append_command(command: dict) -> None:
    """指示を1行追記する（画面から）."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with COMMANDS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(command, ensure_ascii=False) + "\n")


class CommandReader:
    """指示ファイルの、まだ読んでいない行を読む（シミュレーションから）."""

    def __init__(self):
        self.offset = 0

    def read_new(self) -> list:
        try:
            with COMMANDS_FILE.open("rb") as f:
                f.seek(self.offset)
                data = f.read()
        except OSError:
            return []
        # 書き込み途中の最後の行は次回に回す
        end = data.rfind(b"\n") + 1
        self.offset += end
        commands = []
        for line in data[:end].decode("utf-8").splitlines():
            if line.strip():
                commands.append(json.loads(line))
        return commands
