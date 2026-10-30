"""言語指示タスクの画面：判定モニタ（右上）とチャット（右下）.

instruct_sort.py が自動で起動する。isaacsim に依存しない（tkinter と Pillow だけで動く）。
- チャット：入力した指示を LLM（なければルール）で解析し、指示ファイルに書く。ロボットからの返事を表示する
- 判定モニタ：上からのカメラ画像と、物体ごとの「指示された物体である信頼度」を表示する
シミュレーションとは instruct_io.py のファイルでやり取りし、状態を 0.2 秒ごとに読む。
"""
import json
import queue
import threading
import tkinter as tk
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageTk

from instruct_io import RUN_DIR, append_command, read_status
from llm_parser import load_api_key, parse_instruction
from screen_layout import CHAT_WINDOW, MONITOR_WINDOW

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "instruct_rules.yaml"
POLL_MS = 200
SCENE_SIZE = (416, 312)  # カメラ画像（640×480）を縮小して表示する
THUMB_SIZE = 56
BAR_WIDTH = 110

BG = "#1f2329"
PANEL = "#2a2f36"
TEXT = "#e8e8e8"
SUBTEXT = "#a0a6ad"
HIGHLIGHT = "#ffd84d"
OK_COLOR = "#5cc98a"
NG_COLOR = "#ff6b6b"
USER_COLOR = "#8ab4f0"
FONT = "Yu Gothic UI"

STATE_TEXT = {
    "grounding": ("物体を探しています…", TEXT),
    "placing": ("運んでいます…", TEXT),
    "placed": ("置きました", OK_COLOR),
    "missed": ("置き場に置けませんでした", NG_COLOR),
    "not_found": ("指示に合う物体が見つかりませんでした", NG_COLOR),
}


def load_image(path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def place_window(window, rect):
    x, y, w, h = rect
    window.geometry(f"{w}x{h}+{x}+{y}")


class Monitor:
    """判定モニタ（右上）."""

    def __init__(self, window: tk.Misc):
        self.photos = {}  # PhotoImage は参照を持っておかないと表示が消える
        window.configure(bg=BG)

        self.instruction = tk.Label(window, font=(FONT, 13, "bold"), fg=TEXT, bg=BG, anchor="w")
        self.instruction.pack(fill="x", padx=10, pady=(8, 0))
        self.target = tk.Label(window, font=(FONT, 11), fg=SUBTEXT, bg=BG, anchor="w")
        self.target.pack(fill="x", padx=10)

        body = tk.Frame(window, bg=BG)
        body.pack(fill="x", padx=10, pady=6)
        left = tk.Frame(body, bg=PANEL)
        left.pack(side="left", anchor="n")
        tk.Label(left, text="上からのカメラ画像（黄色の枠 = 選んだ物体）", font=(FONT, 9), fg=SUBTEXT, bg=PANEL).pack(
            anchor="w", padx=6, pady=(4, 2)
        )
        self.scene_label = tk.Label(left, bg=PANEL)
        self.scene_label.pack(padx=6, pady=(0, 6))

        right = tk.Frame(body, bg=PANEL)
        right.pack(side="left", anchor="n", fill="both", expand=True, padx=(8, 0))
        # 物体の判定は LLM ではなく CLIP（画像と言葉の対応を学習したモデル）が行っていることを示す
        tk.Label(right, text="CLIP による判断", font=(FONT, 11, "bold"), fg=HIGHLIGHT, bg=PANEL, anchor="w").pack(
            fill="x", padx=6, pady=(4, 0)
        )
        self.candidates_title = tk.Label(right, font=(FONT, 9), fg=SUBTEXT, bg=PANEL, anchor="w")
        self.candidates_title.pack(fill="x", padx=6, pady=(4, 2))
        self.candidates = tk.Canvas(right, width=260, height=4 * 72, bg=PANEL, highlightthickness=0)
        self.candidates.pack(padx=6, pady=(0, 6))

        self.state = tk.Label(window, font=(FONT, 12, "bold"), fg=TEXT, bg=BG, anchor="w")
        self.state.pack(fill="x", padx=10)
        self.footer = tk.Label(window, font=(FONT, 9), fg=SUBTEXT, bg=BG, anchor="w")
        self.footer.pack(fill="x", padx=10, pady=(2, 8))
        self.show_waiting()

    def set_image(self, widget, key, image):
        self.photos[key] = ImageTk.PhotoImage(image)
        widget.config(image=self.photos[key])
        return self.photos[key]

    def show_waiting(self, text="シミュレーションの起動を待っています…"):
        self.instruction.config(text=text)
        self.target.config(text="")
        self.set_image(self.scene_label, "scene", Image.new("RGB", SCENE_SIZE, PANEL))
        self.candidates_title.config(text="")
        self.candidates.delete("all")
        self.state.config(text="")
        self.footer.config(text="")

    def refresh(self, status):
        if not status:
            self.show_waiting()
            return
        objects = {o["label"]: o for o in status["objects"]}
        bins = {b["label"]: b for b in status["bins"]}
        self.footer.config(text=f"机の上の物体 {status['remaining']} / {status['total']}    seed = {status['seed']}")
        current = status["current"]
        if current is None:
            self.instruction.config(text="指示を待っています（チャットに入力してください）")
            self.target.config(text="")
            if (RUN_DIR / "scene.png").exists():
                self.set_image(self.scene_label, "scene", load_image(RUN_DIR / "scene.png").resize(SCENE_SIZE))
            self.candidates_title.config(text="")
            self.candidates.delete("all")
            self.state.config(text="")
            return

        target, bin_ = objects[current["object"]], bins[current["bin"]]
        self.instruction.config(text=f"指示：「{current['text']}」")
        self.target.config(text=f"対象：{target['name']}　→　{bin_['name']}")
        message, color = STATE_TEXT.get(current["state"], ("", TEXT))
        self.state.config(text=message, fg=color)
        candidates = current["candidates"]
        if not candidates:  # 判定中
            return

        scene = load_image(RUN_DIR / "scene.png")
        if current["state"] in ("placed", "missed"):
            # 置き終わった後の画像は撮り直したものなので、判定時の枠は描かない
            self.set_image(self.scene_label, "scene", scene.resize(SCENE_SIZE))
            self.draw_candidates(candidates, current["selected"], target, objects)
            return
        draw = ImageDraw.Draw(scene)
        for c in candidates:
            selected = c["number"] == current["selected"]
            draw.rectangle(c["box"], outline=HIGHLIGHT if selected else "#9aa0a6", width=5 if selected else 2)
            draw.text((c["box"][0] + 4, c["box"][1] + 2), str(c["number"]), fill=HIGHLIGHT if selected else "#ffffff")
        self.set_image(self.scene_label, "scene", scene.resize(SCENE_SIZE))
        self.draw_candidates(candidates, current["selected"], target, objects)

    def draw_candidates(self, candidates, selected, target, objects):
        """物体ごとに、切り出し画像・指示された物体である信頼度・CLIP の判定を並べる."""
        self.candidates_title.config(text=f"物体ごとの「{target['name']}」である信頼度")
        canvas = self.candidates
        canvas.delete("all")
        row = 72
        for i, c in enumerate(candidates):
            y = i * row
            if c["number"] == selected:
                canvas.create_rectangle(1, y + 1, 258, y + row - 3, outline=HIGHLIGHT, width=2)
            thumb = load_image(RUN_DIR / c["crop_image"]).resize((THUMB_SIZE, THUMB_SIZE))
            photo = ImageTk.PhotoImage(thumb)
            self.photos[f"thumb{i}"] = photo
            canvas.create_image(6, y + 7, image=photo, anchor="nw")
            x = 6 + THUMB_SIZE + 8
            canvas.create_text(x, y + 14, text=f"物体{c['number']}", anchor="w", fill=TEXT, font=(FONT, 10, "bold"))
            canvas.create_rectangle(x, y + 28, x + BAR_WIDTH, y + 42, fill=BG, outline="")
            canvas.create_rectangle(x, y + 28, x + BAR_WIDTH * c["target_score"], y + 42, fill=target["color"], outline="")
            canvas.create_text(x + BAR_WIDTH + 6, y + 35, text=f"{c['target_score'] * 100:.1f}%", anchor="w", fill=TEXT,
                               font=(FONT, 10))
            top = objects[c["top_label"]]
            canvas.create_text(x, y + 56, text=f"判定：{top['name']} {c['top_score'] * 100:.0f}%", anchor="w",
                               fill=SUBTEXT, font=(FONT, 9))


class Chat:
    """チャット（右下）."""

    def __init__(self, window: tk.Misc, config: dict):
        self.config = config
        self.results = queue.Queue()
        self.next_id = 1
        window.configure(bg=BG)

        llm_model = config.get("llm", {}).get("model", "")
        mode = f"解析：LLM（{llm_model}）" if load_api_key() else "解析：ルール（OpenAI の API キーが未設定）"
        tk.Label(window, text=f"チャットで指示    {mode}", font=(FONT, 9), fg=SUBTEXT, bg=BG, anchor="w").pack(
            fill="x", padx=10, pady=(6, 2)
        )

        entry_row = tk.Frame(window, bg=BG)
        entry_row.pack(side="bottom", fill="x", padx=10, pady=(4, 10))
        self.entry = tk.Entry(entry_row, font=(FONT, 12), bg=PANEL, fg=TEXT, insertbackground=TEXT, relief="flat")
        self.entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.entry.bind("<Return>", lambda _: self.send())
        tk.Button(entry_row, text="送信", font=(FONT, 11), command=self.send, bg="#3a4048", fg=TEXT, relief="flat",
                  activebackground="#4a5058", activeforeground=TEXT, padx=14).pack(side="left", padx=(6, 0), ipady=2)

        self.log = tk.Text(window, font=(FONT, 11), bg=PANEL, fg=TEXT, relief="flat", wrap="char", padx=8, pady=6,
                           state="disabled")
        self.log.pack(fill="both", expand=True, padx=10)
        self.log.tag_configure("user", foreground=USER_COLOR)
        self.log.tag_configure("assistant", foreground=TEXT)
        self.log.tag_configure("note", foreground=SUBTEXT)
        self.log.tag_configure("llm", foreground=HIGHLIGHT, font=("Consolas", 10))
        self.log.tag_configure("ok", foreground=OK_COLOR)
        self.log.tag_configure("ng", foreground=NG_COLOR)
        self.log.tag_configure("info", foreground=TEXT)
        self.entry.focus_set()

    def write(self, speaker: str, text: str, tag: str):
        self.log.config(state="normal")
        self.log.insert("end", f"{speaker}：{text}\n", tag)
        self.log.see("end")
        self.log.config(state="disabled")

    def send(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self.write("あなた", text, "user")
        # LLM の応答を待つ間も画面が固まらないよう、別スレッドで解析する
        threading.Thread(target=lambda: self.results.put((text, parse_instruction(text, self.config))), daemon=True).start()

    def poll_results(self):
        while not self.results.empty():
            text, result = self.results.get()
            if result["note"]:
                self.write("システム", result["note"], "note")
            if result["raw"]:
                # LLM が指示を変換した結果（JSON）と、使ったトークン数
                try:
                    raw = json.dumps(json.loads(result["raw"]), ensure_ascii=False)
                except json.JSONDecodeError:
                    raw = result["raw"]
                self.write("LLM の出力", raw, "llm")
                usage = result["usage"]
                if usage:
                    self.write(
                        "トークン数",
                        f"入力 {usage.get('prompt_tokens', '?')} / 出力 {usage.get('completion_tokens', '?')}"
                        f" / 合計 {usage.get('total_tokens', '?')}",
                        "note",
                    )
            objects = {o["label"]: o["display_name"] for o in self.config["objects"]}
            bins = {b["label"]: b["display_name"] for b in self.config["bins"]}
            if result["reset"]:
                append_command({"id": self.next_id, "type": "reset", "text": text})
                self.next_id += 1
            for c in result["commands"]:
                append_command({"id": self.next_id, "type": "move", "object": c["object"], "bin": c["bin"], "text": text})
                self.next_id += 1
            reply = result["reply"]
            if not reply and result["commands"]:
                reply = "、".join(f"{objects[c['object']]}を{bins[c['bin']]}へ" for c in result["commands"]) + "運びます。"
            if reply:
                source = "LLM" if result["source"] == "llm" else "ルール"
                self.write(f"アシスタント（{source}）", reply, "assistant")

    def show_events(self, events):
        for event in events:
            self.write("ロボット", event["text"], event["kind"])


class InstructUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.last_key = None
        self.last_event_id = 0
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

        root.title("判定モニタ")
        place_window(root, MONITOR_WINDOW)
        self.monitor = Monitor(root)

        chat_window = tk.Toplevel(root)
        chat_window.title("チャット")
        place_window(chat_window, CHAT_WINDOW)
        chat_window.protocol("WM_DELETE_WINDOW", root.destroy)
        self.chat = Chat(chat_window, config)
        self.poll()

    def poll(self):
        self.chat.poll_results()
        status = read_status()
        key = json.dumps(status, sort_keys=True) if status else None
        if key != self.last_key:
            try:
                self.monitor.refresh(status)
                self.last_key = key
            except OSError:
                pass  # 画像の書き込み途中などで読めなかったときは、次の周期で読み直す
            if status:
                events = status["events"]
                if events and events[-1]["id"] < self.last_event_id:
                    self.last_event_id = 0  # シミュレーションが起動し直された
                self.chat.show_events([e for e in events if e["id"] > self.last_event_id])
                if events:
                    self.last_event_id = events[-1]["id"]
        self.root.after(POLL_MS, self.poll)


def main():
    try:
        # 高 DPI の画面で文字がぼやけないようにする（Windows のみ）
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    InstructUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
