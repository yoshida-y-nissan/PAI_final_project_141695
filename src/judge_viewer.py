"""判定モニタ：phase4_sort.py の判定の様子（CLIP に入れた画像・判定結果・信頼度）を別ウィンドウに表示する.

phase4_sort.py が自動で起動する。単体でも起動できる（最後の実行結果を表示する）：python src\\judge_viewer.py
isaacsim に依存しない（tkinter と Pillow だけで動く）。状態は judge_status.py の状態ファイルから 0.2 秒ごとに読む。
"""
import json
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk

from judge_status import RUN_DIR, read_status

POLL_MS = 200
SCENE_SIZE = (512, 384)  # カメラ画像（640×480）を縮小して表示する
CROP_SIZE = 224
BAR_WIDTH = 240

BG = "#1f2329"
PANEL = "#2a2f36"
TEXT = "#e8e8e8"
SUBTEXT = "#a0a6ad"
HIGHLIGHT = "#ffd84d"
OK_COLOR = "#5cc98a"
NG_COLOR = "#ff6b6b"
FONT = "Yu Gothic UI"


def load_image(path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class JudgeViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.last_key = None
        self.photos = {}  # PhotoImage は参照を持っておかないと表示が消える

        root.title("CLIP 判定モニタ")
        root.configure(bg=BG)
        root.resizable(False, False)

        self.header = tk.Label(root, font=(FONT, 12), fg=SUBTEXT, bg=BG, anchor="w")
        self.header.grid(row=0, column=0, columnspan=2, sticky="we", padx=12, pady=(10, 4))

        # 左：上からのカメラ画像
        left = tk.Frame(root, bg=PANEL)
        left.grid(row=1, column=0, padx=(12, 6), pady=6, sticky="nsew")
        tk.Label(left, text="上からのカメラ画像（黄色の枠 = 判定する立方体）", font=(FONT, 10), fg=SUBTEXT, bg=PANEL).pack(
            anchor="w", padx=8, pady=(6, 2)
        )
        self.scene_label = tk.Label(left, bg=PANEL)
        self.scene_label.pack(padx=8, pady=(0, 8))

        # 右：CLIP に入れた画像と判定結果
        right = tk.Frame(root, bg=PANEL, width=300)
        right.grid(row=1, column=1, padx=(6, 12), pady=6, sticky="nsew")
        tk.Label(right, text="CLIP に入れた画像", font=(FONT, 10), fg=SUBTEXT, bg=PANEL).pack(anchor="w", padx=8, pady=(6, 2))
        self.crop_label = tk.Label(right, bg=PANEL)
        self.crop_label.pack(padx=8)
        self.cube_label = tk.Label(right, font=(FONT, 12), fg=SUBTEXT, bg=PANEL)
        self.cube_label.pack(pady=(4, 0))
        self.result_label = tk.Label(right, font=(FONT, 26, "bold"), fg=TEXT, bg=PANEL)
        self.result_label.pack()
        self.confidence_label = tk.Label(right, font=(FONT, 14), fg=TEXT, bg=PANEL)
        self.confidence_label.pack()
        self.bars = tk.Canvas(right, width=BAR_WIDTH + 120, height=60, bg=PANEL, highlightthickness=0)
        self.bars.pack(padx=8, pady=(6, 0))
        self.state_label = tk.Label(right, font=(FONT, 12), fg=TEXT, bg=PANEL)
        self.state_label.pack(pady=(4, 8))

        # 下：これまでの判定
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            "Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=24, font=(FONT, 11), borderwidth=0
        )
        style.configure("Treeview.Heading", background=BG, foreground=SUBTEXT, font=(FONT, 10), borderwidth=0)
        style.map("Treeview", background=[("selected", PANEL)], foreground=[("selected", TEXT)])
        columns = ("round", "cube", "result", "confidence", "bin", "state")
        headings = ("ラウンド", "立方体", "判定", "信頼度", "置き場", "結果")
        widths = (80, 90, 120, 90, 180, 110)
        self.history = ttk.Treeview(root, columns=columns, show="headings", height=9, selectmode="none")
        for column, heading, width in zip(columns, headings, widths):
            self.history.heading(column, text=heading)
            self.history.column(column, width=width, anchor="center")
        self.history.tag_configure("ok", foreground=OK_COLOR)
        self.history.tag_configure("ng", foreground=NG_COLOR)
        self.history.grid(row=2, column=0, columnspan=2, sticky="we", padx=12, pady=(6, 12))

        self.show_waiting()
        self.poll()

    def show_waiting(self):
        self.header.config(text="phase4_sort.py の判定を待っています…")
        self.set_image(self.scene_label, "scene", Image.new("RGB", SCENE_SIZE, PANEL))
        self.set_image(self.crop_label, "crop", Image.new("RGB", (CROP_SIZE, CROP_SIZE), PANEL))
        for label in (self.cube_label, self.result_label, self.confidence_label, self.state_label):
            label.config(text="")
        self.bars.delete("all")
        self.history.delete(*self.history.get_children())

    def set_image(self, widget: tk.Label, key: str, image: Image.Image):
        self.photos[key] = ImageTk.PhotoImage(image)
        widget.config(image=self.photos[key])

    def poll(self):
        status = read_status()
        key = json.dumps(status, sort_keys=True) if status else None
        if key != self.last_key:
            try:
                self.refresh(status)
                self.last_key = key
            except OSError:
                pass  # 画像の書き込み途中などで読めなかったときは、次の周期で読み直す
        self.root.after(POLL_MS, self.poll)

    def refresh(self, status):
        if not status:
            self.show_waiting()
            return
        classes = {c["label"]: c for c in status["classes"]}
        history = status["history"]
        round_number = status.get("round", 1)
        placed = sum(h["state"] == "placed" and h.get("round", 1) == round_number for h in history)
        header = (
            f"seed = {status['seed']}    ルール: {status['rules']}    "
            f"ラウンド {round_number} / {status.get('rounds', 1)}    "
            f"このラウンド {placed} / {status.get('per_round', status['total'])}"
        )
        if status["done"]:
            ok = sum(bool(h["in_bin"]) for h in history)
            header += f"    完了（置き場に置けた数 {ok} / {status['total']}）"
        self.header.config(text=header)

        current = status["current"]
        if current is None:
            self.show_waiting()
            self.header.config(text=header)
            return

        scene = load_image(RUN_DIR / current["scene_image"])
        crop = load_image(RUN_DIR / current["crop_image"])
        draw = ImageDraw.Draw(scene)
        draw.rectangle(current["box"], outline=HIGHLIGHT, width=4)
        self.set_image(self.scene_label, "scene", scene.resize(SCENE_SIZE))
        self.set_image(self.crop_label, "crop", crop.resize((CROP_SIZE, CROP_SIZE)))

        predicted = classes[current["label"]]
        self.cube_label.config(text=current["cube"])
        self.result_label.config(text=predicted["name"], fg=predicted["color"])
        self.confidence_label.config(text=f"信頼度 {current['confidence'] * 100:.1f}%")
        self.draw_bars(status["classes"], current["scores"])

        if current["state"] == "placing":
            self.state_label.config(text=f"→ {predicted['name']}の置き場へ運んでいます…", fg=TEXT)
        elif current["in_bin"]:
            self.state_label.config(text="→ 置き場に置きました", fg=OK_COLOR)
        else:
            self.state_label.config(text="→ 置き場から外れました", fg=NG_COLOR)

        self.history.delete(*self.history.get_children())
        for h in history:
            name = classes[h["label"]]["name"]
            if h["state"] == "placing":
                result, tag = "運搬中", ()
            elif h["in_bin"]:
                result, tag = "OK", ("ok",)
            else:
                result, tag = "NG", ("ng",)
            values = (h.get("round", 1), h["cube"], name, f"{h['confidence'] * 100:.1f}%", f"{name}の置き場", result)
            self.history.insert("", "end", values=values, tags=tag)

    def draw_bars(self, classes: list, scores: dict):
        """クラスごとの確率を横棒で表示する."""
        self.bars.delete("all")
        row_height = 28
        self.bars.config(height=row_height * len(classes))
        for i, c in enumerate(classes):
            y = i * row_height + row_height / 2
            score = scores.get(c["label"], 0.0)
            self.bars.create_text(0, y, text=c["name"], anchor="w", fill=TEXT, font=(FONT, 11))
            self.bars.create_rectangle(80, y - 8, 80 + BAR_WIDTH, y + 8, fill=BG, outline="")
            self.bars.create_rectangle(80, y - 8, 80 + BAR_WIDTH * score, y + 8, fill=c["color"], outline="")
            self.bars.create_text(84 + BAR_WIDTH, y, text=f"{score * 100:.1f}%", anchor="w", fill=TEXT, font=(FONT, 11))


def main():
    try:
        # 高 DPI の画面で文字がぼやけないようにする（Windows のみ）
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    JudgeViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
