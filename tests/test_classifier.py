"""フェーズ2で保存した切り出し画像を CLIP で判定し、立方体の色から決まる正解と比べる.

Isaac Sim は起動しない。実行：python tests\\test_classifier.py
"""
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classifier import load_classifier  # noqa: E402

CROPS_DIR = Path(__file__).resolve().parents[1] / "assets" / "crops"
# 正解：赤 = 燃えるゴミ、青 = 資源ゴミ（config/sorting_rules.yaml の既定ルール）
EXPECTED = {"red": "burnable", "blue": "recyclable"}


def main() -> int:
    labels = json.loads((CROPS_DIR / "labels.json").read_text(encoding="utf-8"))

    start = time.perf_counter()
    classifier = load_classifier()
    print(f"[phase3] model loaded on {classifier.device} in {time.perf_counter() - start:.1f}s")

    correct = 0
    for item in labels:
        image = Image.open(CROPS_DIR / item["crop_file"]).convert("RGB")
        start = time.perf_counter()
        pred = classifier.classify(image)
        elapsed_ms = (time.perf_counter() - start) * 1000
        expected = EXPECTED[item["color"]]
        ok = pred.label == expected
        correct += ok
        scores = " ".join(f"{k}={v:.3f}" for k, v in pred.scores.items())
        print(
            f"[phase3] {item['name']} ({item['color']}): {pred.label} ({pred.confidence:.3f}) "
            f"expected={expected} {'OK' if ok else 'NG'} [{scores}] {elapsed_ms:.0f}ms"
        )

    print(f"[phase3] accuracy: {correct}/{len(labels)}")
    return 0 if correct == len(labels) else 1


if __name__ == "__main__":
    sys.exit(main())
