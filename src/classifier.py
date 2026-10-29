"""CLIP のゼロショット推論で、切り出し画像をゴミの種類（分別ルールのクラス）に判定する.

isaacsim に依存しないので、Isaac Sim を起動せずに単体で試せる。
"""
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "sorting_rules.yaml"


@dataclass
class Prediction:
    label: str
    bin: str
    confidence: float  # 判定したクラスの確率（全クラスの合計が 1）
    scores: dict  # {label: 確率}


class TrashClassifier:
    def __init__(self, model_name: str, classes: list, cache_dir=None, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.classes = classes
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.text_features = self._encode_class_prompts()

    @torch.no_grad()
    def _encode_class_prompts(self) -> torch.Tensor:
        """クラスごとにプロンプトの埋め込みを平均し、(クラス数, 次元) の正規化済みテンソルにする."""
        features = []
        for c in self.classes:
            inputs = self.processor(text=c["prompts"], return_tensors="pt", padding=True).to(self.device)
            emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean = emb.mean(dim=0)
            features.append(mean / mean.norm())
        return torch.stack(features)

    @torch.no_grad()
    def classify(self, image) -> Prediction:
        """PIL.Image を受け取り、最も近いクラスを返す."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        logits = self.model.logit_scale.exp() * emb @ self.text_features.T
        probs = logits.softmax(dim=-1)[0].tolist()
        best = max(range(len(probs)), key=probs.__getitem__)
        return Prediction(
            label=self.classes[best]["label"],
            bin=self.classes[best]["bin"],
            confidence=probs[best],
            scores={c["label"]: p for c, p in zip(self.classes, probs)},
        )


def load_classifier(config_path=DEFAULT_CONFIG, device=None) -> TrashClassifier:
    """設定ファイルから分類器を作る. device を指定すると設定ファイルの device より優先する."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cache_dir = config.get("cache_dir")
    if cache_dir is not None:
        cache_dir = str(PROJECT_ROOT / cache_dir)
    return TrashClassifier(
        model_name=config["model"],
        classes=config["classes"],
        cache_dir=cache_dir,
        device=device or config.get("device", "auto"),
    )
