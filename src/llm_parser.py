"""チャットの指示（例：「赤の箱を緑へ」）を、「どの物体を・どの置き場へ」の形に変換する.

OpenAI API（Chat Completions）の LLM で解釈する。API キーがない・通信できないときは、簡単なルールで解析する。
API は Python 標準の urllib で呼ぶ（openai パッケージは不要）。isaacsim に依存しない。
"""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.openai.com/v1/chat/completions"

# ルールによる解析で使う言い換え
COLOR_WORDS = {
    "red": ["赤", "あか", "アカ", "red"],
    "blue": ["青", "あお", "アオ", "blue"],
    "green": ["緑", "みどり", "ミドリ", "green"],
    "yellow": ["黄", "きいろ", "キイロ", "yellow"],
}
SHAPE_WORDS = {
    "box": ["箱", "はこ", "ボックス", "キューブ", "立方体", "四角", "box", "cube"],
    "cylinder": ["円柱", "えんちゅう", "シリンダー", "筒", "丸", "まる", "cylinder"],
}
RESET_WORDS = ["リセット", "並べ直", "やり直", "reset"]


def load_api_key():
    """環境変数 OPENAI_API_KEY か、プロジェクト直下の .env から API キーを読む. なければ None."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENAI_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def _system_prompt(objects: list, bins: list) -> str:
    object_lines = "\n".join(f'- "{o["label"]}": {o["display_name"]}' for o in objects)
    bin_lines = "\n".join(f'- "{b["label"]}": {b["display_name"]}' for b in bins)
    return f"""あなたはロボットアームへの指示を解釈するアシスタントです。
机の上の物体（ラベル: 日本語名）:
{object_lines}
置き場（ラベル: 日本語名）:
{bin_lines}

ユーザーの文から「どの物体を、どの置き場へ運ぶか」を読み取り、次の JSON だけを返してください。
{{"commands": [{{"object": "<物体のラベル>", "bin": "<置き場のラベル>"}}], "reset": false, "reply": "<日本語の短い返事>"}}

- 1つの文に複数の指示があれば、commands に順番どおり並べる
- 色や形の言い換え（例：丸いの→円柱、四角いの→箱、黄色→yellow）も解釈する
- 物体の配置をやり直す・リセットする指示なら reset を true にし、commands は空にする
- 物体か置き場が分からない・上の一覧にないときは、commands を空にし、reply で聞き返す
- reply は「赤の箱を緑の置き場へ運びます。」のように、何をするかを1文で伝える"""


def _describe_http_error(error: urllib.error.HTTPError) -> str:
    """OpenAI のエラー応答から、種類と説明を取り出す. API キーが含まれていても伏せ字にする."""
    try:
        detail = json.loads(error.read().decode("utf-8"))["error"]
        text = f"{error.code} {detail.get('code') or detail.get('type')}: {detail.get('message', '')}"
    except (OSError, ValueError, KeyError, TypeError):
        text = f"{error.code} {error.reason}"
    return re.sub(r"sk-[A-Za-z0-9_\-*]+", "sk-***", text)


def parse_with_llm(text: str, objects: list, bins: list, model: str, api_key: str, timeout: float) -> tuple:
    """LLM で解析する. 戻り値は (解析結果, LLM が出力した文字列そのもの, トークン数)."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt(objects, bins)},
            {"role": "user", "content": text},
        ],
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    content = result["choices"][0]["message"]["content"]
    return json.loads(content), content, result.get("usage", {})


def _find(words_by_key: dict, text: str):
    """text に最初に出てくる言葉のキーを返す. なければ None."""
    best = None
    for key, words in words_by_key.items():
        for word in words:
            i = text.find(word)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, key)
    return best[1] if best else None


def parse_with_rules(text: str, objects: list, bins: list) -> dict:
    """「<色>の<形>を<色>へ」の形だけを読み取る簡単な解析."""
    lowered = text.lower()
    if any(word in lowered for word in RESET_WORDS):
        return {"commands": [], "reset": True, "reply": "物体を並べ直します。"}
    object_labels = {o["label"] for o in objects}
    bin_labels = {b["label"] for b in bins}
    commands = []
    for part in re.split(r"[、。,.;\n]|そして|それから|and", lowered):
        if "を" in part:
            target, _, destination = part.partition("を")
        elif " to " in part:
            target, _, destination = part.partition(" to ")
        else:
            continue
        color, shape, bin_label = _find(COLOR_WORDS, target), _find(SHAPE_WORDS, target), _find(COLOR_WORDS, destination)
        label = f"{color}_{shape}"
        if label in object_labels and bin_label in bin_labels:
            commands.append({"object": label, "bin": bin_label})
    if not commands:
        return {"commands": [], "reset": False, "reply": "指示が読み取れませんでした。「赤の箱を緑へ」のように指示してください。"}
    return {"commands": commands, "reset": False, "reply": ""}


def parse_instruction(text: str, config: dict) -> dict:
    """指示を解析する.

    戻り値は {"commands": [...], "reset": bool, "reply": str, "source": "llm" | "rules", "note": str,
    "raw": LLM が出力した文字列（LLM を使ったときだけ）, "usage": トークン数（同）}。
    """
    objects, bins = config["objects"], config["bins"]
    llm = config.get("llm", {})
    api_key = load_api_key()
    note = ""
    result = None
    raw, usage = "", {}
    if api_key:
        try:
            result, raw, usage = parse_with_llm(
                text, objects, bins, llm.get("model", "gpt-4o-mini"), api_key, llm.get("timeout", 20)
            )
            source = "llm"
        except urllib.error.HTTPError as e:
            note = f"LLM を使えなかったため、ルールで解析しました（HTTP {_describe_http_error(e)}）"
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            note = f"LLM を使えなかったため、ルールで解析しました（{type(e).__name__}: {e}）"
    else:
        note = "API キーが未設定のため、ルールで解析しました"
    if result is None:
        result = parse_with_rules(text, objects, bins)
        source = "rules"

    # LLM の出力に一覧にない物体・置き場が入っていたら除く
    object_labels = {o["label"] for o in objects}
    bin_labels = {b["label"] for b in bins}
    commands = [
        {"object": c.get("object"), "bin": c.get("bin")}
        for c in result.get("commands") or []
        if c.get("object") in object_labels and c.get("bin") in bin_labels
    ]
    return {
        "commands": commands,
        "reset": bool(result.get("reset")),
        "reply": str(result.get("reply") or ""),
        "source": source,
        "note": note,
        "raw": raw,
        "usage": usage,
    }
