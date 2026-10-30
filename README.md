# 言葉で指示できるゴミ分別ロボット（Isaac Sim + LLM + CLIP）

Isaac Sim 上の Franka アームが、チャットで受け取った指示（例：「赤の箱を緑へ」）を LLM で解釈し、上からのカメラ画像から CLIP で対象の物体を探して、指定された置き場へ運びます。

Physical AI 応用編1 最終課題の提出物です。

- デモ動画：https://youtu.be/1Rlr5fJo5hg

このリポジトリには、次の2つのタスクが入っています。

| タスク | 内容 | 実行するスクリプト |
|---|---|---|
| **言語指示タスク**（デモ動画） | 赤・青の箱と円柱が1個ずつランダムに置かれた机から、チャットで指示された物体を探し、4色の置き場の指定された場所へ運ぶ | `src/instruct_sort.py` |
| 自動仕分けタスク | 赤・青の立方体3個を CLIP で「燃えるゴミ / 資源ゴミ」に判定し、2つの置き場へ仕分ける。配置を変えて3ラウンド繰り返す | `src/phase4_sort.py` |

## 言語指示タスクのしくみ

```
チャットに入力「赤の箱を緑へ」
  └─ LLM（OpenAI API）が {"object": "red_box", "bin": "green"} に変換 ……… AI その1：言葉の理解
       └─ 上からのカメラ画像から、机の上の物体を1個ずつ切り出す（224×224）
            └─ CLIP で物体ごとに4種類（赤の箱・赤の円柱・青の箱・青の円柱）の確率を求める ……… AI その2：画像の理解
                 └─ 「赤の箱」である確率（信頼度）がいちばん高い物体を選ぶ
                      └─ PickPlaceController（RMPflow）で掴み、緑の置き場の空いている場所へ置く
                           └─ 置いた位置が置き場の中にあるかを確かめ、チャットに報告する
```

- **役割分担**：言葉の理解は LLM、画像の理解は CLIP が担当します。LLM は物体の位置や見た目を知らず、CLIP は指示の文を読みません
- **誤った物体を運ばない確認**：信頼度がいちばん高い物体でも、その物体自身の判定（いちばん確率の高い種類）が指示と違うときは運ばず、「見つかりませんでした」と返します（すでに運んだ物体を指示されたときなど）
- **指示の例**：「青の円柱を赤のエリアに置いて」「赤の箱を黄色のエリアに置いて」「リセット」（物体を新しい位置に並べ直す）。LLM には、1文に複数の指示があれば順番に並べるよう指示しています
- **画面**：1920×1080 の画面で、左に Isaac Sim、右上に判定モニタ（カメラ画像と物体ごとの信頼度）、右下にチャットが並ぶように起動します
- **配置**：物体の位置はランダムで、物体どうしは 0.12m 以上離します。置き場は固定の位置で、それぞれ最大4個置けます

## 自動仕分けタスクのしくみ

```
上からのカメラ画像
  └─ 立方体ごとに切り出す（224×224）
       └─ CLIP でゼロショット判定 ……… AI の部分
            └─ 分別ルール（config/sorting_rules.yaml）で置き場を決める
                 └─ PickPlaceController で掴んで置き場へ運ぶ
                      └─ 置いた位置がその置き場の中にあるかを確かめる
```

- **分別ルール**：`config/sorting_rules.yaml` のプロンプトと置き場の対応を書き換えるだけで、どれをどこに仕分けるかが変わります
- **シーン**：1ラウンドで立方体3個を仕分けます。3個の位置（2列×3マスのグリッドから3マス）と色（赤・青。偏ることもある）はランダムです。ラウンドが終わると、Isaac Sim を開いたままシーンをリセットし、置き直した3個で次のラウンドを始めます（既定は3ラウンド）
- **判定モニタ**：Isaac Sim とは別のウィンドウに、判定に使った画像・判定結果・信頼度をリアルタイムに表示します

## 共通の点

- **CLIP**（`openai/clip-vit-base-patch32`）：学習はせず、推論だけで使います（ゼロショット）。クラスごとに複数のプロンプトを用意し、テキスト埋め込みを平均してから画像と比べます（プロンプトアンサンブル）。「信頼度」は CLIP が出したクラスごとの確率です（正解率ではありません）
- **物体の位置**：掴む位置はシミュレータの正解値を使います（カメラからの位置推定はしていません）
- **見た目**：床は無彩色にし、照明はドームライトだけにしています。床の色が切り出し画像の背景に入ると、CLIP の判定がその色に引っ張られるためです
- **再現**：配置は seed で決まります。ターミナルに表示される seed を `--seed` で指定すると、同じ配置を再現できます

## 動作環境

| 項目 | 確認した環境 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 3060 Ti（VRAM 8GB）、ドライバ 610.88 |
| 画面 | 1920×1080、表示倍率 100%（言語指示タスクのウィンドウ配置） |
| Python | 3.10（venv） |
| Isaac Sim | 4.5.0（pip 版） |
| torch | 2.5.1 + CUDA 12.1 |

Franka のモデルは実行時に NVIDIA のサーバーから読み込むため、インターネット接続が必要です。

## セットアップ

```powershell
py -3.10 -m venv <venv のパス>
<venv のパス>\Scripts\Activate.ps1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
pip install -r requirements.txt
```

CLIP のモデル（約600MB）は、初回実行時に `models/` へ自動でダウンロードされます。

### OpenAI の API キー（言語指示タスク）

チャットの指示の解釈に OpenAI API を使います（OpenAI の Python パッケージは不要で、Python 標準の `urllib` で呼びます）。

1. https://platform.openai.com/api-keys で API キーを作ります（プロジェクトに属する通常のキー。組織の管理用の Admin key ではモデルを呼べません）
2. このフォルダの直下に `.env` を作り、次の1行を書きます（`.env` は `.gitignore` で除外しています）

```
OPENAI_API_KEY=sk-...
```

- モデルは `config/instruct_rules.yaml` の `llm.model`（既定 `gpt-4o-mini`）で変えられます。1回の指示で使うトークンは 400 程度です
- API キーがない・通信できないときは、簡単なルールによる解析（「<色>の<形>を<色>へ」の形だけを読み取る）に自動で切り替わります

### Windows で Isaac Sim 4.5 が起動しない・描画されない場合

この環境では次の2点が必要でした。

- **描画を DirectX 12 にする**：新しい NVIDIA ドライバでは、Vulkan での描画がレイトレーシングの確認に失敗して止まることがありました。`src/app_config.py` で `--/app/vulkan=false` を指定しています（あわせてマルチ GPU を無効にし、GPU 0 番だけを使います）
- **同梱の古い VC++ ランタイムを無効にする**：起動直後に `MSVCP140.dll` で落ちる場合、`site-packages\omni\` と `site-packages\omni\kernel\plugins\` にある `msvcp140*.dll` / `vcruntime140*.dll` の名前を `.bak` に変えると、システムの新しいランタイムが使われて起動しました

## 使い方

venv を有効にし、このフォルダで実行します。

### 言語指示タスク

```powershell
python src\instruct_sort.py                     # 配置はランダム
python src\instruct_sort.py --seed 1            # 配置を固定する
python src\instruct_sort.py --device cpu        # GPU のメモリが足りないときは CLIP を CPU で動かす
```

起動すると、判定モニタ（右上）とチャット（右下）のウィンドウが開きます（`--no-ui` で開かない）。チャットに指示を入力すると、次のように表示されます。

```
あなた：赤の箱を黄色のエリアに置いて
LLM の出力：{"commands": [{"object": "red_box", "bin": "yellow"}], "reset": false, "reply": "赤の箱を黄色の置き場へ運びます。"}
トークン数：入力 392 / 出力 35 / 合計 427
アシスタント（LLM）：赤の箱を黄色の置き場へ運びます。
ロボット：物体1を赤の箱と判断しました（信頼度 88.5%）。黄の置き場へ運びます。
ロボット：物体1を黄の置き場に置きました。
```

ターミナルには、起動時に4個すべてを判定して正解と比べた結果（`self-check: 4/4`）と、指示ごとに選んだ物体の正解の種類も表示します。

ウィンドウの位置と大きさは `src/screen_layout.py` で変えられます。

### 自動仕分けタスク

```powershell
python src\phase4_sort.py                       # 3個 × 3ラウンド。配置はランダム
python src\phase4_sort.py --seed 858            # 配置を固定する
python src\phase4_sort.py --rounds 5            # ラウンド数を変える
python src\phase4_sort.py --rules <別の YAML>   # 別の分別ルールで動かす
python src\phase4_sort.py --no-viewer           # 判定モニタを開かない
```

```
[phase4] ===== round 1/3 =====
[phase4] round 1 cube_0: CLIP -> burnable (0.985) -> burnable_bin slot 0
[phase4] round 1 cube_0: placed at [0.302 0.396 0.036] in burnable_bin: OK
...
[phase4] placed in bin: 9/9
```

### 途中段階の確認用

```powershell
python src\phase1_pick_place.py                 # 立方体1個の pick & place
python src\phase2_capture.py                    # カメラ画像と切り出し画像を assets/crops/ に保存
python tests\test_classifier.py                 # assets/crops/ の画像で CLIP の判定を確かめる（Isaac Sim は起動しない）
```

## ファイル構成

| パス | 内容 |
|---|---|
| `src/instruct_sort.py` | 言語指示タスクの本体。指示を受け取り、CLIP で物体を探して運ぶ |
| `src/instruct_scene.py` | 言語指示タスクのシーン（箱と円柱4個・置き場4か所） |
| `src/instruct_ui.py` | 言語指示タスクの判定モニタとチャット（tkinter）。Isaac Sim とは別のプロセスで動く |
| `src/llm_parser.py` | チャットの指示の解析（OpenAI API、使えないときはルール） |
| `src/instruct_io.py` | 言語指示タスクのシミュレーションと画面のやり取り（状態ファイル・指示ファイル） |
| `src/screen_layout.py` | 言語指示タスクのウィンドウの位置と大きさ |
| `src/phase4_sort.py` | 自動仕分けタスクの本体（複数ラウンド） |
| `src/scene.py` | シーン（Franka・立方体・置き場・床・照明・カメラ）の組み立て、カメラ画像の取得、切り出し範囲の計算。言語指示タスクも床・照明・カメラはこれを使う |
| `src/classifier.py` | CLIP による判定（`TrashClassifier`）。Isaac Sim に依存しない |
| `src/judge_viewer.py` | 自動仕分けタスクの判定モニタ（tkinter） |
| `src/judge_status.py` | 自動仕分けタスクの判定モニタに渡す状態ファイルの読み書き |
| `src/app_config.py` | Isaac Sim の起動設定と、スクリプト引数の処理 |
| `src/phase1_pick_place.py` | 立方体1個を掴んで置く確認 |
| `src/phase2_capture.py` | カメラ画像と立方体ごとの切り出し画像を保存する |
| `config/instruct_rules.yaml` | 言語指示タスクの設定（物体4種類のプロンプト、置き場、LLM のモデル） |
| `config/sorting_rules.yaml` | 自動仕分けタスクの分別ルール（クラス・プロンプト・置き場） |
| `tests/test_classifier.py` | 保存した切り出し画像で CLIP の判定を確かめる |
| `assets/crops/` | `phase2_capture.py` で保存したテスト用の画像 |

## 参考にしたもの・使用したもの

- [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/) 4.5：Franka のモデル、`PickPlaceController`（RMPflow）、カメラ。立方体1個の pick & place は公式のチュートリアルとサンプルを参考にしています
- [CLIP](https://github.com/openai/CLIP)（`openai/clip-vit-base-patch32`、Hugging Face Transformers 経由）
- [OpenAI API](https://platform.openai.com/docs/)（Chat Completions、JSON 形式の出力）

公式の pick & place から追加・変更した点：上からのカメラによる切り出しと CLIP による判定、LLM によるチャットの指示の解析、指示と物体の照合（信頼度による選択と、誤った物体を運ばない確認）、箱と円柱・4つの置き場、置き場の中の置き場所の割り当て、置いた結果の確認、判定モニタとチャットの画面、配置のランダム化とリセット。
