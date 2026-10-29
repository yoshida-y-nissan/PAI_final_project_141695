# ゴミ分別ロボット（Isaac Sim + CLIP）

Isaac Sim 上の Franka アームが、床に置かれた赤と青の立方体を上からのカメラで見て、CLIP のゼロショット推論で「燃えるゴミ / 資源ゴミ」に判定し、それぞれの置き場へ仕分けます。

Physical AI 応用編1 最終課題の提出物です。

- デモ動画：（YouTube のリンクを追記予定）

## しくみ

```
上からのカメラ画像
  └─ 立方体ごとに切り出す（224×224）
       └─ CLIP（openai/clip-vit-base-patch32）でゼロショット判定 ……… AI の部分
            └─ 分別ルール（config/sorting_rules.yaml）で置き場を決める
                 └─ PickPlaceController（RMPflow）で掴んで置き場へ運ぶ
                      └─ 置いた位置がその置き場の中にあるかを確かめる
```

- **判定（CLIP）**：クラスごとに複数のプロンプトを用意し、テキスト埋め込みを平均してから画像と比べます（プロンプトアンサンブル）。学習はしていません
- **分別ルール**：`config/sorting_rules.yaml` のプロンプトと置き場の対応を書き換えるだけで、どれをどこに仕分けるかが変わります
- **シーン**：1ラウンドで立方体3個を仕分けます。3個の位置（2列×3マスのグリッドから3マス）と色（赤・青。偏ることもある）はランダムです。ラウンドが終わると、Isaac Sim を開いたままシーンをリセットし、置き直した3個で次のラウンドを始めます（既定は3ラウンド）。各置き場には最大6個（2列×3個）置けます
- **物体の位置**：掴む位置はシミュレータの正解値を使います（カメラからの位置推定はしていません）

## 動作環境

| 項目 | 確認した環境 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 3060 Ti（VRAM 8GB）、ドライバ 610.88 |
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

### Windows で Isaac Sim 4.5 が起動しない・描画されない場合

この環境では次の2点が必要でした。

- **描画を DirectX 12 にする**：新しい NVIDIA ドライバでは、Vulkan での描画がレイトレーシングの確認に失敗して止まることがありました。`src/app_config.py` で `--/app/vulkan=false` を指定しています（あわせてマルチ GPU を無効にし、GPU 0 番だけを使います）
- **同梱の古い VC++ ランタイムを無効にする**：起動直後に `MSVCP140.dll` で落ちる場合、`site-packages\omni\` と `site-packages\omni\kernel\plugins\` にある `msvcp140*.dll` / `vcruntime140*.dll` の名前を `.bak` に変えると、システムの新しいランタイムが使われて起動しました

## 使い方

venv を有効にし、このフォルダで実行します。

```powershell
# 仕分けの本体（3個 × 3ラウンド。配置はランダムで、表示される seed を指定すると同じ配置を再現できる）
python src\phase4_sort.py
python src\phase4_sort.py --seed 858            # 配置を固定する
python src\phase4_sort.py --rounds 5            # ラウンド数を変える
python src\phase4_sort.py --rules <別の YAML>   # 別の分別ルールで動かす
python src\phase4_sort.py --device cpu          # GPU のメモリが足りないときは CLIP を CPU で動かす

# 途中段階の確認用
python src\phase1_pick_place.py                 # 立方体1個の pick & place
python src\phase2_capture.py                    # カメラ画像と切り出し画像を assets/crops/ に保存
python tests\test_classifier.py                 # assets/crops/ の画像で CLIP の判定を確かめる（Isaac Sim は起動しない）
```

`phase4_sort.py` を実行すると、Isaac Sim とは別に「CLIP 判定モニタ」のウィンドウが開き、判定のたびに次の内容がリアルタイムに切り替わります（`--no-viewer` で開かない）。

- 上からのカメラ画像（判定する立方体を黄色の枠で表示）と、CLIP に入れた切り出し画像
- 判定結果、信頼度（CLIP が出したクラスごとの確率）、クラスごとの確率の横棒
- これまでの判定と、置き場に置けたか（OK / NG）の一覧

ターミナルにも、立方体ごとの判定結果と置いた結果を表示し、最後に集計を出します。

```
[phase4] ===== round 1/3 =====
[phase4] round 1 cube_0: CLIP -> burnable (0.985) -> burnable_bin slot 0
[phase4] round 1 cube_0: placed at [0.302 0.396 0.036] in burnable_bin: OK
...
[phase4] placed in bin: 9/9
```

## ファイル構成

| パス | 内容 |
|---|---|
| `src/app_config.py` | Isaac Sim の起動設定と、スクリプト引数の処理 |
| `src/scene.py` | シーン（Franka・立方体・置き場・床・照明・カメラ）の組み立て、カメラ画像の取得、切り出し範囲の計算、置き場所の計算 |
| `src/classifier.py` | CLIP による判定（`TrashClassifier`）。Isaac Sim に依存しない |
| `src/phase1_pick_place.py` | 立方体1個を掴んで置く確認 |
| `src/phase2_capture.py` | カメラ画像と立方体ごとの切り出し画像を保存する |
| `src/phase4_sort.py` | 判定と仕分けの統合スクリプト（複数ラウンド） |
| `src/judge_viewer.py` | 判定モニタ（tkinter）。Isaac Sim とは別のプロセスで動く |
| `src/judge_status.py` | 判定モニタに渡す状態ファイル（`assets/run/status.json`）の読み書き |
| `config/sorting_rules.yaml` | 分別ルール（クラス・プロンプト・置き場） |
| `tests/test_classifier.py` | 保存した切り出し画像で CLIP の判定を確かめる |
| `assets/crops/` | `phase2_capture.py` で保存したテスト用の画像 |

## 参考にしたもの・使用したもの

- [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/) 4.5：Franka のモデル、`PickPlaceController`（RMPflow）、カメラ。立方体1個の pick & place は公式のチュートリアルとサンプルを参考にしています
- [CLIP](https://github.com/openai/CLIP)（`openai/clip-vit-base-patch32`、Hugging Face Transformers 経由）

公式の pick & place から追加・変更した点：上からのカメラによる切り出しと CLIP による判定、設定ファイルによる分別ルール、複数の立方体と2つの置き場への仕分け、置き場の中の置き場所の割り当て、置いた結果の確認と集計、色の並びのランダム化。
