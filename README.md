# faceswapjp

映像制作向けの、ローカルで完結するフェイススワップツール（開発中）。
本人の同意を得た出演者、自分自身、AI 生成の架空の顔だけを対象にします。

- 設計：[docs/DESIGN.md](docs/DESIGN.md)
- モデルのライセンス：[docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md)

> **ライセンス**：既定のモデル（InsightFace buffalo_l / inswapper_128）は**非商用ライセンス**です。
> 自主制作・検証用途に限って使ってください。

## 進捗

| マイルストーン | 状態 |
|---|---|
| M1 静止画 1 枚のスワップ（CLI） | ✅ |
| M2 動画対応（音声保持、ffmpeg 連携） | 未着手 |
| M3 人物指定・トラッキング・スムージング | 未着手 |
| M4 Gradio UI、同意ログ、メタデータ | 一部実装済み（同意ログ、画像メタデータ） |
| M5 マット書き出し、ProRes、バッチ処理 | 未着手 |

## セットアップ（macOS / Apple Silicon）

```bash
brew install python@3.11 ffmpeg
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[mac,dev]"      # NVIDIA 環境では ".[cuda,dev]"
faceswapjp models download       # ~/.faceswapjp/models に取得して SHA-256 を検証
faceswapjp doctor                # 実行プロバイダ（CoreML / CUDA / CPU）とモデルの状態を表示
```

モデルの置き場所は環境変数 `FACESWAPJP_MODELS_DIR` で変えられます。

## 使い方（M1：静止画）

```bash
# 1. プロジェクトを作る
faceswapjp project init ./myproj

# 2. ソース顔を登録する（氏名・同意日・同意書は必須）
faceswapjp identity add -p ./myproj --label "Stunt A" \
  -i a1.jpg -i a2.jpg \
  --person-name "山田 太郎" --consent-date 2026-10-01 \
  --consent-doc ./consent/yamada.pdf --source-type consented_person
faceswapjp identity list -p ./myproj

# 3. スワップする（出力に "AI face-swapped" のメタデータが入る）
faceswapjp swap image -p ./myproj --identity stunt-a \
  --target shot.png --out out.png --faces largest --matte out_matte.png

# 同意ログの改ざんチェック
faceswapjp log verify -p ./myproj
```

主なオプション：

- `--faces all|largest`：置き換える顔（全員か、一番大きい顔か）
- `--mask-blur`：マスクのぼかし幅（顔の切り出し画像に対する比率、既定 0.12）
- `--color-strength`：色合わせの強さ（0〜1、既定 0.5）
- `--device auto|coreml|cuda|cpu`：実行デバイス

16bit の PNG / TIFF を入力すると、16bit のまま処理して出力します。顔以外の画素はビット単位で元と同じ値のまま残ります。

## テスト

```bash
pytest                                   # モデル不要の単体テスト
FACESWAPJP_TEST_SOURCE=me.jpg FACESWAPJP_TEST_TARGET=me2.jpg pytest -m models   # 実モデルでのテスト
```
