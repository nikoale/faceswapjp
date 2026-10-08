# faceswapjp

映像制作向けの、ローカルで完結するフェイススワップツールです。
本人の同意を得た出演者、自分自身、AI 生成の架空の顔だけを対象にします。

- 設計：[docs/DESIGN.md](docs/DESIGN.md)
- モデルのライセンス：[docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md)

> **ライセンス**：既定のモデル（InsightFace buffalo_l / inswapper_128 ほか）は**非商用ライセンス**です。
> 自主制作・検証用途に限って使ってください。

## 進捗

| マイルストーン | 状態 |
|---|---|
| M1 静止画 1 枚のスワップ（CLI） | ✅ |
| M2 動画対応（音声・タイムコード・fps・色タグの保持） | ✅ |
| M3 人物指定・トラッキング・スムージング・オクルージョン | ✅ |
| M4 Gradio UI、同意ログ、メタデータ、透かし、NSFW ゲート | ✅ |
| M5 マット書き出し、ProRes、バッチ処理 | ✅ |

## セットアップ（macOS / Apple Silicon）

```bash
brew install python@3.11 ffmpeg
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[mac,ui]"       # NVIDIA 環境では ".[cuda,ui]"
faceswapjp models download       # ~/.faceswapjp/models に取得して SHA-256 を検証（約 1.2 GB）
faceswapjp doctor                # 実行プロバイダ、エンコーダ、モデルの状態を表示
```

モデルの置き場所は環境変数 `FACESWAPJP_MODELS_DIR` で変えられます。

## UI（はじめての方はこちら）

```bash
faceswapjp ui
```

ブラウザが自動で開きます（http://127.0.0.1:7860）。画面の指示に沿って 5 つのステップを進めるだけです。

| ステップ | やること |
|---|---|
| 1. 使う顔 | 差し替えに使う顔を選ぶか、写真と本人の同意（氏名・日付・同意書）を入れて登録します |
| 2. 素材 | 差し替えたい動画・写真をドラッグ＆ドロップします |
| 3. 置き換える人 | 見つかった顔に番号が付くので、置き換える人の顔をクリックします |
| 4. 仕上がり確認 | 元と差し替え後をスライダーで見比べます。必要なら「境目のなじませ」などを調整します |
| 5. 書き出し | 「確認・共有用（MP4）」か「編集用の高画質（ProRes）」を選んで書き出します |

- **初回の準備**：AI モデルのファイルがない場合は、画面上部に「必要なファイルをダウンロード」ボタンが出ます。
- **作品**：作品ごとに、同意の記録と書き出したファイルを別々のフォルダに保存します。初回は「はじめての作品」が自動で作られます。
- **まとめて処理**：フォルダ内のファイルを、「かんたん作成」で選んだ設定でまとめて処理します。
- **同意の記録**：これまでの登録と書き出しの履歴を表示し、書き換えられていないかを確認できます。
- **プライバシー**：`127.0.0.1` だけで待ち受けます。共有リンクは作らず、利用統計も送信しません。

## CLI

```bash
# プロジェクトとソース顔（氏名・同意日・同意書は必須）
faceswapjp project init ./myproj
faceswapjp identity add -p ./myproj --label "Stunt A" -i a1.jpg -i a2.jpg \
  --person-name "山田 太郎" --consent-date 2026-10-01 \
  --consent-doc ./consent/yamada.pdf --source-type consented_person

# プレビュー（元 | 結果 を横に並べた PNG）
faceswapjp preview -p ./myproj --identity stunt-a --target clip.mov --frame 120 --out prev.png \
  --only-person stunt_ref.jpg --mask box,occlusion,region

# 動画：ProRes 422 HQ ＋ ルミナンスマット
faceswapjp swap video -p ./myproj --identity stunt-a --target clip.mov --out out.mov \
  --format prores422hq --matte luma --only-person stunt_ref.jpg --mask box,occlusion

# 静止画
faceswapjp swap image -p ./myproj --identity stunt-a --target shot.png --out out.png --matte out_matte.png

# フォルダ一括（失敗したクリップはスキップし、batch_report.json に記録）
faceswapjp batch -p ./myproj --identity stunt-a --input-dir ./clips --output-dir ./out --format h264

faceswapjp log verify -p ./myproj
```

### 主なオプション

| オプション | 内容 |
|---|---|
| `--only-person img`（複数可） | 参照画像の人物だけを置き換えます。判定はトラック単位です |
| `--select all\|largest\|reference` | 置き換え対象 |
| `--ref-threshold` | 人物一致のしきい値（既定 0.4） |
| `--mask box,occlusion,region` | `occlusion`：手や小道具を前景に残す。`region`：髪・耳・首を除外 |
| `--smoothing 0..1` | ランドマークの時間方向スムージング（既定 0.5、0 でオフ） |
| `--enhance gfpgan` / `--enhance-blend` | 顔補正 |
| `--color-strength`、`--mask-blur` | 色合わせの強さ、マスクのぼかし |
| `--format h264\|prores422hq\|prores4444` | 出力形式（h264 は `.mp4`、ProRes は `.mov`） |
| `--encoder` | エンコーダを強制（例：`prores_videotoolbox`、`h264_videotoolbox`） |
| `--matte luma\|alpha` | `<出力名>_matte.mov` も書き出します。luma は白黒の ProRes 422 HQ、alpha は ProRes 4444 のフィル＋アルファ |
| `--start` / `--end` | フレーム範囲。タイムコードと音声は範囲に合わせてずらします |
| `--watermark` / `--watermark-text` / `--watermark-position` / `--watermark-font` | 目に見える透かし。日本語テキストを使う場合はフォントファイルを指定します |
| `--device auto\|coreml\|cuda\|cpu` | 実行デバイス |

## 元素材との一致

- **fps**：有理数のまま引き継ぎます（例：`24000/1001`）。
- **フレーム数**：書き出し後に ffprobe で照合し、ずれていれば警告します。
- **解像度・SAR**：引き継ぎます。
- **色**：色タグ（primaries / transfer / matrix / range）を引き継ぎます。
- **タイムコード**：引き継ぎます。範囲指定時は開始フレームの分だけずらします（ドロップフレーム対応）。
- **音声**：コンテナが対応していれば再エンコードせずにそのままコピーします。範囲指定時は切り出して再エンコードします。
- **ビット深度**：10bit 以上の素材は 16bit で読み込んで処理します。
- **警告の出る素材**：可変フレームレート、インターレース、アルファ付きの素材は警告を出します。
- **ProRes エンコーダ**：既定は FFmpeg の `prores_ks` です。Apple 純正のエンコーダが必要な場合は、Mac で `--encoder prores_videotoolbox` を指定してください。

## DaVinci Resolve での使い方

マットは本編と同じ fps・解像度・タイムコードで書き出されます。

1. 本編とマットをメディアプールに読み込みます。
2. カラーページで本編のノードにマットを「外部マット」として追加するか、Fusion で MatteControl に接続します。
3. `--matte alpha` で書き出した場合は、差し替え部分がアルファ付きのフィルになっているので、元素材の上に重ねるだけで合成と修正ができます。

## 安全機能（無効化できません）

- **ソース顔の登録**：氏名・同意日・同意書が必須です。
- **同意ログ**：プロジェクトごとの `consent_log.json` に、登録と書き出しをすべて記録します。各エントリには前のエントリのハッシュが含まれ、改ざんを検知できます。
- **出力メタデータ**：出力ファイルに「AI face-swapped」を書き込みます（MOV / MP4 は comment / description、PNG は iTXt、JPEG は COM）。使用したモデルとソース顔の ID も JSON で記録します。
- **NSFW ゲート**：参照画像・ターゲット画像の全枚と、動画の 1 秒ごとのフレームおよびシーンチェンジ時のフレームを判定します。性的コンテンツと判定された時点で処理を止めます。

## テスト

```bash
pip install -e ".[cpu,dev]"
pytest                                   # モデル不要（ffmpeg を使うテストを含む）
FACESWAPJP_TEST_SOURCE=me.jpg FACESWAPJP_TEST_TARGET=me2.jpg pytest -m models   # 実モデルでのテスト
```
