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

## UI（faceswapjp studio）

```bash
pip install -e ".[mac,ui]"
faceswapjp ui            # ブラウザで http://127.0.0.1:7860 が開きます
```

動画編集ソフトのような画面で、左のステップを上から順に進めます。

| ステップ | やること |
|---|---|
| 1. 使う顔 | 顔ライブラリから選ぶか、「新しい顔を登録」で写真・氏名・日付を入れて登録します。「この顔を使う権利がある」ことの確認は必須、同意書の添付は任意です |
| 2. 素材 | 動画・写真を画面にドラッグ＆ドロップします。数 GB の素材はパスで指定すると速く読み込めます |
| 3. 置き換える人 | プレビュー上の顔をクリックすると緑の枠になり、その人だけが置き換わります。選んだ人は動画全体で自動的に追いかけます |
| 4. 仕上がり | 比較スライダーで元と差し替え後を見比べます。設定を変えるとプレビューが自動で更新されます |
| 5. 書き出し | 「確認・共有用」か「編集用」を選ぶか、「形式を細かく選ぶ」でコーデック・画質・エンコーダを指定して書き出します |

### 操作

- **タイムライン**：場面の移動に使います。
- **キーボード**：`←` `→` で 1 フレーム、`Shift` を押しながらで 10 フレーム移動します。`I` `O` で書き出す範囲を指定します。
- **まとめて処理**：今の設定で、フォルダ内のファイルを一括処理します。
- **同意の記録**：これまでの登録と書き出しの履歴を表示し、書き換えられていないかを確認します。

### プライバシーと以前の UI

- **通信**：`127.0.0.1` だけで待ち受けます。外部のフォントやスクリプトは読み込まず、通信はすべてこのパソコンの中で完結します。
- **他のサイトからの操作**：起動ごとに作られるトークンと Host ヘッダの確認で防いでいます。
- **以前の UI**：Gradio 版は `pip install -e ".[classic-ui]"` のうえ `faceswapjp ui --classic` で使えます。

## CLI

```bash
# プロジェクトとソース顔（氏名・同意日・--confirm-consent は必須、同意書は任意）
faceswapjp project init ./myproj
faceswapjp identity add -p ./myproj --label "Stunt A" -i a1.jpg -i a2.jpg \
  --person-name "山田 太郎" --consent-date 2026-10-01 \
  --source-type consented_person --confirm-consent --consent-doc ./consent/yamada.pdf

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
| `--format` | 出力形式。下の表を参照（`faceswapjp formats` で一覧と使えるエンコーダを表示） |
| `--quality high\|standard\|light` | 画質（H.264 / H.265） |
| `--encoder auto\|software\|hardware` | エンコーダの種類。`h264_videotoolbox` のように名前で指定することもできます |
| `--matte luma\|alpha` | `<出力名>_matte.mov` も書き出します。luma は白黒の ProRes 422 HQ、alpha は ProRes 4444 のフィル＋アルファ |
| `--start` / `--end` | フレーム範囲。タイムコードと音声は範囲に合わせてずらします |
| `--watermark` / `--watermark-text` / `--watermark-position` / `--watermark-font` | 目に見える透かし。日本語テキストを使う場合はフォントファイルを指定します |
| `--device auto\|coreml\|cuda\|cpu` | 実行デバイス |

## 出力形式

| `--format` | 内容 | 拡張子 |
|---|---|---|
| `h264` | H.264。確認・共有用 | .mp4 |
| `h265` | H.265 / HEVC。10bit 素材は 10bit のまま | .mp4 |
| `prores_proxy` / `prores_lt` / `prores422` / `prores422hq` | ProRes 422 系 | .mov |
| `prores4444` / `prores4444xq` | ProRes 4444 系 | .mov |
| `dnxhr_lb` / `dnxhr_sq` / `dnxhr_hq` | DNxHR（8bit 4:2:2） | .mov |
| `dnxhr_hqx` / `dnxhr_444` | DNxHR（10bit） | .mov |

- **ハードウェアエンコーダ**：Apple の VideoToolbox（H.264 / H.265 / ProRes）と NVIDIA の NVENC（H.264 / H.265）を、使える環境なら `--encoder hardware` で選べます。
- **UI**：「書き出し」ステップの「書き出し形式を細かく選ぶ」で、同じ設定ができます。

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

- **ソース顔の登録**：氏名・同意日と、「この顔を使う権利がある」ことの確認が必須です。同意書ファイルの添付は任意です。
- **同意ログ**：プロジェクトごとの `consent_log.json` に、登録と書き出しをすべて記録します。各エントリには前のエントリのハッシュが含まれ、改ざんを検知できます。
- **出力メタデータ**：出力ファイルに「AI face-swapped」を書き込みます（MOV / MP4 は comment / description、PNG は iTXt、JPEG は COM）。使用したモデルとソース顔の ID も JSON で記録します。
- **NSFW ゲート**：参照画像・ターゲット画像の全枚と、動画の 1 秒ごとのフレームおよびシーンチェンジ時のフレームを判定します。性的コンテンツと判定された時点で処理を止めます。

## テスト

```bash
pip install -e ".[cpu,dev]"
pytest                                   # モデル不要（ffmpeg を使うテストを含む）
FACESWAPJP_TEST_SOURCE=me.jpg FACESWAPJP_TEST_TARGET=me2.jpg pytest -m models   # 実モデルでのテスト
```
