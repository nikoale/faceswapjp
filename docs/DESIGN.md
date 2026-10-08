# faceswapjp 設計提案（v0.1 / 着手前レビュー用）

映像制作向け・ローカル完結型フェイススワップツール。
本人同意のある出演者、自分自身、AI 生成の架空の顔のみを対象とする。

---

## 1. 設計方針

| 方針 | 内容 |
|---|---|
| ローカル完結 | 推論・エンコードはすべてローカル。初回のモデル取得以外はネットワークに出ない。モデルは `models/` に手動配置も可能で、SHA-256 で検証する。Gradio は `127.0.0.1` バインド固定、`share=False` 固定、analytics 無効 |
| コアと UI の分離 | すべての処理は `faceswapjp.pipeline` の関数として実装。CLI（Typer）と Gradio はそれを呼ぶだけの薄い層 |
| モデルはプラグイン | 検出・識別・スワップ・補正・オクルージョン・NSFW を、それぞれ Protocol（インターフェース）で抽象化。モデル manifest にライセンス情報も持たせる |
| 画質を落とさない | 元フレームは 16bit（rgb48）で読み、顔領域だけ float32 で処理して貼り戻す。顔以外の画素は元素材と同じ値のまま |
| 安全機能は迂回不可 | 同意情報の未入力、NSFW 判定、メタデータ書き込みは、パイプラインの必須ステップ。CLI にスキップ用フラグは用意しない |

## 2. 全体アーキテクチャ

```
           ┌──────────── CLI (Typer) ────────────┐  ┌──── Gradio UI (localhost) ────┐
           └──────────────────┬──────────────────┘  └───────────────┬───────────────┘
                              ▼                                     ▼
                     ┌──────────────── Project（プロジェクト単位の状態）────────────────┐
                     │ identities/（ソース顔） consent_log.json  renders/  previews/  │
                     └────────────────────────────┬─────────────────────────────────┘
                                                  ▼
   ┌─────────────────────────────── Pipeline ───────────────────────────────┐
   │ probe → SafetyGate(NSFW) → decode → [per frame]                        │
   │   detect → track/associate → identity match → smooth landmarks         │
   │   → align(affine) → swap → (enhance) → color match                     │
   │   → mask(feather ∧ occlusion) → paste back → (watermark)               │
   │ → encode(main) + encode(matte) → mux(audio, timecode, metadata)        │
   └────────────────────────────────────────────────────────────────────────┘
          ▲ plugins: Detector / Recognizer / Swapper / Enhancer / Occluder / NSFWClassifier
          ▲ runtime: onnxruntime (CoreML → CUDA → CPU を自動選択)
          ▲ io: ffmpeg / ffprobe（サブプロセス、パイプ経由）
```

### 2.1 フレーム処理の詳細

1. **検出**：SCRFD（InsightFace）で bbox と 5 点ランドマーク。動画では N フレームごとにフル検出し、間は前フレームの bbox 周辺だけを再検出して高速化
2. **トラッキング**：IoU と ArcFace 埋め込みのコサイン類似度を組み合わせて検出結果をトラックに紐付ける（簡易 SORT 方式）。一時的な見失い（横顔、遮蔽）は数フレームまで保持
3. **人物指定**：参照画像の埋め込みと各トラックの平均埋め込みを比較し、閾値以上のトラックだけを置換対象にする。判定はフレーム単位ではなくトラック単位で行うので、途中で別人に切り替わることがない
4. **スムージング**：ランドマークに One Euro Filter をかける（静止時は強く、速い動きでは弱くなる適応型）。アフィン行列とマスクにも軽い時間平滑化をかける
5. **スワップ**：アラインした顔（128px、モデル依存）をスワッパーに渡す
6. **補正（任意）**：GFPGAN などで解像感を回復。元の顔とのブレンド率を調整可能
7. **色合わせ**：LAB 空間で平均と分散を合わせる（Reinhard 法）。マスク内部の統計だけを使う
8. **マスク**：顔形状マスク（ランドマーク由来の凸包＋フェザー）と、オクルージョンマスク（手・髪・小道具）の積をとる
9. **貼り戻し**：逆アフィン変換で元解像度に戻し、マスクでアルファ合成。同じマスクをマット出力に使う

### 2.2 ビデオ I/O（M2 / M5）

- `ffprobe` で fps（`24000/1001` のような有理数のまま保持）、解像度、SAR、色情報（primaries / trc / matrix / range）、タイムコード（`tmcd` トラックまたは `timecode` タグ）、音声ストリームを取得
- デコード：`ffmpeg -f rawvideo -pix_fmt rgb48le` をパイプで読む（8bit 素材は rgb24）
- エンコード：raw フレームをパイプで ffmpeg に渡し、元素材の音声を `-map 1:a -c:a copy` で直接 mux（再エンコードしない。ProRes の場合は必要に応じて PCM）
- タイムコード：`-timecode <開始TC>` を付け、fps と合わせる。色タグは `-color_primaries` などで元素材の値を引き継ぐ
- VFR 素材は検知して警告し、CFR に変換するか確認する
- 書き出しプリセット

| プリセット | エンコーダ | 備考 |
|---|---|---|
| ProRes 422 HQ | `prores_videotoolbox`（Mac）→ なければ `prores_ks -profile:v 3` | 10bit `yuv422p10le`、`.mov` |
| H.264 | `h264_videotoolbox` / `h264_nvenc` / `libx264` | 8bit `yuv420p`、`.mp4` または `.mov` |
| マット | ProRes 4444（`yuva444p10le`）でアルファ付き、または ProRes 422 HQ のルミナンスマット（白＝差し替え領域） | 本編と同じ fps、解像度、タイムコード。Resolve ではタイムコードで同期して Fusion / Color の外部マットとして使う |

### 2.3 同意・安全まわり

- **同意ログ**：ソース顔の登録時に `person_name`、`consent_date`、`consent_document`（存在確認と SHA-256 を記録）、`source_type`（`consented_person` / `self` / `synthetic`）を必須にする。`projects/<name>/consent_log.json` に追記専用で保存し、各エントリに前エントリのハッシュを含めて改ざんを検知できるようにする。レンダリング時には、使ったソース顔、入力ファイルのハッシュ、出力ファイルのパスもログに記録する
- **出力メタデータ**：`-metadata comment="AI face-swapped"`、`description`、独自キー（`com.faceswapjp.ai_modified=true`、使用モデル、同意ログ ID）を書き込む。MOV/MP4 とも `-movflags use_metadata_tags` で保存。静止画は PNG の tEXt チャンクまたは EXIF / XMP に書き込む
  - 将来の拡張案：C2PA（Content Credentials）署名。メタデータは再エンコードで消えやすいため、改ざんに強い来歴情報として検討の価値がある
- **透かし（任意）**：位置、不透明度、テキストを設定可能。マット出力には入れない
- **NSFW フィルタ**：ソース画像は全枚、ターゲット動画は 1fps でのサンプリングとシーンチェンジ時のフレームで判定。閾値を超えたら処理を中止し、該当フレーム番号を表示する。判定はスワップ前の入力に対して行う

## 3. ディレクトリ構成

```
faceswapjp/
├── pyproject.toml              # 依存関係。extras: [mac] [cuda] [ui] [enhance]
├── README.md
├── docs/
│   ├── DESIGN.md               # 本書
│   └── MODEL_LICENSES.md       # モデルライセンス一覧
├── models/                     # .gitignore 対象。モデル実体を置く
│   └── manifest.toml           # （リポジトリ管理）URL / sha256 / ライセンス / 商用可否
├── src/faceswapjp/
│   ├── __init__.py
│   ├── config.py               # 設定（パス、閾値、デフォルト）
│   ├── runtime.py              # onnxruntime の EP 選択（CoreML / CUDA / CPU）
│   ├── models/
│   │   ├── registry.py         # manifest 読み込み、取得、ハッシュ検証、ライセンス確認
│   │   └── interfaces.py       # Detector / Recognizer / Swapper / Enhancer / Occluder / NSFW の Protocol
│   ├── analysis/
│   │   ├── face.py             # Face データクラス（bbox, kps, embedding, track_id）
│   │   └── insightface_backend.py
│   ├── swappers/
│   │   ├── __init__.py         # 名前 → 実装の登録（entry points でも拡張可能）
│   │   └── inswapper.py
│   ├── enhancers/
│   │   └── gfpgan.py
│   ├── occluders/
│   │   └── ...                 # 顔パーシング / セグメンテーション
│   ├── compositing/
│   │   ├── align.py            # アフィン推定、warp、逆変換
│   │   ├── mask.py             # 顔マスク、フェザー、マスク合成
│   │   ├── color.py            # 色合わせ
│   │   └── blend.py            # 貼り戻し、マット生成
│   ├── tracking/
│   │   ├── tracker.py          # トラック紐付け
│   │   └── filters.py          # One Euro Filter
│   ├── pipeline/
│   │   ├── frame.py            # 1 フレーム分の処理（静止画、動画、プレビューで共通）
│   │   ├── image.py
│   │   └── video.py
│   ├── io/
│   │   ├── probe.py            # ffprobe ラッパ
│   │   ├── reader.py           # デコードパイプ
│   │   ├── writer.py           # エンコードパイプ、mux
│   │   └── presets.py          # ProRes / H.264 / マット
│   ├── safety/
│   │   ├── consent.py          # 同意ログ（ハッシュチェーン付き）
│   │   ├── nsfw.py
│   │   ├── provenance.py       # メタデータ書き込み
│   │   └── watermark.py
│   ├── project.py              # プロジェクトフォルダの作成と読み込み
│   ├── identity.py             # ソース顔の登録（複数画像の埋め込みを平均して正規化）
│   ├── batch.py
│   ├── cli.py                  # `faceswapjp ...`
│   └── ui/
│       └── app.py              # Gradio
└── tests/
    ├── fixtures/               # 合成テスト素材（ffmpeg の testsrc や AI 生成の顔）
    └── test_*.py
```

プロジェクトのデータ（作業フォルダ。リポジトリとは別の場所に置く）：

```
<project>/
├── project.json
├── consent_log.json
├── identities/<id>/{identity.json, embedding.npy, refs/*.jpg}
├── previews/
└── renders/
```

## 4. CLI（案）

```
faceswapjp project init ./myproj
faceswapjp identity add  --project ./myproj --name "Actor A" \
    --images a1.jpg a2.jpg --consent-date 2026-10-01 --consent-doc ./consent/actorA.pdf \
    --source-type consented_person
faceswapjp swap image    --project ./myproj --identity actor-a --target shot.jpg --out out.png
faceswapjp swap video    --project ./myproj --identity actor-a --target clip.mov \
    --only-person ref_stunt.jpg --format prores422hq --matte --watermark
faceswapjp preview       --project ./myproj --identity actor-a --target clip.mov --frame 1200
faceswapjp batch         --project ./myproj --identity actor-a --input-dir ./clips --format h264
faceswapjp models list   # 各モデルのライセンスと商用可否を表示
```

## 5. マイルストーン（受け入れ基準）

| M | 内容 | 完了条件 |
|---|---|---|
| M1 | 静止画 CLI。`runtime`、`models/registry`、`analysis`、`swappers/inswapper`、`compositing`（フェザーと色合わせ）、`identity`（同意情報は必須入力。ログの完全版は M4） | AI 生成の顔 2 枚でスワップ画像が出力される。CoreML と CPU の両方で実行できる。単体テストが通る |
| M2 | 動画対応。`io/*`、フレームごとのパイプライン、音声の直接 mux、タイムコードと色タグの引き継ぎ | 23.976 / 29.97 / 59.94 の素材で、出力の fps、フレーム数、TC、音声長が元素材と一致する（ffprobe で自動テスト） |
| M3 | トラッキング、人物指定、One Euro Filter、オクルージョンマスク | 2 人が映るテスト素材で指定した人物だけが置換される。ジッター指標（ランドマーク位置の 2 次差分）が平滑化なしより下がる |
| M4 | Gradio UI、同意ログ（ハッシュチェーン）、メタデータ、透かし、NSFW ゲート | UI で登録 → プレビュー → 書き出しまでできる。ffprobe でタグが確認できる。NSFW のテスト画像で処理が止まる |
| M5 | マット書き出し、ProRes 422 HQ / 4444、バッチ処理 | Resolve でマットを読み込み、TC で同期できる。フォルダ内の複数クリップが連続処理され、失敗したクリップはスキップしてレポートに記録される |

## 6. 提案（依頼の構成からの変更点・追加点）

1. **商用案件では InsightFace の商用ライセンスが事実上必要**（詳細は `MODEL_LICENSES.md`）。InsightFace の配布モデル（検出、ArcFace、inswapper）はすべて非商用研究用途に限られる。さらに、オープンな高品質スワッパーのほとんどが ArcFace の埋め込みを入力にしているため、スワッパーだけ差し替えても問題は解決しない。そこで以下のようにする
   - 開発と評価は現行モデルで進める
   - `manifest.toml` に `commercial_use` フラグを持たせ、プロジェクト設定で `commercial: true` にした場合、商用不可のモデルが選ばれていたら処理を止める（ライセンス取得済みのモデルは `license_override` に契約 ID を記録して解除する）
   - 検出は YuNet（MIT）に差し替え可能な設計にしておく
2. **inswapper_128 は出力が 128px** なので、顔が大きく映る 4K のクローズアップでは解像度が足りない。M1〜M3 は inswapper で進めつつ、高解像度スワッパー（商用ライセンスのあるもの）を後から足せる構造にする。補正（GFPGAN）での底上げはあくまで補助
3. **オクルージョン**：手・髪の分離には顔パーシング（BiSeNet / CelebAMask-HQ 学習。非商用）が定番。商用でも使える候補は MediaPipe Selfie Multiclass（髪、体の肌、顔の肌、服を分離）と SAM 2（Apache-2.0。ポイント指定で手や小道具を抽出）。Occluder プラグインとして複数用意する
4. **16bit パイプライン**：ProRes の素材を 8bit に落とすと、顔以外の領域までバンディングが出る。rgb48 で読み書きし、変更は顔領域だけにする
5. **マット出力は ProRes 4444（アルファ付き）とルミナンスマットを選べるようにする**。Resolve ではどちらも扱えるが、ルミナンスマットのほうが他のツールとの互換性が高い
6. **C2PA 署名**（将来の拡張）：「AI face-swapped」タグを、改ざんに強い形でも残せる

## 7. 動作確認できない点（このクラウド環境の制約）

この開発環境は Linux（CPU のみ）なので、CoreML（Apple Silicon）と CUDA の実機検証はできない。実行プロバイダの選択とフォールバックのロジックは、CPU 上でのモックテストで確認する。Mac での実測はユーザー側でお願いしたい。
