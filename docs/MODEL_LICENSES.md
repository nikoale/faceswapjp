# モデル・依存ライブラリのライセンス一覧

調査日：2026-10-08。法的助言ではない。商用案件で使う前に、各配布元の最新の条文を確認し、必要なら法務に相談すること。

凡例：◎ 商用可 / △ 条件付き・グレー / ✕ 商用不可（別途ライセンスが必要）

## 1. 依頼の構成で使うモデル

| 用途 | モデル | コードのライセンス | 重みのライセンス | 商用 | 備考 |
|---|---|---|---|---|---|
| 顔検出・ランドマーク・特徴量 | InsightFace `buffalo_l`（SCRFD + ArcFace R50 など） | MIT | **非商用研究用途のみ** | ✕ | README に「pretrained models … are for non-commercial research only」とあり、自動ダウンロードした場合も同じ。商用ライセンスは recognition-oss-pack@insightface.ai に問い合わせる（法人メールアドレスのみ受付） |
| 同上 | InsightFace `antelopev2` | MIT | 非商用研究用途のみ | ✕ | buffalo_l と同じ扱い |
| スワップ | `inswapper_128.onnx`（InsightFace） | MIT（サンプルコード） | **非商用**（モデル単体の明示的な条文はなく、InsightFace モデル全般の非商用条項が適用される扱い） | ✕ | 公式にはメンテナンスが終了しており、同社の商用製品 Picsi.Ai への移行が案内されている。商用ライセンスは contact@insightface.ai に問い合わせる。用途の審査があり、悪用目的には許諾しないと明記されている |
| 顔補正 | GFPGAN v1.3 / v1.4 | Apache-2.0 | △ | △ | コードは Apache-2.0 だが、学習データ FFHQ は CC BY-NC-SA 4.0。NVIDIA StyleGAN2 由来部分（研究・評価目的のみ）と DFDNet（CC BY-NC-SA 4.0）の条項も同梱されている。Clean 版（v1.3/1.4）は stylegan2-pytorch（MIT）を元にしているが、重みが NC データ由来である点はグレー |
| 顔補正（代替） | CodeFormer | S-Lab License 1.0 | 同左 | ✕ | 非商用のみ。商用は作者への個別許諾が必要 |

**結論**：依頼の推奨構成（InsightFace + inswapper_128 + GFPGAN）は、**そのままでは商用の映像制作には使えない**。社内検証・自主制作・研究用途に限られる。商用案件で使う場合は、InsightFace から buffalo_l と inswapper の商用ライセンスを取得するのが最も現実的。

## 2. 代替・追加候補

| 用途 | モデル | ライセンス | 商用 | 備考 |
|---|---|---|---|---|
| 顔検出 | YuNet（OpenCV Zoo） | MIT（モデル・コードとも） | ◎ | 5 点ランドマーク付き。SCRFD の代替になる |
| 顔認識（人物照合） | SFace（OpenCV Zoo） | Apache-2.0 | ◎ | 人物指定（トラックの照合）には使える。ただしスワッパーが要求する ArcFace 埋め込みの代わりにはならない |
| スワップ | GHOST（ai-forever） | コードは Apache-2.0 | △ | 重みのライセンスは明記なし。ID エンコーダに ArcFace（InsightFace 系）を使っている |
| スワップ | SimSwap | CC BY-NC 4.0 | ✕ | ArcFace 依存 |
| スワップ | FaceFusion HyperSwap 系 | 未確認 | ？ | FaceFusion 本体のコードとは別に、重みのライセンスを確認できなかった。採用するならモデル配布ページで条文を確認する |
| オクルージョン | BiSeNet 顔パーシング（face-parsing.PyTorch） | コードは MIT | ✕ | 重みは CelebAMask-HQ で学習されており、このデータセットは非商用研究用途のみで再配布も禁止 |
| オクルージョン | SAM 2（Meta） | Apache-2.0（チェックポイント含む） | ◎ | プロンプト（点・矩形）で手や小道具を抽出できる。やや重い |
| オクルージョン | MediaPipe Selfie Multiclass | MediaPipe 本体は Apache-2.0。このモデル固有の条文は未確認 | △ | 髪、体の肌、顔の肌、服を分離できる。採用前にモデルカードで確認する |
| NSFW | OpenNSFW2（Yahoo open_nsfw の移植） | コードは MIT、元モデルは BSD-2-Clause（Yahoo） | ◎ | 古いモデルで精度はそこそこ。閾値は保守的に設定する |
| NSFW | Falconsai/nsfw_image_detection（ViT） | Apache-2.0（Hugging Face 上の表記。この環境から Hugging Face にアクセスできず、直接は確認できていない） | △→◎ | 採用前にモデルカードを確認する |
| 超解像（任意） | Real-ESRGAN | BSD-3-Clause | ◎ | 背景や全体のアップスケール用。顔専用ではない |

## 3. ランタイム・ツール

| ソフトウェア | ライセンス | 備考 |
|---|---|---|
| onnxruntime | MIT | CoreML / CUDA の実行プロバイダを含む |
| InsightFace Python パッケージ | MIT | コードのみ。モデルは上記の条件に従う |
| Gradio | Apache-2.0 | |
| OpenCV (opencv-python) | Apache-2.0 | |
| FFmpeg | LGPL-2.1+（`--enable-gpl` でビルドした場合は GPL） | libx264 は GPL。ツールからは外部プロセスとして呼ぶのでリンクの問題は生じない。社内で使う分には影響しない。ツール自体を配布する場合は、同梱の扱いに注意する |
| ProRes エンコード | `prores_ks`（FFmpeg）/ `prores_videotoolbox`（Apple） | `prores_ks` は Apple 公認のエンコーダではない。納品先が Apple 認定エンコーダを要求する場合は、Mac で `prores_videotoolbox` を使う |

## 4. 実装で使っているモデル（`faceswapjp models list` と同じ内容）

InsightFace 以外のモデルは、FaceFusion プロジェクトが GitHub Releases で配布している ONNX 変換版（facefusion/facefusion-assets）を取得し、SHA-256 を manifest に記録している。変換元のライセンスがそのまま適用される。

| 名前 | 用途 | 入手元 | 商用 |
|---|---|---|---|
| buffalo_l | 検出・特徴量 | deepinsight/insightface v0.7 release | ✕ |
| inswapper_128 | スワップ | facefusion-assets models-3.0.0 | ✕ |
| xseg_1 | オクルージョン（手・小道具） | facefusion-assets models-3.1.0 | ✕ 扱い（重みのライセンス表記なし。DeepFaceLab 系） |
| bisenet_resnet_34 | 顔領域マスク | facefusion-assets models-3.0.0 | ✕（CelebAMask-HQ） |
| gfpgan_1_4 | 顔補正（任意） | facefusion-assets models-3.0.0 | ✕ 扱い（FFHQ 由来） |
| open_nsfw | NSFW ゲート | facefusion-assets models-3.0.0 | ◎（Yahoo、BSD-2-Clause） |

## 5. 実装上の対応

- `src/faceswapjp/models/manifest.toml` に各モデルの `license`、`commercial_use`（true / false / unknown）、`source_url`、`sha256` を記録する
- `faceswapjp models list` でライセンスの状況を一覧表示する
- プロジェクトに `commercial = true` を設定した場合、`commercial_use != true` のモデルが選ばれていると処理を止める。商用ライセンスを取得したら `license_override = "<契約番号など>"` を記録して解除する

## 出典

- InsightFace python-package README（モデルは非商用研究用途のみ）：https://github.com/deepinsight/insightface/tree/master/python-package
- InsightFace README（ライセンスの問い合わせ先）：https://raw.githubusercontent.com/deepinsight/insightface/HEAD/README.md
- inswapper の案内（メンテナンス終了、Picsi.Ai への移行）：https://github.com/deepinsight/insightface/tree/master/examples/in_swapper
- InsightFace 商用ライセンス：https://www.insightface.ai/services/models-commercial-licensing ／ https://www.insightface.ai/solutions/face-swapping
- GFPGAN LICENSE：https://github.com/TencentARC/GFPGAN/blob/master/LICENSE
- CodeFormer LICENSE（S-Lab 1.0）：https://github.com/sczhou/CodeFormer/blob/master/LICENSE
- YuNet：https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- SFace：https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
- GHOST：https://github.com/ai-forever/ghost
- SAM 2：https://github.com/facebookresearch/sam2
- face-parsing.PyTorch：https://github.com/zllrunning/face-parsing.PyTorch
- CelebAMask-HQ：https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html
- MediaPipe Image Segmenter：https://ai.google.dev/mediapipe/solutions/vision/image_segmenter
- OpenNSFW2：https://github.com/bhky/opennsfw2
