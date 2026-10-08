"""Local Gradio UI. Bound to 127.0.0.1, no share links, analytics off.

All buttons call faceswapjp.jobs, i.e. exactly the same code path as the CLI.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import threading  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ..compositing.blend import from_float, to_float  # noqa: E402
from ..identity import SOURCE_TYPES, ConsentError, list_identities, validate_consent  # noqa: E402
from ..jobs import SwapJob, add_identity, batch, is_video, preview, swap_still, swap_video  # noqa: E402
from ..media.presets import PRESETS  # noqa: E402
from ..media.probe import probe  # noqa: E402
from ..pipeline.frame import FrameOptions  # noqa: E402
from ..pipeline.video import RenderSettings, TrackingOptions  # noqa: E402
from ..project import PROJECT_FILE, Project  # noqa: E402
from ..safety.watermark import POSITIONS, Watermark  # noqa: E402

SOURCE_TYPE_LABELS = {
    "consented_person": "同意を得た出演者",
    "self": "自分自身",
    "synthetic": "AI 生成の架空の顔",
}
SELECT_LABELS = {"all": "全員", "largest": "一番大きい顔", "reference": "参照画像の人物だけ"}
MASK_LABELS = {"occlusion": "オクルージョン（手・小道具）", "region": "顔領域のみ（髪・耳を除外）"}
MATTE_LABELS = {"なし": None, "ルミナンスマット（ProRes 422 HQ）": "luma", "フィル＋アルファ（ProRes 4444）": "alpha"}


class UIState:
    def __init__(self, root: Path, device: str):
        self.root = Path(root)
        self.device = device
        self.cancel = threading.Event()

    def projects(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / PROJECT_FILE).exists())

    def project(self, name: str | None) -> Project:
        if not name:
            raise ValueError("プロジェクトを選択してください")
        return Project.load(self.root / name)


def _display(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """uint8/uint16 BGR -> uint8 RGB, downscaled for the browser."""
    img8 = img[..., :3] if img.dtype == np.uint8 else from_float(to_float(img[..., :3]), np.uint8)
    h, w = img8.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img8 = cv2.resize(img8, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img8[..., ::-1])


def _path(file) -> Path | None:
    if file is None:
        return None
    if isinstance(file, (str, Path)):
        return Path(file)
    return Path(getattr(file, "name", file))


def _target(upload, typed: str) -> Path:
    p = Path(typed.strip()).expanduser() if typed and typed.strip() else _path(upload)
    if p is None or not p.is_file():
        raise ValueError("ターゲットのファイルを指定してください")
    return p


def build_job(identity, select, only_person, ref_threshold, masks, enhance, enhance_blend, mask_blur, color_strength,
              smoothing, wm_on, wm_text, wm_pos, device, fmt="h264", matte_label="なし", start=0, end=None) -> SwapJob:
    if not identity:
        raise ValueError("ソース顔を選択してください")
    select_key = {v: k for k, v in SELECT_LABELS.items()}[select]
    refs = [_path(f) for f in (only_person or [])]
    if select_key == "reference" and not refs:
        raise ValueError("「参照画像の人物だけ」を選んだ場合は参照画像を追加してください")
    mask_keys = [k for k, v in MASK_LABELS.items() if v in (masks or [])]
    return SwapJob(
        identity_id=identity,
        frame=FrameOptions(faces="largest" if select_key == "largest" else "all", mask_blur=mask_blur,
                           color_strength=color_strength, enhance_blend=enhance_blend),
        tracking=TrackingOptions(select=select_key, reference_threshold=ref_threshold, smoothing=smoothing),
        render=RenderSettings(format=fmt, matte=MATTE_LABELS[matte_label], start=int(start or 0),
                              end=int(end) if end not in (None, "", 0) else None,
                              watermark=Watermark(wm_text or "AI face-swapped", wm_pos) if wm_on else None),
        reference_images=refs if select_key == "reference" else [],
        masks=",".join(["box", *mask_keys]),
        enhancer="gfpgan" if enhance else None,
        device=device,
    )


def build_ui(state: UIState):
    import gradio as gr

    def identity_choices(project_name):
        try:
            return [(f"{i.label}（{i.consent.person_name}）", i.id) for i in list_identities(state.project(project_name))]
        except (ValueError, FileNotFoundError):
            return []

    def identity_table(project_name):
        try:
            rows = list_identities(state.project(project_name))
        except (ValueError, FileNotFoundError):
            return []
        return [[i.id, i.label, i.consent.person_name, i.consent.consent_date,
                 SOURCE_TYPE_LABELS[i.consent.source_type], Path(i.consent.consent_document).name] for i in rows]

    def on_create_project(name, commercial):
        name = (name or "").strip()
        if not name or "/" in name or name.startswith("."):
            raise gr.Error("プロジェクト名を入力してください（/ は使えません）")
        try:
            Project.init(state.root / name, name, commercial)
        except FileExistsError:
            raise gr.Error(f"{name} は既に存在します") from None
        return gr.update(choices=state.projects(), value=name), f"プロジェクト「{name}」を作成しました"

    def on_project_change(name):
        return (gr.update(choices=identity_choices(name), value=None), identity_table(name))

    def on_register(project_name, label, images, person_name, consent_date, consent_doc, source_type_label):
        project = state.project(project_name)
        imgs = [_path(f) for f in (images or [])]
        if not imgs:
            raise gr.Error("参照画像を 1 枚以上追加してください")
        doc = _path(consent_doc)
        if doc is None:
            raise gr.Error("同意書ファイルは必須です")
        # keep a copy of the consent document inside the project
        digest = hashlib.sha256(doc.read_bytes()).hexdigest()[:16]
        kept = project.root / "consent" / f"{digest}{doc.suffix.lower()}"
        kept.parent.mkdir(exist_ok=True)
        if not kept.exists():
            shutil.copy2(doc, kept)
        source_type = {v: k for k, v in SOURCE_TYPE_LABELS.items()}[source_type_label]
        try:
            consent = validate_consent(person_name, consent_date, kept, source_type)
            ident = add_identity(project, label or person_name, imgs, consent, state.device)
        except ConsentError as exc:
            raise gr.Error(f"同意情報の不備: {exc}") from None
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(str(exc)) from None
        msg = f"登録しました: {ident.id}（参照画像の一致度 {', '.join(f'{s:.2f}' for s in ident.consistency)}）"
        return msg, gr.update(choices=identity_choices(project_name), value=ident.id), identity_table(project_name)

    def on_target_change(upload, typed):
        try:
            p = _target(upload, typed)
        except ValueError:
            return gr.update(maximum=1, value=0, visible=False), ""
        if is_video(p):
            info = probe(p)
            tc = f"TC {info.timecode}" if info.timecode else "TC なし"
            desc = (f"{info.width}x{info.height} / {float(info.fps):.3f} fps ({info.fps}) / {info.nb_frames} フレーム / "
                    f"{info.pix_fmt} / {tc} / 音声: {', '.join(info.audio_codecs) or 'なし'}")
            if info.is_vfr:
                desc += " / ⚠ 可変フレームレート"
            return gr.update(maximum=max(1, info.nb_frames - 1), value=0, visible=True), desc
        return gr.update(maximum=1, value=0, visible=False), "静止画"

    def on_preview(project_name, upload, typed, frame_idx, *opts):
        try:
            job = build_job(*opts, device=state.device)
            original, result = preview(state.project(project_name), job, _target(upload, typed), int(frame_idx or 0))
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(str(exc)) from None
        matte = (np.clip(result.matte, 0, 1) * 255).astype(np.uint8)
        msg = f"{len(result.faces)} 人の顔を置き換えました"
        if len(result.faces) > 1 and job.tracking.select == "all":
            msg += "。⚠ 複数人が映っています。特定の人物だけを置き換える場合は「参照画像の人物だけ」を選んでください"
        return (_display(original), _display(result.frame), _display(cv2.cvtColor(matte, cv2.COLOR_GRAY2BGR)), msg)

    def default_output(project_name, target: Path, fmt: str) -> Path:
        ext = PRESETS[fmt].ext if is_video(target) else ".png"
        return state.project(project_name).renders_dir / f"{target.stem}_swap{ext}"

    def on_render(project_name, upload, typed, out_typed, fmt, matte_label, start, end, *opts, progress=gr.Progress()):
        state.cancel.clear()
        try:
            project = state.project(project_name)
            target = _target(upload, typed)
            job = build_job(*opts, device=state.device, fmt=fmt, matte_label=matte_label, start=start, end=end)
            out = Path(out_typed).expanduser() if out_typed and out_typed.strip() else default_output(project_name, target, fmt)
            if is_video(target):
                res = swap_video(project, job, target, out, cancel=state.cancel,
                                 progress=lambda d, t: progress(d / t, desc=f"{d}/{t} フレーム"))
                lines = [f"書き出し完了: {out}", f"{res.frames} フレーム（置き換えあり {res.frames_with_faces}）"
                         f" / {res.seconds:.0f} 秒"]
                if res.matte_output:
                    lines.append(f"マット: {res.matte_output}")
                lines += [f"⚠ {w}" for w in res.warnings]
            else:
                matte = out.with_name(f"{out.stem}_matte.png") if job.render.matte else None
                res = swap_still(project, job, target, out, matte)
                lines = [f"書き出し完了: {out}", f"{len(res.faces)} 人の顔を置き換えました"]
            return "\n".join(lines)
        except InterruptedError:
            return "中止しました（途中までのファイルは不完全です）"
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(str(exc)) from None

    def on_batch(project_name, in_dir, out_dir, fmt, matte_label, *opts, progress=gr.Progress()):
        state.cancel.clear()
        try:
            project = state.project(project_name)
            job = build_job(*opts, device=state.device, fmt=fmt, matte_label=matte_label)
            if not in_dir or not Path(in_dir).expanduser().is_dir():
                raise ValueError("入力フォルダが見つかりません")
            out = Path(out_dir).expanduser() if out_dir else project.renders_dir / "batch"
            report = batch(project, job, Path(in_dir).expanduser(), out, cancel=state.cancel,
                           progress=lambda i, n, d, t: progress(d / t, desc=f"クリップ {i + 1}/{n}: {d}/{t}"))
        except InterruptedError:
            return "中止しました", []
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(str(exc)) from None
        rows = [[Path(r.source).name, "OK" if r.ok else "失敗", r.output or "", r.error or "; ".join(r.warnings)]
                for r in report]
        return f"{sum(r.ok for r in report)}/{len(report)} 件成功。レポート: {out / 'batch_report.json'}", rows

    def on_log(project_name):
        try:
            log = state.project(project_name).consent_log
        except (ValueError, FileNotFoundError):
            return "", ""
        problems = log.verify()
        status = "✅ ログは改ざんされていません" if not problems else "❌ " + "; ".join(problems)
        return status, json.dumps(log.entries(), ensure_ascii=False, indent=2)

    with gr.Blocks(title="faceswapjp") as demo:
        gr.Markdown(
            "## faceswapjp — ローカル・フェイススワップ\n"
            "本人の同意を得た出演者・自分自身・AI 生成の架空の顔のみを対象とします。"
            "処理はすべてこのマシン上で行われます。既定モデルは**非商用ライセンス**です。"
        )
        with gr.Row():
            project_dd = gr.Dropdown(state.projects(), label="プロジェクト", scale=3)
            new_name = gr.Textbox(label="新規プロジェクト名", scale=2)
            commercial = gr.Checkbox(label="商用（非商用モデルを拒否）", value=False, scale=1)
            create_btn = gr.Button("作成", scale=1)
        project_msg = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("1. ソース顔の登録"):
                with gr.Row():
                    with gr.Column():
                        reg_images = gr.File(label="参照画像（1〜複数枚）", file_count="multiple", file_types=["image"])
                        reg_label = gr.Textbox(label="表示名（例: スタント A）")
                    with gr.Column():
                        reg_person = gr.Textbox(label="本人の氏名 *")
                        reg_date = gr.Textbox(label="同意日 * (YYYY-MM-DD)", value=date.today().isoformat())
                        reg_doc = gr.File(label="同意書ファイル *", file_count="single")
                        reg_type = gr.Radio(list(SOURCE_TYPE_LABELS.values()), label="種類 *",
                                            value=SOURCE_TYPE_LABELS["consented_person"])
                        reg_btn = gr.Button("登録", variant="primary")
                reg_msg = gr.Markdown()
                id_table = gr.Dataframe(headers=["ID", "表示名", "氏名", "同意日", "種類", "同意書"],
                                        interactive=False, label="登録済み")

            with gr.Tab("2. スワップ"):
                with gr.Row():
                    with gr.Column(scale=1):
                        identity_dd = gr.Dropdown([], label="ソース顔")
                        tgt_upload = gr.File(label="ターゲット（動画・静止画）")
                        tgt_path = gr.Textbox(label="またはローカルパス（大きな素材はこちら推奨）")
                        tgt_info = gr.Markdown()
                        with gr.Accordion("置き換え対象", open=True):
                            select = gr.Radio(list(SELECT_LABELS.values()), value=SELECT_LABELS["all"], label="対象")
                            only_person = gr.File(label="参照画像（この人物だけ置き換え）", file_count="multiple",
                                                  file_types=["image"])
                            ref_threshold = gr.Slider(0.2, 0.8, value=0.4, step=0.01, label="人物一致のしきい値")
                        with gr.Accordion("合成", open=False):
                            masks = gr.CheckboxGroup(list(MASK_LABELS.values()), label="マスク（矩形マスクは常に適用）")
                            mask_blur = gr.Slider(0.02, 0.3, value=0.12, step=0.01, label="マスクのぼかし")
                            color_strength = gr.Slider(0, 1, value=0.5, step=0.05, label="色合わせの強さ")
                            smoothing = gr.Slider(0, 1, value=0.5, step=0.05, label="ランドマークのスムージング（動画）")
                            enhance = gr.Checkbox(label="顔補正（GFPGAN）", value=False)
                            enhance_blend = gr.Slider(0, 1, value=0.8, step=0.05, label="顔補正のブレンド")
                        with gr.Accordion("透かし", open=False):
                            wm_on = gr.Checkbox(label="目に見える透かしを入れる", value=False)
                            wm_text = gr.Textbox(value="AI face-swapped", label="テキスト（ASCII）")
                            wm_pos = gr.Dropdown(list(POSITIONS), value="bottom-right", label="位置")
                    with gr.Column(scale=2):
                        frame_slider = gr.Slider(0, 1, value=0, step=1, label="プレビューするフレーム", visible=False)
                        preview_btn = gr.Button("プレビュー", variant="secondary")
                        with gr.Row():
                            prev_orig = gr.Image(label="元", interactive=False)
                            prev_out = gr.Image(label="結果", interactive=False)
                        prev_matte = gr.Image(label="マット", interactive=False, height=200)
                        prev_msg = gr.Markdown()
                        with gr.Group():
                            with gr.Row():
                                fmt = gr.Dropdown(list(PRESETS), value="h264", label="形式")
                                matte = gr.Dropdown(list(MATTE_LABELS), value="なし", label="マット書き出し")
                            with gr.Row():
                                start = gr.Number(value=0, precision=0, label="開始フレーム")
                                end = gr.Number(value=None, precision=0, label="終了フレーム（0 で最後まで）")
                            out_path = gr.Textbox(label="出力パス（空欄でプロジェクトの renders/）")
                            with gr.Row():
                                render_btn = gr.Button("書き出し", variant="primary")
                                cancel_btn = gr.Button("中止", variant="stop")
                        render_msg = gr.Markdown()

            with gr.Tab("3. バッチ"):
                b_in = gr.Textbox(label="入力フォルダ")
                b_out = gr.Textbox(label="出力フォルダ（空欄でプロジェクトの renders/batch）")
                with gr.Row():
                    b_fmt = gr.Dropdown(list(PRESETS), value="prores422hq", label="形式")
                    b_matte = gr.Dropdown(list(MATTE_LABELS), value="なし", label="マット書き出し")
                gr.Markdown("ソース顔と各種設定は「2. スワップ」タブの値を使います。")
                with gr.Row():
                    b_btn = gr.Button("バッチ実行", variant="primary")
                    b_cancel = gr.Button("中止", variant="stop")
                b_msg = gr.Markdown()
                b_table = gr.Dataframe(headers=["ファイル", "結果", "出力", "備考"], interactive=False)

            with gr.Tab("4. 同意ログ"):
                log_btn = gr.Button("読み込み・検証")
                log_status = gr.Markdown()
                log_view = gr.Code(language="json", label="consent_log.json")

        opts = [identity_dd, select, only_person, ref_threshold, masks, enhance, enhance_blend, mask_blur,
                color_strength, smoothing, wm_on, wm_text, wm_pos]
        create_btn.click(on_create_project, [new_name, commercial], [project_dd, project_msg])
        project_dd.change(on_project_change, [project_dd], [identity_dd, id_table])
        reg_btn.click(on_register, [project_dd, reg_label, reg_images, reg_person, reg_date, reg_doc, reg_type],
                      [reg_msg, identity_dd, id_table])
        tgt_upload.change(on_target_change, [tgt_upload, tgt_path], [frame_slider, tgt_info])
        tgt_path.submit(on_target_change, [tgt_upload, tgt_path], [frame_slider, tgt_info])
        preview_btn.click(on_preview, [project_dd, tgt_upload, tgt_path, frame_slider, *opts],
                          [prev_orig, prev_out, prev_matte, prev_msg])
        render_btn.click(on_render, [project_dd, tgt_upload, tgt_path, out_path, fmt, matte, start, end, *opts],
                         [render_msg])
        cancel_btn.click(lambda: state.cancel.set(), None, None)
        b_btn.click(on_batch, [project_dd, b_in, b_out, b_fmt, b_matte, *opts], [b_msg, b_table])
        b_cancel.click(lambda: state.cancel.set(), None, None)
        log_btn.click(on_log, [project_dd], [log_status, log_view])
    return demo


def launch(project_root: Path, port: int = 7860, device: str = "auto") -> None:
    project_root = Path(project_root)
    project_root.mkdir(parents=True, exist_ok=True)
    demo = build_ui(UIState(project_root, device))
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=port, share=False, inbrowser=False, show_error=True)
