"""Local Gradio UI, built as a step-by-step guide for people who have never used a face swap tool.

Bound to 127.0.0.1, no share links, analytics off. Every action calls faceswapjp.jobs,
i.e. the same code path (license check, NSFW gate, metadata, consent log) as the CLI.
"""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import hashlib  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ..compositing.blend import from_float, to_float  # noqa: E402
from ..identity import ConsentError, list_identities, validate_consent  # noqa: E402
from ..jobs import SwapJob, add_identity, batch, find_faces, is_video, preview, swap_still, swap_video  # noqa: E402
from ..media.probe import FFmpegError, probe  # noqa: E402
from ..models import registry  # noqa: E402
from ..pipeline.frame import FrameOptions  # noqa: E402
from ..pipeline.video import RenderSettings, TrackingOptions  # noqa: E402
from ..project import PROJECT_FILE, Project  # noqa: E402
from ..safety.nsfw import NSFWContentError  # noqa: E402
from ..safety.watermark import Watermark  # noqa: E402

DEFAULT_PROJECT = "はじめての作品"

SOURCE_TYPES = {
    "self": "自分自身の顔",
    "consented_person": "出演者の顔（本人の同意あり）",
    "synthetic": "AI で作った架空の顔",
}
WHO_PICK = "映っている人の中から選ぶ（おすすめ）"
WHO_ALL = "映っている全員を置き換える"
USE_SHARE = "確認・共有用（MP4・ファイルが小さい）"
USE_EDIT = "編集用の高画質（ProRes 422 HQ・DaVinci Resolve / Premiere 向け）"
MATTE_NONE = "書き出さない"
MATTE_LUMA = "白黒のマスク動画（おすすめ）"
MATTE_ALPHA = "透明付きの差し替え素材（ProRes 4444）"

PICK_MATCH = 0.6  # two detections with this similarity are treated as the same person in the picker

CSS = """
.big-button button, button.big-button { font-size: 1.15rem !important; padding: 14px 18px !important; }
.hint { color: var(--body-text-color-subdued); font-size: 0.92rem; }
.summary { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 8px 14px; }
.summary p { margin: 0; }
footer { display: none !important; }
"""


# ----------------------------------------------------------------------------------------------
# helpers


def friendly(exc: Exception) -> str:
    """Turn internal errors into something a beginner can act on."""
    if isinstance(exc, NSFWContentError):
        return "この素材は処理できません。性的な内容が含まれている可能性があるため、処理を止めました。"
    if isinstance(exc, ConsentError):
        return f"同意の情報に不足があります：{exc}"
    if isinstance(exc, registry.ModelError):
        return "必要なファイル（AI モデル）がまだありません。画面上部の「必要なファイルをダウンロード」を押してください。"
    if isinstance(exc, FFmpegError):
        if "not found" in str(exc):
            return "動画を扱うためのソフト「ffmpeg」が見つかりません。ターミナルで `brew install ffmpeg` を実行してください。"
        return f"動画ファイルを読み書きできませんでした。ファイルが壊れていないか確認してください。（{exc}）"
    msg = str(exc)
    if "no target face" in msg or "no face found" in msg:
        return "顔が見つかりませんでした。顔がはっきり写っている画像・場面を選んでください。"
    return msg


def to_display(img: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """uint8/uint16 BGR -> uint8 RGB, downscaled for the browser."""
    img8 = img[..., :3] if img.dtype == np.uint8 else from_float(to_float(img[..., :3]), np.uint8)
    h, w = img8.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img8 = cv2.resize(img8, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img8[..., ::-1])


def file_path(file) -> Path | None:
    if file is None:
        return None
    if isinstance(file, (str, Path)):
        return Path(file)
    return Path(getattr(file, "name", file))


def describe_target(path: Path) -> tuple[str, float, int, np.ndarray | None]:
    """Plain-language description, fps, frame count and a thumbnail."""
    if is_video(path):
        info = probe(path)
        sec = info.nb_frames / float(info.fps) if info.fps else info.duration
        parts = [f"🎬 動画", f"{info.width}×{info.height}", f"{float(info.fps):.3f} fps", f"{sec:.1f} 秒",
                 "音声あり" if info.has_audio else "音声なし"]
        if info.timecode:
            parts.append(f"タイムコード {info.timecode}")
        text = "・".join(parts)
        if info.is_vfr:
            text += "\n\n⚠ この動画はフレームレートが一定ではありません（スマホ撮影に多い）。音ズレする場合は、編集ソフトで一定のフレームレートに変換してから使ってください。"
        return text, float(info.fps), info.nb_frames, None
    from ..imageio import read_image

    img = read_image(path)
    return f"🖼 写真・{img.shape[1]}×{img.shape[0]}", 0.0, 1, to_display(img, 640)


@dataclass
class Picked:
    embedding: np.ndarray
    thumb: np.ndarray


@dataclass
class Session:
    """Per-browser-tab state."""

    project: str | None = None
    identity: str | None = None
    identity_label: str | None = None
    target: str | None = None
    fps: float = 0.0
    frames: int = 1
    who: str = WHO_PICK
    picked: list[Picked] = field(default_factory=list)
    found: list = field(default_factory=list)  # faces detected on the current picker frame
    found_frame: np.ndarray | None = None  # display image (RGB) of the picker frame
    found_scale: float = 1.0
    last_output: str | None = None

    def is_picked(self, emb: np.ndarray) -> int | None:
        for i, p in enumerate(self.picked):
            if float(p.embedding @ emb) >= PICK_MATCH:
                return i
        return None

    def references(self) -> np.ndarray | None:
        if self.who != WHO_PICK or not self.picked:
            return None
        return np.vstack([p.embedding for p in self.picked])


class UIState:
    def __init__(self, root: Path, device: str):
        self.root = Path(root)
        self.device = device
        self.cancel = threading.Event()

    def projects(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / PROJECT_FILE).exists())

    def ensure_default(self) -> str:
        names = self.projects()
        if names:
            return names[0]
        Project.init(self.root / DEFAULT_PROJECT, DEFAULT_PROJECT)
        return DEFAULT_PROJECT

    def project(self, name: str | None) -> Project:
        if not name:
            raise ValueError("作品を選んでください")
        return Project.load(self.root / name)


def missing_models() -> list[str]:
    return [name for name, spec in registry.load_manifest().items() if registry.verify(spec)]


def summary_md(s: Session) -> str:
    face = s.identity_label or s.identity or "未選択"
    target = Path(s.target).name if s.target else "未選択"
    if s.who == WHO_ALL:
        who = "全員"
    else:
        who = f"{len(s.picked)} 人" if s.picked else "未選択"
    mark = lambda ok: "✅" if ok else "⬜"  # noqa: E731
    return (f"{mark(s.identity)} **使う顔**：{face}　　{mark(s.target)} **素材**：{target}　　"
            f"{mark(s.who == WHO_ALL or s.picked)} **置き換える人**：{who}")


def annotate(s: Session) -> np.ndarray | None:
    if s.found_frame is None:
        return None
    img = s.found_frame.copy()
    for i, face in enumerate(s.found):
        x0, y0, x1, y1 = (face.bbox * s.found_scale).astype(int)
        chosen = s.who == WHO_ALL or (face.embedding is not None and s.is_picked(face.embedding) is not None)
        color = (40, 200, 90) if chosen else (255, 255, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 4 if chosen else 2)
        label = f"{i + 1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        ty = y0 if y0 - th - 12 >= 0 else y0 + th + 12  # keep the number visible at the top edge
        cv2.rectangle(img, (x0, ty - th - 12), (x0 + tw + 12, ty), color, -1)
        cv2.putText(img, label, (x0 + 6, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def face_crop(s: Session, i: int) -> np.ndarray:
    h, w = s.found_frame.shape[:2]
    x0, y0, x1, y1 = s.found[i].bbox * s.found_scale
    m = 0.25 * (x1 - x0)
    return s.found_frame[max(0, int(y0 - m)):min(h, int(y1 + m)), max(0, int(x0 - m)):min(w, int(x1 + m))].copy()


def face_choices(s: Session):
    """CheckboxGroup update listing the faces on the current frame, ticked if picked."""
    import gradio as gr

    choices = [f"{i + 1} 番" for i in range(len(s.found))]
    value = [c for i, c in enumerate(choices)
             if s.who == WHO_ALL or (s.found[i].embedding is not None and s.is_picked(s.found[i].embedding) is not None)]
    return gr.update(choices=choices, value=value, visible=bool(choices))


def toggle_face(s: Session, i: int, on: bool | None = None) -> None:
    face = s.found[i]
    if face.embedding is None:
        return
    j = s.is_picked(face.embedding)
    if j is None and on is not False:
        s.picked.append(Picked(face.embedding, face_crop(s, i)))
    elif j is not None and on is not True:
        s.picked.pop(j)


def picked_text(s: Session) -> str:
    if s.who == WHO_ALL:
        return "映っている**全員**を置き換えます。"
    if not s.picked:
        return "まだ選ばれていません。写真の中の顔をクリックするか、番号にチェックを入れてください。"
    return f"**{len(s.picked)} 人**を選択中です（緑の枠）。もう一度クリックすると外せます。"


def reveal(path: str) -> None:
    """Show a file in Finder / the file manager (the UI only runs locally)."""
    p = Path(path)
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(p)], check=False)
    elif sys.platform.startswith("win"):
        subprocess.run(["explorer", "/select,", str(p)], check=False)
    else:
        subprocess.run(["xdg-open", str(p.parent)], check=False)


def build_job(s: Session, keep_front: bool, sharpen: bool, blend: float, color: float, strictness: float,
              smoothing: float, use: str = USE_SHARE, matte: str = MATTE_NONE, watermark: bool = False,
              start_sec: float = 0.0, end_sec: float = 0.0, device: str = "auto") -> SwapJob:
    if not s.identity:
        raise ValueError("「① 使う顔」で顔を選んでください")
    if s.who == WHO_PICK and not s.picked:
        raise ValueError("「③ 置き換える人」で、置き換える人を選んでください")
    fps = s.fps or 1.0
    start = int(round((start_sec or 0) * fps)) if s.fps else 0
    end = int(round(end_sec * fps)) if s.fps and end_sec and end_sec > 0 else None
    return SwapJob(
        identity_id=s.identity,
        frame=FrameOptions(mask_blur=blend, color_strength=color),
        tracking=TrackingOptions(select="reference" if s.who == WHO_PICK else "all",
                                 reference_threshold=strictness, smoothing=smoothing),
        render=RenderSettings(
            format="h264" if use == USE_SHARE else "prores422hq",
            matte={MATTE_NONE: None, MATTE_LUMA: "luma", MATTE_ALPHA: "alpha"}[matte],
            start=start, end=end,
            watermark=Watermark() if watermark else None,
        ),
        reference_embeddings=s.references(),
        masks="box,occlusion,region" if keep_front else "box",
        enhancer="gfpgan" if sharpen else None,
        device=device,
    )


# ----------------------------------------------------------------------------------------------
# UI


def build_ui(state: UIState):
    import gradio as gr

    first_project = state.ensure_default()

    # ---- callbacks ---------------------------------------------------------------------------

    def identity_options(project_name):
        """[(label, id)] for the radio, plus id -> (label, thumbnail path)."""
        try:
            project = state.project(project_name)
            items = list_identities(project)
        except (ValueError, FileNotFoundError):
            return [], {}
        info = {i.id: (f"{i.label}（{i.consent.person_name}）", str(project.identities_dir / i.id / i.reference_images[0]))
                for i in items}
        return [(label, iid) for iid, (label, _) in info.items()], info

    def identity_widgets(project_name, selected=None):
        choices, info = identity_options(project_name)
        thumb = info[selected][1] if selected in info else None
        empty = "まだ顔が登録されていません。下の「新しい顔を登録する」から始めましょう。" if not choices else ""
        return (gr.update(choices=choices, value=selected, visible=bool(choices)),
                gr.update(value=thumb, visible=thumb is not None), empty, gr.update(open=not choices))

    def on_load(s: Session):
        s.project = s.project or first_project
        banner = gr.update(visible=bool(missing_models()))
        return (s, gr.update(choices=state.projects(), value=s.project), *identity_widgets(s.project),
                summary_md(s), banner)

    def on_project(name, s: Session):
        s = Session(project=name)
        return (s, *identity_widgets(name), summary_md(s))

    def on_new_project(name):
        name = (name or "").strip()
        if not name or "/" in name or name.startswith("."):
            raise gr.Error("作品名を入力してください（「/」は使えません）")
        try:
            Project.init(state.root / name, name)
        except FileExistsError:
            raise gr.Error(f"「{name}」はすでにあります") from None
        return gr.update(choices=state.projects(), value=name), ""

    def on_download(progress=gr.Progress()):
        missing = missing_models()
        for i, name in enumerate(missing):
            progress(i / max(1, len(missing)), desc=f"{name} をダウンロード中…（{i + 1}/{len(missing)}）")
            try:
                registry.ensure(name, download=True)
            except Exception as exc:  # noqa: BLE001
                raise gr.Error(f"{name} のダウンロードに失敗しました：{exc}") from None
        return gr.update(visible=False), "✅ 準備ができました"

    def on_pick_identity(iid, s: Session):
        _, info = identity_options(s.project)
        if iid not in info:
            return s, gr.update(visible=False), summary_md(s)
        s.identity, s.identity_label = iid, info[iid][0]
        return s, gr.update(value=info[iid][1], visible=True), summary_md(s)

    def on_register(images, label, kind, person, consent_date, consent_doc, s: Session):
        try:
            project = state.project(s.project)
            imgs = [file_path(f) for f in (images or [])]
            if not imgs:
                raise ValueError("顔の写真を 1 枚以上追加してください")
            doc = file_path(consent_doc)
            if doc is None:
                raise ConsentError("同意書のファイルを追加してください（自分の顔の場合も、同意したことを書いたメモなどを添付）")
            digest = hashlib.sha256(doc.read_bytes()).hexdigest()[:16]
            kept = project.root / "consent" / f"{digest}{doc.suffix.lower()}"
            kept.parent.mkdir(exist_ok=True)
            if not kept.exists():
                shutil.copy2(doc, kept)
            source_type = {v: k for k, v in SOURCE_TYPES.items()}[kind]
            consent = validate_consent(person, consent_date, kept, source_type)
            ident = add_identity(project, (label or "").strip() or person.strip(), imgs, consent, state.device)
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(friendly(exc)) from None
        s.identity, s.identity_label = ident.id, f"{ident.label}（{ident.consent.person_name}）"
        radio, thumb, _, _ = identity_widgets(s.project, ident.id)
        low = [f"{i + 1} 枚目" for i, c in enumerate(ident.consistency) if c < 0.45]
        note = f"\n\n⚠ {', '.join(low)} は他の写真と別人のように見えます。同じ人の写真か確認してください。" if low else ""
        return (s, radio, thumb, f"✅ 「{ident.label}」を登録しました。この顔を使います。{note}", summary_md(s),
                gr.update(open=False), None, "", "", None)

    def on_target(upload, typed, s: Session):
        path = Path(typed.strip()).expanduser() if typed and typed.strip() else file_path(upload)
        if path is None:
            return s, "", None, gr.update(visible=False), gr.update(visible=False), summary_md(s)
        if not path.is_file():
            raise gr.Error("ファイルが見つかりません。パスを確認してください。")
        try:
            text, fps, frames, thumb = describe_target(path)
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(friendly(exc)) from None
        s.target, s.fps, s.frames = str(path), fps, frames
        s.found, s.found_frame = [], None
        if thumb is None and frames > 0:
            from ..media.ffpipe import FrameReader

            thumb = to_display(FrameReader(probe(path), start=0, count=1).read_one(), 640)
        slider = gr.update(visible=frames > 1, maximum=max(1, frames - 1), value=0)
        return s, text, thumb, slider, slider, summary_md(s)

    def search_faces(s: Session, frame_idx):
        if not s.target:
            raise gr.Error("「② 素材」でファイルを選んでください")
        try:
            frame, faces = find_faces(state.project(s.project), Path(s.target), int(frame_idx or 0), state.device)
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(friendly(exc)) from None
        disp = to_display(frame)
        s.found, s.found_frame, s.found_scale = faces, disp, disp.shape[1] / frame.shape[1]
        if len(faces) == 1 and not s.picked and s.who == WHO_PICK:
            toggle_face(s, 0, on=True)
        if not faces:
            msg = "この場面では顔が見つかりませんでした。スライダーで別の場面を選んで、もう一度探してください。"
        elif len(faces) == 1:
            msg = "1 人見つかりました（自動で選択しました）。"
        else:
            msg = f"{len(faces)} 人見つかりました。置き換える人の顔を写真の中でクリックしてください。"
        return s, annotate(s), face_choices(s), msg, picked_text(s), summary_md(s)

    def on_enter_who(s: Session, frame_idx):
        """Moving to step 3 searches faces automatically so the user sees them right away."""
        if not s.target:
            raise gr.Error("ファイルを選んでください")
        if not s.found:
            return search_faces(s, frame_idx)
        return s, annotate(s), face_choices(s), gr.update(), picked_text(s), summary_md(s)

    def on_click_scene(s: Session, evt: gr.SelectData):
        """Click on the scene image: toggle the face under the cursor."""
        x, y = evt.index
        hit = None
        for i, face in enumerate(s.found):
            x0, y0, x1, y1 = face.bbox * s.found_scale
            pad = 0.15 * (x1 - x0)
            if x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad:
                hit = i
                break
        if hit is not None:
            if s.who == WHO_ALL:
                s.who = WHO_PICK
            toggle_face(s, hit)
        return s, annotate(s), face_choices(s), picked_text(s), summary_md(s), gr.update(value=s.who)

    def on_check_faces(checked, s: Session):
        if s.who == WHO_ALL:
            return s, annotate(s), picked_text(s), summary_md(s)
        for i in range(len(s.found)):
            toggle_face(s, i, on=f"{i + 1} 番" in (checked or []))
        return s, annotate(s), picked_text(s), summary_md(s)

    def on_who(who, s: Session):
        s.who = who
        return s, annotate(s), picked_text(s), summary_md(s), face_choices(s)

    def on_clear(s: Session):
        s.picked = []
        return s, annotate(s), face_choices(s), picked_text(s), summary_md(s)

    def on_preview(s: Session, frame_idx, keep_front, sharpen, blend, color, strictness, smoothing):
        try:
            job = build_job(s, keep_front, sharpen, blend, color, strictness, smoothing, device=state.device)
            original, result = preview(state.project(s.project), job, Path(s.target), int(frame_idx or 0))
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(friendly(exc)) from None
        n = len(result.faces)
        if n == 0:
            msg = "この場面には置き換える人が映っていません。スライダーで別の場面を選んでください。"
        else:
            msg = f"✅ {n} 人を置き換えました。画像の上で左右にドラッグすると、元と比べられます。"
        matte = (np.clip(result.matte, 0, 1) * 255).astype(np.uint8)
        return (to_display(original), to_display(result.frame)), msg, to_display(cv2.cvtColor(matte, cv2.COLOR_GRAY2BGR), 640)

    def default_output(s: Session, use: str) -> Path:
        target = Path(s.target)
        ext = (".mp4" if use == USE_SHARE else ".mov") if is_video(target) else ".png"
        return state.project(s.project).renders_dir / f"{target.stem}_swap{ext}"

    def on_use(use, s: Session):
        if not s.target:
            return ""
        return f"保存先：`{default_output(s, use)}`"

    def on_render(s: Session, keep_front, sharpen, blend, color, strictness, smoothing, use, matte, watermark,
                  start_sec, end_sec, progress=gr.Progress()):
        state.cancel.clear()
        try:
            if not s.target:
                raise ValueError("「② 素材」でファイルを選んでください")
            job = build_job(s, keep_front, sharpen, blend, color, strictness, smoothing, use, matte, watermark,
                            start_sec, end_sec, state.device)
            project = state.project(s.project)
            out = default_output(s, use)
            if is_video(Path(s.target)):
                progress(0, desc="安全チェック中…")
                res = swap_video(project, job, Path(s.target), out, cancel=state.cancel,
                                 progress=lambda d, t: progress(d / t, desc=f"処理中… {d}/{t} フレーム"))
                took = f"{res.seconds:.0f} 秒" if res.seconds < 90 else f"{res.seconds / 60:.1f} 分"
                lines = ["### ✅ 書き出しが終わりました", f"`{out}`", f"{res.frames} フレーム・{took}かかりました"]
                if out.suffix == ".mov":
                    lines.append("高画質ファイルはブラウザでは再生できません。DaVinci Resolve などの編集ソフトで開いてください。")
                if res.matte_output:
                    lines.append(f"マスク動画：`{res.matte_output}`")
                lines += [f"⚠ {w}" for w in res.warnings]
                video = str(out) if out.suffix == ".mp4" else None
            else:
                mpath = out.with_name(f"{out.stem}_matte.png") if job.render.matte else None
                swap_still(project, job, Path(s.target), out, mpath)
                lines = ["### ✅ 書き出しが終わりました", f"`{out}`"]
                video = None
            s.last_output = str(out)
            return s, "\n\n".join(lines), gr.update(value=video, visible=video is not None), gr.update(visible=True)
        except InterruptedError:
            return s, "中止しました。途中までのファイルは使えません。", gr.update(visible=False), gr.update(visible=False)
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(friendly(exc)) from None

    def on_reveal(s: Session):
        if s.last_output:
            reveal(s.last_output)

    def on_batch(s: Session, in_dir, keep_front, sharpen, blend, color, strictness, smoothing, use, matte, watermark,
                 progress=gr.Progress()):
        state.cancel.clear()
        try:
            folder = Path((in_dir or "").strip()).expanduser()
            if not folder.is_dir():
                raise ValueError("フォルダが見つかりません。パスを確認してください。")
            job = build_job(s, keep_front, sharpen, blend, color, strictness, smoothing, use, matte, watermark,
                            device=state.device)
            project = state.project(s.project)
            out = project.renders_dir / f"batch_{folder.name}"
            report = batch(project, job, folder, out, cancel=state.cancel,
                           progress=lambda i, n, d, t: progress(d / t, desc=f"{i + 1}/{n} 本目… {d}/{t} フレーム"))
        except InterruptedError:
            return "中止しました。", []
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(friendly(exc)) from None
        rows = [[Path(r.source).name, "✅ 完了" if r.ok else "❌ 失敗", Path(r.output).name if r.output else "",
                 friendly(Exception(r.error)) if r.error else ""] for r in report]
        ok = sum(r.ok for r in report)
        return f"**{ok} / {len(report)} 本** が完了しました。保存先：`{out}`", rows

    def on_log(s: Session):
        try:
            log = state.project(s.project).consent_log
        except (ValueError, FileNotFoundError):
            return "", []
        problems = log.verify()
        status = "✅ 記録は改ざんされていません。" if not problems else "❌ 記録が書き換えられています：" + "; ".join(problems)
        names = {"project_created": "作品を作成", "identity_registered": "顔を登録", "render": "書き出し"}
        rows = []
        for e in reversed(log.entries()):
            d = e["data"]
            detail = d.get("person_name") or d.get("name") or ""
            if e["event"] == "render":
                detail = f"{d.get('person_name', '')} → {Path(d.get('output', '')).name}"
            rows.append([e["timestamp"].replace("T", " ")[:19], names.get(e["event"], e["event"]), detail])
        return status, rows

    # ---- layout ------------------------------------------------------------------------------

    with gr.Blocks(title="faceswapjp") as demo:
        s = gr.State(Session())

        gr.Markdown("# faceswapjp\n顔の差し替えを、5 つのステップで行います。処理はすべてこのパソコンの中だけで行われ、素材が外部に送られることはありません。")
        with gr.Group(visible=False) as banner:
            gr.Markdown("### 🔧 はじめに準備が必要です\n顔の差し替えに使う AI のファイル（約 1.2 GB）をダウンロードします。最初の 1 回だけです。")
            dl_btn = gr.Button("必要なファイルをダウンロード", variant="primary", elem_classes="big-button")
        dl_msg = gr.Markdown()
        with gr.Row(equal_height=True):
            project_dd = gr.Dropdown([], label="作品（フォルダごとに記録と書き出しファイルを分けます）", scale=4)
            with gr.Column(scale=2):
                with gr.Accordion("＋ 新しい作品", open=False):
                    new_project = gr.Textbox(label="作品名", placeholder="例：短編映画_スタントシーン")
                    new_project_btn = gr.Button("作成")
        summary = gr.Markdown(elem_classes="summary")

        with gr.Tabs():
            with gr.Tab("かんたん作成"):
                with gr.Walkthrough(selected=0) as wt:
                    # ① -------------------------------------------------------------------------
                    with gr.Step("使う顔", id=0):
                        gr.Markdown("### どの顔に差し替えますか？\n新しく映したい顔を選んでください。登録した顔はクリックで選べます。")
                        with gr.Row():
                            face_radio = gr.Radio([], label="登録済みの顔（クリックで選択）", scale=3)
                            face_thumb = gr.Image(label="選んだ顔", interactive=False, height=180, visible=False,
                                                  scale=1)
                        face_msg = gr.Markdown()
                        with gr.Accordion("＋ 新しい顔を登録する", open=True) as reg_acc:
                            with gr.Row():
                                with gr.Column():
                                    reg_images = gr.File(label="1. 顔の写真（1〜5 枚）", file_count="multiple",
                                                         file_types=["image"])
                                    gr.Markdown("📷 **きれいに仕上げるコツ**：本人だけが写った、明るく正面に近い写真。"
                                                "角度や表情が違う写真を数枚入れると安定します。", elem_classes="hint")
                                    reg_label = gr.Textbox(label="2. 呼び名（あとで見分けるため）", placeholder="例：スタント A")
                                with gr.Column():
                                    gr.Markdown("**3. 本人の同意（必須）**\n\nこの顔を使ってよいという同意を記録します。"
                                                "記録はこの作品のフォルダにだけ保存されます。", elem_classes="hint")
                                    reg_kind = gr.Radio(list(SOURCE_TYPES.values()), value=SOURCE_TYPES["self"],
                                                        label="誰の顔ですか？")
                                    reg_person = gr.Textbox(label="本人の氏名", placeholder="例：山田 太郎（AI の顔なら作成者名）")
                                    reg_date = gr.Textbox(label="同意した日（年-月-日）", value=date.today().isoformat())
                                    reg_doc = gr.File(label="同意書のファイル（PDF・写真など）", file_count="single")
                            reg_btn = gr.Button("この顔を登録する", variant="primary")
                        next1 = gr.Button("次へ：素材を選ぶ →", variant="primary", elem_classes="big-button")

                    # ② -------------------------------------------------------------------------
                    with gr.Step("素材", id=1):
                        gr.Markdown("### どの映像（または写真）の顔を差し替えますか？")
                        with gr.Row():
                            with gr.Column():
                                tgt_upload = gr.File(label="ここにファイルをドラッグ＆ドロップ",
                                                     file_types=["video", "image", ".mov", ".mxf", ".mts"])
                                with gr.Accordion("大きなファイル（数 GB 以上）の場合", open=False):
                                    gr.Markdown("アップロードせずに、ファイルの場所を貼り付けて Enter を押すと速く読み込めます。"
                                                "（Mac：Finder でファイルを右クリック →「option」キーを押しながら「パス名をコピー」）",
                                                elem_classes="hint")
                                    tgt_path = gr.Textbox(label="ファイルの場所", placeholder="/Users/you/Movies/clip.mov")
                            with gr.Column():
                                tgt_thumb = gr.Image(label="最初の場面", interactive=False, height=260)
                                tgt_info = gr.Markdown()
                        with gr.Row():
                            back2 = gr.Button("← 戻る")
                            next2 = gr.Button("次へ：置き換える人を選ぶ →", variant="primary", elem_classes="big-button")

                    # ③ -------------------------------------------------------------------------
                    with gr.Step("置き換える人", id=2):
                        gr.Markdown("### 誰の顔を置き換えますか？")
                        who = gr.Radio([WHO_PICK, WHO_ALL], value=WHO_PICK, show_label=False)
                        with gr.Group() as pick_group:
                            with gr.Row():
                                pick_slider = gr.Slider(0, 1, value=0, step=1, label="顔を探す場面（スライダーで移動）",
                                                        visible=False, scale=4)
                                pick_btn = gr.Button("この場面で顔を探す", scale=1)
                            pick_msg = gr.Markdown()
                            pick_image = gr.Image(label="顔をクリックして選択（緑の枠＝置き換える人）", interactive=False,
                                                  height=440)
                            with gr.Row():
                                pick_boxes = gr.CheckboxGroup([], label="番号で選ぶこともできます", visible=False, scale=3)
                                clear_btn = gr.Button("選択をすべて外す", size="sm", scale=1)
                            picked_md = gr.Markdown()
                        with gr.Row():
                            back3 = gr.Button("← 戻る")
                            next3 = gr.Button("次へ：仕上がりを確認 →", variant="primary", elem_classes="big-button")

                    # ④ -------------------------------------------------------------------------
                    with gr.Step("仕上がり確認", id=3):
                        gr.Markdown("### 書き出す前に、仕上がりを確認しましょう")
                        with gr.Row():
                            with gr.Column(scale=3):
                                with gr.Row():
                                    prev_slider = gr.Slider(0, 1, value=0, step=1, label="確認する場面", visible=False,
                                                            scale=4)
                                    prev_btn = gr.Button("この場面で試す", variant="primary", scale=1)
                                prev_view = gr.ImageSlider(label="元 ◀ ▶ 差し替え後", interactive=False, height=480)
                                prev_msg = gr.Markdown()
                            with gr.Column(scale=2):
                                gr.Markdown("#### 仕上がりの調整\n変えたら「この場面で試す」をもう一度押してください。")
                                keep_front = gr.Checkbox(value=True, label="手や髪の毛を顔の前に残す（おすすめ）")
                                sharpen = gr.Checkbox(value=False, label="顔をくっきり補正する（処理が遅くなります）")
                                blend = gr.Slider(0.04, 0.3, value=0.12, step=0.01,
                                                  label="境目のなじませ（左：くっきり ／ 右：なめらか）")
                                color = gr.Slider(0, 1, value=0.5, step=0.05,
                                                  label="肌の色を周りに合わせる（左：弱い ／ 右：強い）")
                                with gr.Accordion("詳細設定（通常は変更不要）", open=False):
                                    strictness = gr.Slider(0.25, 0.7, value=0.4, step=0.01,
                                                           label="人物判定の厳しさ（別人まで置き換わる → 右へ／選んだ人が置き換わらない → 左へ）")
                                    smoothing = gr.Slider(0, 1, value=0.5, step=0.05,
                                                          label="動きのなめらかさ（顔がプルプル揺れる → 右へ／動きに遅れる → 左へ）")
                                    prev_matte = gr.Image(label="差し替え範囲（白い部分が差し替えられます）",
                                                          interactive=False, height=200)
                        with gr.Row():
                            back4 = gr.Button("← 戻る")
                            next4 = gr.Button("次へ：書き出す →", variant="primary", elem_classes="big-button")

                    # ⑤ -------------------------------------------------------------------------
                    with gr.Step("書き出し", id=4):
                        gr.Markdown("### ファイルに書き出します")
                        with gr.Row():
                            with gr.Column():
                                use = gr.Radio([USE_SHARE, USE_EDIT], value=USE_SHARE, label="何に使いますか？")
                                watermark = gr.Checkbox(value=False, label="画面の隅に「AI face-swapped」の文字を入れる")
                                with gr.Accordion("編集ソフトで仕上げる人向け", open=False):
                                    matte = gr.Radio([MATTE_NONE, MATTE_LUMA, MATTE_ALPHA], value=MATTE_NONE,
                                                     label="差し替え範囲のマスクも書き出す（DaVinci Resolve などで後から修正できます）")
                                    gr.Markdown("マスクは本編と同じタイムコードで書き出されるので、編集ソフトで本編に重ねてそのまま使えます。",
                                                elem_classes="hint")
                                with gr.Accordion("一部分だけ書き出す", open=False):
                                    with gr.Row():
                                        start_sec = gr.Number(value=0, label="開始（秒）")
                                        end_sec = gr.Number(value=0, label="終了（秒・0 なら最後まで）")
                                out_md = gr.Markdown()
                                gr.Markdown("ℹ️ 書き出したファイルには「AI face-swapped」という情報が自動で記録されます。",
                                            elem_classes="hint")
                                with gr.Row():
                                    render_btn = gr.Button("▶ 書き出しを開始", variant="primary", elem_classes="big-button",
                                                           scale=3)
                                    cancel_btn = gr.Button("中止", variant="stop", scale=1)
                            with gr.Column():
                                render_msg = gr.Markdown()
                                result_video = gr.Video(label="できあがり", visible=False, interactive=False)
                                reveal_btn = gr.Button("📂 保存したファイルを表示", visible=False)
                        back5 = gr.Button("← 戻る")

            with gr.Tab("まとめて処理"):
                gr.Markdown("### フォルダ内の動画・写真をまとめて処理\n"
                            "「かんたん作成」で選んだ**使う顔・置き換える人・仕上がりの設定**をそのまま使います。"
                            "先に「かんたん作成」の ①〜④ を済ませてください。")
                b_dir = gr.Textbox(label="フォルダの場所", placeholder="/Users/you/Movies/shoot_day1")
                with gr.Row():
                    b_btn = gr.Button("▶ まとめて処理を開始", variant="primary", elem_classes="big-button", scale=3)
                    b_cancel = gr.Button("中止", variant="stop", scale=1)
                b_msg = gr.Markdown()
                b_table = gr.Dataframe(headers=["ファイル", "結果", "書き出したファイル", "メモ"], interactive=False)

            with gr.Tab("同意の記録"):
                gr.Markdown("### この作品の記録\n顔の登録と書き出しは、すべて自動で記録されています。記録が後から書き換えられていないかも確認できます。")
                log_btn = gr.Button("記録を表示")
                log_status = gr.Markdown()
                log_table = gr.Dataframe(headers=["日時（UTC）", "内容", "詳細"], interactive=False)

        # ---- wiring ------------------------------------------------------------------------------
        look = [keep_front, sharpen, blend, color, strictness, smoothing]
        out_opts = [use, matte, watermark]

        id_widgets = [face_radio, face_thumb, face_msg, reg_acc]
        demo.load(on_load, [s], [s, project_dd, *id_widgets, summary, banner])
        dl_btn.click(on_download, None, [banner, dl_msg])
        project_dd.input(on_project, [project_dd, s], [s, *id_widgets, summary])
        new_project_btn.click(on_new_project, [new_project], [project_dd, new_project]).then(
            on_project, [project_dd, s], [s, *id_widgets, summary])

        face_radio.input(on_pick_identity, [face_radio, s], [s, face_thumb, summary])
        reg_btn.click(on_register, [reg_images, reg_label, reg_kind, reg_person, reg_date, reg_doc, s],
                      [s, face_radio, face_thumb, face_msg, summary, reg_acc, reg_images, reg_label, reg_person, reg_doc])

        def go(step):
            return lambda: gr.Walkthrough(selected=step)

        def need_identity(s_: Session):
            if not s_.identity:
                raise gr.Error("顔を選ぶか、新しく登録してください")
            return gr.Walkthrough(selected=1)

        next1.click(need_identity, [s], [wt])
        back2.click(go(0), None, [wt])
        tgt_upload.change(on_target, [tgt_upload, tgt_path, s], [s, tgt_info, tgt_thumb, pick_slider, prev_slider, summary])
        tgt_path.submit(on_target, [tgt_upload, tgt_path, s], [s, tgt_info, tgt_thumb, pick_slider, prev_slider, summary])
        next2.click(on_enter_who, [s, pick_slider], [s, pick_image, pick_boxes, pick_msg, picked_md, summary]).success(
            go(2), None, [wt])

        who.input(on_who, [who, s], [s, pick_image, picked_md, summary, pick_boxes])
        pick_btn.click(search_faces, [s, pick_slider], [s, pick_image, pick_boxes, pick_msg, picked_md, summary])
        pick_image.select(on_click_scene, [s], [s, pick_image, pick_boxes, picked_md, summary, who])
        pick_boxes.input(on_check_faces, [pick_boxes, s], [s, pick_image, picked_md, summary])
        clear_btn.click(on_clear, [s], [s, pick_image, pick_boxes, picked_md, summary])
        back3.click(go(1), None, [wt])

        def need_people(s_: Session):
            if s_.who == WHO_PICK and not s_.picked:
                raise gr.Error("置き換える人の顔をクリックして選んでください")
            return gr.Walkthrough(selected=3)

        next3.click(need_people, [s], [wt]).success(
            on_preview, [s, prev_slider, *look], [prev_view, prev_msg, prev_matte])
        prev_btn.click(on_preview, [s, prev_slider, *look], [prev_view, prev_msg, prev_matte])
        back4.click(go(2), None, [wt])
        next4.click(go(4), None, [wt]).then(on_use, [use, s], [out_md])
        use.change(on_use, [use, s], [out_md])

        render_btn.click(on_render, [s, *look, *out_opts, start_sec, end_sec], [s, render_msg, result_video, reveal_btn])
        cancel_btn.click(lambda: state.cancel.set(), None, None)
        reveal_btn.click(on_reveal, [s], None)
        back5.click(go(3), None, [wt])

        b_btn.click(on_batch, [s, b_dir, *look, *out_opts], [b_msg, b_table])
        b_cancel.click(lambda: state.cancel.set(), None, None)
        log_btn.click(on_log, [s], [log_status, log_table])
    return demo


def launch(project_root: Path, port: int = 7860, device: str = "auto") -> None:
    import gradio as gr

    project_root = Path(project_root)
    project_root.mkdir(parents=True, exist_ok=True)
    demo = build_ui(UIState(project_root, device))
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=port, share=False, inbrowser=True, show_error=True,
                allowed_paths=[str(project_root.resolve())],
                theme=gr.themes.Soft(primary_hue="indigo"), css=CSS, footer_links=[])
