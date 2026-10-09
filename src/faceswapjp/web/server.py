"""Local web UI backend (FastAPI). Serves the single-page studio UI in ./static.

Only binds to 127.0.0.1. Every /api request must carry the per-launch token (header or ?t=)
and a localhost Host header, so other websites open in the same browser cannot drive it
(CSRF / DNS rebinding).
"""

import base64
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..compositing.blend import from_float, to_float
from ..identity import SOURCE_TYPES, ConsentError, list_identities, load_identity, validate_consent
from ..jobs import SwapJob, add_identity, batch, find_faces, is_video, preview, swap_still, swap_video
from ..imageio import SUPPORTED_EXT, read_image
from ..media.ffpipe import FrameReader
from ..media.presets import OUTPUT_FORMATS, PRESETS, available_encoders
from ..media.probe import FFmpegError, VideoInfo, probe
from ..messages import friendly
from ..models import registry
from ..pipeline.frame import FrameOptions
from ..pipeline.video import RenderSettings, TrackingOptions
from ..project import PROJECT_FILE, Project
from ..runtime import provider_names, select_providers
from ..safety.watermark import Watermark

STATIC = Path(__file__).parent / "static"
DEFAULT_PROJECT = "はじめての作品"


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


# ---------------------------------------------------------------------------------------------
# state


@dataclass
class Target:
    id: str
    path: Path
    kind: str  # video | image
    info: VideoInfo | None
    width: int
    height: int
    frames: int
    fps: float
    display_name: str = ""
    waveform: bytes | None = None

    def describe(self) -> dict[str, Any]:
        d = {"id": self.id, "name": self.display_name or self.path.name, "path": str(self.path), "kind": self.kind,
             "width": self.width, "height": self.height, "frames": self.frames, "fps": self.fps}
        if self.info is not None:
            i = self.info
            d.update(fps_exact=f"{i.fps.numerator}/{i.fps.denominator}", duration=self.frames / self.fps if self.fps else 0,
                     timecode=i.timecode, audio=i.audio_codecs, codec=i.codec, pix_fmt=i.pix_fmt, bit_depth=i.bit_depth,
                     vfr=i.is_vfr, interlaced=i.is_interlaced)
        return d


@dataclass
class Job:
    id: str
    kind: str
    title: str
    status: str = "queued"  # queued | running | done | error | cancelled
    done: int = 0
    total: int = 0
    started: float = 0.0
    message: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)
    live: bytes | None = None  # latest rendered frame (JPEG) for the live view
    live_version: int = 0
    live_frame: int = 0
    _live_at: float = 0.0

    def offer_frame(self, index: int, frame: np.ndarray, every: float = 0.5) -> None:
        """Keep a downscaled copy of a recent frame, at most every `every` seconds."""
        now = time.time()
        if now - self._live_at < every:
            return
        self._live_at = now
        self.live, self.live_frame = jpeg(frame, 1280, 80), index
        self.live_version += 1

    def public(self) -> dict[str, Any]:
        elapsed = time.time() - self.started if self.started else 0.0
        rate = self.done / elapsed if elapsed > 0 and self.done else 0.0
        eta = (self.total - self.done) / rate if rate > 0 and self.total else None
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status, "done": self.done,
                "total": self.total, "fps": round(rate, 2), "eta": eta, "elapsed": elapsed,
                "message": self.message, "result": self.result, "live": self.live_version,
                "live_frame": self.live_frame}


class Context:
    def __init__(self, root: Path, device: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.targets: dict[str, Target] = {}
        self.blobs: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fsj-job")
        self.lock = threading.Lock()
        if not self.projects():
            Project.init(self.root / DEFAULT_PROJECT, DEFAULT_PROJECT)

    # projects -------------------------------------------------------------------------------
    def projects(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / PROJECT_FILE).exists())

    def project(self, name: str | None) -> Project:
        if not name or "/" in name or name.startswith("."):
            raise ApiError("作品を選んでください")
        try:
            return Project.load(self.root / name)
        except FileNotFoundError:
            raise ApiError(f"作品「{name}」が見つかりません", 404) from None

    def within_root(self, path: Path) -> Path:
        p = Path(path).resolve()
        if self.root.resolve() not in p.parents:
            raise ApiError("この場所のファイルは開けません", 403)
        return p

    # blobs (preview images) -----------------------------------------------------------------
    def put_blob(self, data: bytes, mime: str = "image/jpeg") -> str:
        bid = uuid.uuid4().hex
        with self.lock:
            self.blobs[bid] = (data, mime)
            while len(self.blobs) > 60:
                self.blobs.popitem(last=False)
        return f"/api/blob/{bid}"

    # targets --------------------------------------------------------------------------------
    def add_target(self, path: Path, display_name: str = "") -> Target:
        path = Path(path).expanduser()
        if not path.is_file():
            raise ApiError("ファイルが見つかりません。パスを確認してください。")
        tid = uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())).hex[:12]
        if is_video(path):
            info = probe(path)
            t = Target(tid, path, "video", info, info.width, info.height, max(1, info.nb_frames), float(info.fps))
        elif path.suffix.lower() in SUPPORTED_EXT:
            img = read_image(path)
            t = Target(tid, path, "image", None, img.shape[1], img.shape[0], 1, 0.0)
        else:
            raise ApiError("動画（mov / mp4 など）か写真（png / jpg など）を選んでください")
        t.display_name = display_name or path.name
        self.targets[tid] = t
        return t

    def target(self, tid: str) -> Target:
        if tid not in self.targets:
            raise ApiError("素材をもう一度読み込んでください", 404)
        return self.targets[tid]

    def read_frame(self, t: Target, n: int) -> np.ndarray:
        if t.kind == "image":
            return read_image(t.path)
        n = min(max(0, int(n)), t.frames - 1)
        return FrameReader(t.info, start=n, count=1).read_one()

    # jobs -----------------------------------------------------------------------------------
    def submit(self, kind: str, title: str, fn) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind, title)
        with self.lock:
            self.jobs[job.id] = job
            while len(self.jobs) > 30:
                self.jobs.popitem(last=False)

        def run():
            job.status, job.started = "running", time.time()
            try:
                job.result = fn(job) or {}
                job.status = "cancelled" if job.cancel.is_set() else "done"
            except InterruptedError:
                job.status, job.message = "cancelled", "中止しました。途中までのファイルは使えません。"
            except Exception as exc:  # noqa: BLE001
                job.status, job.message = "error", friendly(exc)

        self.executor.submit(run)
        return job


# ---------------------------------------------------------------------------------------------
# helpers


def jpeg(img: np.ndarray, max_side: int = 1600, quality: int = 88) -> bytes:
    img8 = img[..., :3] if img.dtype == np.uint8 else from_float(to_float(img[..., :3]), np.uint8)
    h, w = img8.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img8 = cv2.resize(img8, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def data_url(img: np.ndarray, max_side: int = 160) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg(img, max_side, 85)).decode()


def swap_size(value: Any) -> int:
    try:
        size = int(value if value not in (None, "") else 256)
    except (TypeError, ValueError):
        raise ApiError("顔の解像度が正しくありません") from None
    if size not in (128, 256, 512):
        raise ApiError("顔の解像度が正しくありません")
    return size


def job_from(p: dict[str, Any], t: Target | None) -> SwapJob:
    if not p.get("identity"):
        raise ApiError("使う顔を選んでください")
    who = p.get("who", "pick")
    picked = p.get("picked") or []
    if who == "pick" and not picked:
        raise ApiError("置き換える人を選んでください")
    refs = np.asarray(picked, dtype=np.float32) if who == "pick" else None
    if refs is not None and (refs.ndim != 2 or refs.shape[1] != 512):
        raise ApiError("選んだ人の情報が正しくありません。もう一度選び直してください")
    start = int(p.get("in_frame") or 0)
    out_frame = p.get("out_frame")
    end = int(out_frame) + 1 if out_frame not in (None, "") and t and int(out_frame) < t.frames - 1 else None
    fmt = p.get("codec") or "h264"
    if fmt not in OUTPUT_FORMATS:
        raise ApiError("書き出し形式が正しくありません")
    return SwapJob(
        identity_id=p["identity"],
        frame=FrameOptions(mask_blur=float(p.get("blend", 0.12)), color_strength=float(p.get("color", 0.5)),
                           enhance_blend=float(p.get("enhance_blend", 0.8)),
                           swap_size=swap_size(p.get("detail"))),
        tracking=TrackingOptions(select="reference" if who == "pick" else "all",
                                 reference_threshold=float(p.get("strictness", 0.4)),
                                 smoothing=float(p.get("smoothing", 0.5))),
        render=RenderSettings(format=fmt, encoder=p.get("encoder") or "auto", quality=p.get("quality") or "standard",
                              matte=p.get("matte") or None, start=start, end=end,
                              watermark=Watermark() if p.get("watermark") else None),
        reference_embeddings=refs,
        masks="box,occlusion,region" if p.get("keep_front", True) else "box",
        enhancer="gfpgan" if p.get("sharpen") else None,
        device=p.get("device") or "auto",
    )


def reveal(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=False)
    elif sys.platform.startswith("win"):
        subprocess.run(["explorer", "/select,", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path.parent)], check=False)


# ---------------------------------------------------------------------------------------------
# app


def create_app(root: Path, device: str = "auto", token: str | None = None):
    ctx = Context(root, device)
    token = token or secrets.token_urlsafe(24)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, title="faceswapjp")
    app.state.ctx = ctx
    app.state.token = token

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = (request.headers.get("host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost", "[::1]", "testserver"):
            return JSONResponse({"error": "forbidden host"}, status_code=403)
        if request.url.path.startswith("/api/"):
            supplied = request.headers.get("x-fsj-token") or request.query_params.get("t")
            if not supplied or not secrets.compare_digest(supplied, token):
                return JSONResponse({"error": "forbidden"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ApiError)
    async def api_error(_, exc: ApiError):
        return JSONResponse({"error": exc.message}, status_code=exc.status)

    @app.exception_handler(Exception)
    async def any_error(_, exc: Exception):
        return JSONResponse({"error": friendly(exc)}, status_code=400)

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        return html.replace("__FSJ_TOKEN__", token).replace("__FSJ_VERSION__", __version__)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # ---- environment ----------------------------------------------------------------------
    @app.get("/api/state")
    def state():
        try:
            enc = available_encoders()
            ffmpeg_ok = True
        except FFmpegError:
            enc, ffmpeg_ok = frozenset(), False
        formats = []
        for name in OUTPUT_FORMATS:
            pr = PRESETS[name]
            formats.append({
                "name": name, "label": pr.label, "ext": pr.ext, "group": pr.group, "has_quality": pr.has_quality,
                "software": [e.name for e in pr.encoders if not e.hardware and e.name in enc],
                "hardware": [e.name for e in pr.encoders if e.hardware and e.name in enc],
            })
        missing = [n for n, s in registry.load_manifest().items() if not registry.present(s)]
        return {"version": __version__, "projects": ctx.projects(), "missing_models": missing, "ffmpeg": ffmpeg_ok,
                "providers": provider_names(select_providers(ctx.device)), "formats": formats,
                "source_types": list(SOURCE_TYPES), "today": date.today().isoformat()}

    @app.post("/api/models/download")
    def download_models():
        def run(job: Job):
            missing = [n for n, s in registry.load_manifest().items() if not registry.present(s)]
            job.total = len(missing)
            for name in missing:
                job.message = f"{name} をダウンロード中…"
                registry.ensure(name, download=True)
                job.done += 1
            return {}

        return ctx.submit("download", "AI モデルのダウンロード", run).public()

    # ---- projects & identities ------------------------------------------------------------
    @app.post("/api/projects")
    async def create_project(request: Request):
        name = ((await request.json()).get("name") or "").strip()
        if not name or "/" in name or name.startswith("."):
            raise ApiError("作品名を入力してください（「/」は使えません）")
        try:
            Project.init(ctx.root / name, name)
        except FileExistsError:
            raise ApiError(f"「{name}」はすでにあります") from None
        return {"projects": ctx.projects(), "project": name}

    @app.get("/api/projects/{name}/identities")
    def identities(name: str):
        project = ctx.project(name)
        return [{"id": i.id, "label": i.label, "person": i.consent.person_name, "type": i.consent.source_type,
                 "date": i.consent.consent_date, "images": len(i.reference_images),
                 "thumb": f"/api/projects/{name}/identities/{i.id}/thumb"} for i in list_identities(project)]

    @app.get("/api/projects/{name}/identities/{iid}/thumb")
    def identity_thumb(name: str, iid: str):
        project = ctx.project(name)
        ident, _ = load_identity(project, iid)
        img = read_image(project.identities_dir / iid / ident.reference_images[0])
        return Response(jpeg(img, 320), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/projects/{name}/identities")
    async def register(name: str, images: list[UploadFile] = File(...), label: str = Form(""),
                       person: str = Form(""), consent_date: str = Form(""), source_type: str = Form("self"),
                       attested: str = Form("false"), consent_doc: UploadFile | None = File(None)):
        project = ctx.project(name)
        tmp = project.root / "uploads" / f"reg_{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)
        try:
            paths = []
            for i, f in enumerate(images):
                dst = tmp / f"{i:02d}{Path(f.filename or 'x.jpg').suffix.lower()}"
                dst.write_bytes(await f.read())
                paths.append(dst)
            doc = None
            if consent_doc is not None and consent_doc.filename:
                data = await consent_doc.read()
                doc = project.root / "consent" / (uuid.uuid4().hex[:16] + Path(consent_doc.filename).suffix.lower())
                doc.parent.mkdir(exist_ok=True)
                doc.write_bytes(data)
            consent = validate_consent(person, consent_date, source_type, attested == "true", doc)
            ident = add_identity(project, label.strip() or person.strip(), paths, consent, ctx.device)
        except ConsentError as exc:
            raise ApiError(friendly(exc)) from None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        low = [i + 1 for i, c in enumerate(ident.consistency) if c < 0.45]
        return {"id": ident.id, "warning": f"{', '.join(map(str, low))} 枚目は他の写真と別人のように見えます" if low else None}

    @app.get("/api/projects/{name}/log")
    def consent_log(name: str):
        log = ctx.project(name).consent_log
        return {"problems": log.verify(), "entries": list(reversed(log.entries()))}

    # ---- targets --------------------------------------------------------------------------
    @app.post("/api/targets/upload")
    async def upload_target(project: str = Form(...), file: UploadFile = File(...)):
        proj = ctx.project(project)
        dst = proj.root / "uploads" / f"{uuid.uuid4().hex[:6]}_{Path(file.filename or 'clip').name}"
        dst.parent.mkdir(exist_ok=True)
        with open(dst, "wb") as out:
            while chunk := await file.read(1 << 22):
                out.write(chunk)
        return ctx.add_target(dst, Path(file.filename or dst.name).name).describe()

    @app.post("/api/targets/path")
    async def target_from_path(request: Request):
        return ctx.add_target(Path((await request.json()).get("path", "").strip())).describe()

    @app.get("/api/targets/{tid}/frame")
    def frame(tid: str, n: int = 0, w: int = 1600):
        t = ctx.target(tid)
        return Response(jpeg(ctx.read_frame(t, n), max(64, min(w, 3840))), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=600"})

    @app.get("/api/targets/{tid}/waveform")
    def waveform(tid: str):
        t = ctx.target(tid)
        if t.kind != "video" or not t.info.has_audio:
            raise ApiError("音声がありません", 404)
        if t.waveform is None:
            from ..media.probe import ffmpeg_bin

            cmd = [ffmpeg_bin(), "-v", "error", "-i", str(t.path), "-filter_complex",
                   "[0:a:0]aformat=channel_layouts=mono,"
                   "showwavespic=s=2400x96:colors=0x9a8cff:scale=sqrt", "-frames:v", "1",
                   "-f", "image2pipe", "-vcodec", "png", "-"]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0 or not proc.stdout:
                raise ApiError("波形を作れませんでした", 404)
            t.waveform = proc.stdout
        return Response(t.waveform, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/targets/{tid}/faces")
    async def faces(tid: str, request: Request):
        body = await request.json()
        t = ctx.target(tid)
        n = int(body.get("frame", 0))
        project = ctx.project(body.get("project")) if body.get("project") else None
        frame_img, found = find_faces(project, t.path, n, ctx.device)
        h, w = frame_img.shape[:2]
        out = []
        for i, f in enumerate(found):
            x0, y0, x1, y1 = f.bbox
            m = 0.25 * (x1 - x0)
            crop = frame_img[max(0, int(y0 - m)):min(h, int(y1 + m)), max(0, int(x0 - m)):min(w, int(x1 + m))]
            out.append({"i": i, "box": [float(x0 / w), float(y0 / h), float(x1 / w), float(y1 / h)],
                        "emb": [round(float(v), 5) for v in f.embedding], "crop": data_url(crop, 160)})
        return {"frame": n, "faces": out}

    # ---- preview & render ------------------------------------------------------------------
    @app.post("/api/preview")
    async def run_preview(request: Request):
        body = await request.json()
        t = ctx.target(body.get("target", ""))
        job = job_from(body, t)
        original, result = preview(ctx.project(body.get("project")), job, t.path, int(body.get("frame", 0)))
        matte = cv2.cvtColor((np.clip(result.matte, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return {"before": ctx.put_blob(jpeg(original)), "after": ctx.put_blob(jpeg(result.frame)),
                "matte": ctx.put_blob(jpeg(matte, 1280)), "faces": len(result.faces)}

    @app.get("/api/blob/{bid}")
    def blob(bid: str):
        if bid not in ctx.blobs:
            raise ApiError("画像の有効期限が切れました", 404)
        data, mime = ctx.blobs[bid]
        return Response(data, media_type=mime)

    def output_path(project: Project, t: Target, fmt: str, requested: str | None) -> Path:
        if requested:
            return Path(requested).expanduser()
        ext = PRESETS[fmt].ext if t.kind == "video" else ".png"
        stem = Path(t.display_name or t.path.name).stem
        out = project.renders_dir / f"{stem}_swap{ext}"
        n = 2
        while out.exists():
            out = project.renders_dir / f"{stem}_swap_{n}{ext}"
            n += 1
        return out

    @app.post("/api/render")
    async def render(request: Request):
        body = await request.json()
        t = ctx.target(body.get("target", ""))
        project = ctx.project(body.get("project"))
        job_spec = job_from(body, t)
        out = output_path(project, t, job_spec.render.format, body.get("output"))

        def run(job: Job):
            if t.kind == "video":
                job.total = (job_spec.render.end or t.frames) - job_spec.render.start
                job.message = "安全チェック中…"

                def progress(d, total):
                    job.done, job.total, job.message = d, total, "処理中…"

                res = swap_video(project, job_spec, t.path, out, progress=progress, cancel=job.cancel,
                                 on_frame=job.offer_frame)
                return {"output": str(out), "matte": str(res.matte_output) if res.matte_output else None,
                        "frames": res.frames, "seconds": res.seconds, "warnings": res.warnings,
                        "playable": out.suffix == ".mp4", "url": f"/api/file?path={out}"}
            job.total = 1
            matte = out.with_name(f"{out.stem}_matte.png") if job_spec.render.matte else None
            swap_still(project, job_spec, t.path, out, matte)
            job.done = 1
            return {"output": str(out), "matte": str(matte) if matte else None, "frames": 1, "warnings": [],
                    "image": True, "url": f"/api/file?path={out}"}

        return ctx.submit("render", t.display_name or t.path.name, run).public()

    @app.post("/api/batch")
    async def run_batch(request: Request):
        body = await request.json()
        project = ctx.project(body.get("project"))
        folder = Path((body.get("folder") or "").strip()).expanduser()
        if not folder.is_dir():
            raise ApiError("フォルダが見つかりません。パスを確認してください。")
        job_spec = job_from(body, None)
        out = project.renders_dir / f"batch_{folder.name}"

        def run(job: Job):
            def progress(i, n, d, total):
                job.message = f"{i + 1} / {n} 本目"
                job.done, job.total = d, total

            report = batch(project, job_spec, folder, out, progress=progress, cancel=job.cancel)
            return {"folder": str(out), "items": [{"source": Path(r.source).name, "ok": r.ok,
                                                   "output": r.output, "error": friendly(Exception(r.error)) if r.error else None}
                                                  for r in report]}

        return ctx.submit("batch", folder.name, run).public()

    @app.post("/api/bench")
    async def bench(request: Request):
        from ..bench import run_bench, summary_text

        body = await request.json()
        t = ctx.target(body.get("target", ""))
        if t.kind != "video":
            raise ApiError("速度チェックには動画を読み込んでください")
        frames = max(12, min(int(body.get("frames") or 48), 240))

        def run(job: Job):
            job.total = frames
            job.message = "AI モデルを準備しています…"

            def progress(d, total):
                job.done, job.total, job.message = d, total, "測定中…"

            result = run_bench(t.path, frames=frames, start=int(body.get("start") or 0),
                               masks="box,occlusion,region" if body.get("keep_front") else "box",
                               enhancer="gfpgan" if body.get("sharpen") else None,
                               detect_every=int(body.get("detect_every") or 3), device=ctx.device,
                               swap_size=swap_size(body.get("detail")),
                               progress=progress, cancel=job.cancel)
            result["text"] = summary_text(result)
            return result

        return ctx.submit("bench", "速度チェック", run).public()

    @app.get("/api/jobs/{jid}")
    def job_status(jid: str):
        if jid not in ctx.jobs:
            raise ApiError("処理が見つかりません", 404)
        return ctx.jobs[jid].public()

    @app.get("/api/jobs/{jid}/live")
    def job_live(jid: str):
        job = ctx.jobs.get(jid)
        if job is None or job.live is None:
            raise ApiError("まだ表示できるフレームがありません", 404)
        return Response(job.live, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        if jid in ctx.jobs:
            ctx.jobs[jid].cancel.set()
        return {"ok": True}

    @app.get("/api/file")
    def file(path: str):
        p = ctx.within_root(Path(path))
        if not p.is_file():
            raise ApiError("ファイルが見つかりません", 404)
        return FileResponse(p)

    @app.post("/api/shutdown")
    def shutdown():
        """Stop the local server (the UI's 終了 button). Running jobs are cancelled."""
        import os
        import signal

        for job in ctx.jobs.values():
            job.cancel.set()
        threading.Timer(0.6, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
        return {"ok": True}

    @app.post("/api/reveal")
    async def reveal_file(request: Request):
        p = ctx.within_root(Path((await request.json()).get("path", "")))
        reveal(p)
        return {"ok": True}

    return app


def launch(root: Path, port: int = 7860, device: str = "auto", open_browser: bool = True) -> None:
    import uvicorn

    app = create_app(root, device)
    url = f"http://127.0.0.1:{port}/"
    if open_browser:
        threading.Timer(1.2, lambda: __import__("webbrowser").open(url)).start()
    print(f"faceswapjp studio: {url}  (Ctrl+C で終了)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
