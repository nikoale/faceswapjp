"""Command line interface. The Gradio UI (M4) calls the same functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from . import __version__, config
from .models import registry

app = typer.Typer(help="faceswapjp: local, consent-first face swap for video production.", no_args_is_help=True)
models_app = typer.Typer(help="Model files and licenses.", no_args_is_help=True)
project_app = typer.Typer(help="Projects.", no_args_is_help=True)
identity_app = typer.Typer(help="Source identities (with consent records).", no_args_is_help=True)
swap_app = typer.Typer(help="Run face swaps.", no_args_is_help=True)
log_app = typer.Typer(help="Consent / usage log.", no_args_is_help=True)
for sub, name in ((models_app, "models"), (project_app, "project"), (identity_app, "identity"), (swap_app, "swap"), (log_app, "log")):
    app.add_typer(sub, name=name)

ProjectOpt = Annotated[Path, typer.Option("--project", "-p", help="Project folder.")]
DeviceOpt = Annotated[str, typer.Option(help="auto | coreml | cuda | cpu")]


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _fail(msg: str) -> None:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Show environment, execution providers and model status."""
    import onnxruntime as ort

    from .runtime import provider_names, select_providers

    typer.echo(f"faceswapjp {__version__} / onnxruntime {ort.__version__}")
    typer.echo(f"available providers: {', '.join(ort.get_available_providers())}")
    typer.echo(f"auto selection:      {', '.join(provider_names(select_providers('auto')))}")
    typer.echo(f"models dir:          {config.models_dir()}")
    try:
        from .media.presets import OUTPUT_FORMATS, PRESETS, available_encoders

        enc = available_encoders()
        for name in OUTPUT_FORMATS:
            p = PRESETS[name]
            names = [e.name + (" (hw)" if e.hardware else "") for e in p.encoders if e.name in enc]
            typer.echo(f"format {name:14} {', '.join(names) or 'NOT AVAILABLE'}")
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"ffmpeg:              {exc}")
    models_list()


@app.command()
def formats() -> None:
    """List output formats and the encoders available on this machine."""
    from .media.presets import OUTPUT_FORMATS, PRESETS, available_encoders

    enc = available_encoders()
    for name in OUTPUT_FORMATS:
        p = PRESETS[name]
        names = [e.name + (" (hw)" if e.hardware else "") for e in p.encoders if e.name in enc]
        typer.echo(f"{name:14} {p.label}\n{'':14} encoders: {', '.join(names) or 'NOT AVAILABLE'}")


@models_app.command("list")
def models_list() -> None:
    """List models with license and status."""
    for spec in registry.load_manifest().values():
        problems = registry.verify(spec)
        status = "ok" if not problems else ("missing" if all(p.startswith("missing") for p in problems) else "CORRUPT")
        commercial = {True: "commercial OK", False: "NON-COMMERCIAL", None: "unknown"}[spec.commercial_use]
        typer.echo(f"{spec.name:18} {spec.kind:14} {status:8} {commercial:15} {spec.license}")


@models_app.command("download")
def models_download(names: Annotated[Optional[list[str]], typer.Argument()] = None) -> None:
    """Download and verify model files (from the URLs in the manifest)."""
    for name in names or list(registry.load_manifest()):
        spec = registry.get_spec(name)
        typer.echo(f"{name}: {spec.license}")
        try:
            path = registry.ensure(name, download=True)
        except registry.ModelError as exc:
            _fail(str(exc))
        typer.echo(f"  ready: {path}")


@project_app.command("init")
def project_init(
    path: Path,
    name: Annotated[Optional[str], typer.Option()] = None,
    commercial: Annotated[bool, typer.Option(help="Refuse models without a commercial license.")] = False,
) -> None:
    """Create a project folder."""
    from .project import Project

    try:
        project = Project.init(path, name, commercial)
    except FileExistsError as exc:
        _fail(str(exc))
    typer.echo(f"created project {project.settings.name!r} at {path}")


# ---- shared swap options -------------------------------------------------------------------
IdentityOpt = Annotated[str, typer.Option("--identity", help="Identity id (see `identity list`).")]
SelectOpt = Annotated[str, typer.Option("--select", help="all | largest | reference (set automatically by --only-person)")]
OnlyPersonOpt = Annotated[Optional[list[Path]], typer.Option("--only-person", help="Image(s) of the person to replace; others are left alone.")]
RefThresholdOpt = Annotated[float, typer.Option(help="Similarity needed to count as the --only-person.")]
MaskOpt = Annotated[str, typer.Option("--mask", help="Comma list: box, occlusion (hands/props), region (exclude hair/ears).")]
EnhanceOpt = Annotated[Optional[str], typer.Option("--enhance", help="Face restoration: gfpgan")]
DetailOpt = Annotated[int, typer.Option("--detail", help="Face resolution: 128 (fast), 256 (default), 512 (4K close-ups, ~16x slower).")]
EnhanceBlendOpt = Annotated[float, typer.Option(help="Blend of the enhanced face 0..1.")]
MaskBlurOpt = Annotated[float, typer.Option(help="Mask feather, fraction of the face crop.")]
ColorOpt = Annotated[float, typer.Option(help="Color match strength 0..1.")]
SmoothingOpt = Annotated[float, typer.Option(help="Landmark smoothing 0 (off)..1 (strong).")]
WatermarkOpt = Annotated[bool, typer.Option("--watermark/--no-watermark", help="Burn in a visible watermark.")]
WatermarkTextOpt = Annotated[str, typer.Option(help="Watermark text.")]
WatermarkPosOpt = Annotated[str, typer.Option(help="bottom-right | bottom-left | top-right | top-left")]
WatermarkFontOpt = Annotated[Optional[Path], typer.Option(help="TTF/OTF font for non-ASCII watermark text.")]
FormatOpt = Annotated[str, typer.Option("--format", help="h264, h265, prores_proxy, prores_lt, prores422, prores422hq, prores4444, prores4444xq, dnxhr_lb, dnxhr_sq, dnxhr_hq, dnxhr_hqx, dnxhr_444 (see `faceswapjp formats`)")]
EncoderOpt = Annotated[Optional[str], typer.Option(help="auto | software | hardware, or an ffmpeg encoder name (e.g. h264_videotoolbox).")]
QualityOpt = Annotated[str, typer.Option(help="high | standard | light (H.264 / H.265).")]
MatteOpt = Annotated[Optional[str], typer.Option("--matte", help="Also write <name>_matte.mov: luma (ProRes 422 HQ) | alpha (ProRes 4444 fill+alpha).")]
StartOpt = Annotated[int, typer.Option(help="First frame (0-based).")]
EndOpt = Annotated[Optional[int], typer.Option(help="End frame (exclusive).")]


def _job(identity, select, only_person, ref_threshold, mask, enhance, enhance_blend, mask_blur, color_strength,
         smoothing, watermark, watermark_text, watermark_position, watermark_font, device,
         fmt="h264", encoder=None, matte=None, start=0, end=None, quality="standard", detect_every=3,
         detail=256):
    from .jobs import SwapJob
    from .pipeline.frame import FrameOptions
    from .pipeline.video import RenderSettings, TrackingOptions
    from .safety.watermark import Watermark

    if detail not in (128, 256, 512):
        _fail("--detail must be 128, 256 or 512")
    for p in only_person or []:
        if not p.is_file():
            _fail(f"reference image not found: {p}")
    wm = Watermark(watermark_text, watermark_position, font_path=watermark_font) if watermark else None
    return SwapJob(
        identity_id=identity,
        frame=FrameOptions(faces="largest" if select == "largest" else "all", mask_blur=mask_blur,
                           color_strength=color_strength, enhance_blend=enhance_blend, swap_size=detail),
        tracking=TrackingOptions(select=select, reference_threshold=ref_threshold, smoothing=smoothing,
                                 detect_every=detect_every),
        render=RenderSettings(format=fmt, encoder=encoder, quality=quality, matte=matte, start=start, end=end,
                              watermark=wm),
        reference_images=list(only_person or []),
        masks=mask,
        enhancer=enhance,
        device=device,
    )


def _run(fn):
    """Run a job function, turning expected failures into clean CLI errors."""
    from .media.probe import FFmpegError
    from .safety.nsfw import NSFWContentError

    try:
        return fn()
    except NSFWContentError as exc:
        _fail(f"STOPPED: {exc}")
    except (registry.ModelError, FFmpegError, FileNotFoundError, FileExistsError, ValueError) as exc:
        _fail(str(exc))


def _progress_bar(total_label: str = "frames"):
    import sys

    def cb(done: int, total: int) -> None:
        if done == total or done % 10 == 0:
            sys.stderr.write(f"\r  {done}/{total} {total_label} ({100 * done // max(total, 1)}%)")
            if done == total:
                sys.stderr.write("\n")
            sys.stderr.flush()

    return cb


@identity_app.command("add")
def identity_add(
    project: ProjectOpt,
    label: Annotated[str, typer.Option("--label", help="Display name for this source face.")],
    images: Annotated[list[Path], typer.Option("--image", "-i", help="Reference image (repeatable).")],
    person_name: Annotated[str, typer.Option(help="Full name of the person (required).")],
    consent_date: Annotated[str, typer.Option(help="Date consent was given, YYYY-MM-DD (required).")],
    source_type: Annotated[str, typer.Option(help="consented_person | self | synthetic")],
    confirm_consent: Annotated[bool, typer.Option("--confirm-consent", help="Confirm you have the right to use this face (required).")] = False,
    consent_doc: Annotated[Optional[Path], typer.Option(help="Signed consent document (optional).")] = None,
    identity_id: Annotated[Optional[str], typer.Option("--id")] = None,
    device: DeviceOpt = "auto",
) -> None:
    """Register a source face. Consent information is mandatory."""
    from .identity import ConsentError, validate_consent
    from .jobs import add_identity
    from .project import Project

    proj = Project.load(project)
    try:
        consent = validate_consent(person_name, consent_date, source_type, confirm_consent, consent_doc)
    except ConsentError as exc:
        _fail(str(exc))
    for img in images:
        if not img.is_file():
            _fail(f"image not found: {img}")
    ident = _run(lambda: add_identity(proj, label, images, consent, device, identity_id))
    sims = ", ".join(f"{s:.2f}" for s in ident.consistency)
    typer.echo(f"registered identity {ident.id!r} ({len(images)} image(s), consistency: {sims})")


@identity_app.command("list")
def identity_list(project: ProjectOpt) -> None:
    from .identity import list_identities
    from .project import Project

    for ident in list_identities(Project.load(project)):
        c = ident.consent
        typer.echo(f"{ident.id:20} {ident.label:20} {c.source_type:17} {c.person_name} (consent {c.consent_date})")


@swap_app.command("image")
def swap_image_cmd(
    project: ProjectOpt,
    identity: IdentityOpt,
    target: Annotated[Path, typer.Option(help="Target still image.")],
    out: Annotated[Path, typer.Option(help="Output image (.png or .jpg).")],
    matte: Annotated[Optional[Path], typer.Option(help="Also write a 16-bit matte PNG.")] = None,
    select: SelectOpt = "all",
    only_person: OnlyPersonOpt = None,
    ref_threshold: RefThresholdOpt = 0.4,
    mask: MaskOpt = "box",
    enhance: EnhanceOpt = None,
    enhance_blend: EnhanceBlendOpt = 0.8,
    detail: DetailOpt = 256,
    mask_blur: MaskBlurOpt = 0.12,
    color_strength: ColorOpt = 0.5,
    watermark: WatermarkOpt = False,
    watermark_text: WatermarkTextOpt = "AI face-swapped",
    watermark_position: WatermarkPosOpt = "bottom-right",
    watermark_font: WatermarkFontOpt = None,
    device: DeviceOpt = "auto",
) -> None:
    """Swap faces in a still image."""
    from .jobs import swap_still
    from .project import Project

    proj = Project.load(project)
    if out.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        _fail("output must be .png or .jpg (metadata tagging is required)")
    if not target.is_file():
        _fail(f"target not found: {target}")
    job = _job(identity, select, only_person, ref_threshold, mask, enhance, enhance_blend, mask_blur,
               color_strength, 0.0, watermark, watermark_text, watermark_position, watermark_font, device, detail=detail)
    result = _run(lambda: swap_still(proj, job, target, out, matte))
    typer.echo(f"replaced {len(result.faces)} face(s) -> {out}")


@swap_app.command("video")
def swap_video_cmd(
    project: ProjectOpt,
    identity: IdentityOpt,
    target: Annotated[Path, typer.Option(help="Target video clip.")],
    out: Annotated[Path, typer.Option(help="Output file (.mp4 for h264, .mov for ProRes).")],
    fmt: FormatOpt = "h264",
    encoder: EncoderOpt = None,
    quality: QualityOpt = "standard",
    matte: MatteOpt = None,
    start: StartOpt = 0,
    end: EndOpt = None,
    select: SelectOpt = "all",
    only_person: OnlyPersonOpt = None,
    ref_threshold: RefThresholdOpt = 0.4,
    mask: MaskOpt = "box",
    enhance: EnhanceOpt = None,
    enhance_blend: EnhanceBlendOpt = 0.8,
    detail: DetailOpt = 256,
    mask_blur: MaskBlurOpt = 0.12,
    color_strength: ColorOpt = 0.5,
    smoothing: SmoothingOpt = 0.5,
    detect_every: Annotated[int, typer.Option(help="Full face detection every N frames (1 = every frame, slower).")] = 3,
    watermark: WatermarkOpt = False,
    watermark_text: WatermarkTextOpt = "AI face-swapped",
    watermark_position: WatermarkPosOpt = "bottom-right",
    watermark_font: WatermarkFontOpt = None,
    device: DeviceOpt = "auto",
) -> None:
    """Swap faces in a video, keeping frame rate, size, audio and timecode."""
    from .jobs import swap_video
    from .project import Project

    proj = Project.load(project)
    if not target.is_file():
        _fail(f"target not found: {target}")
    job = _job(identity, select, only_person, ref_threshold, mask, enhance, enhance_blend, mask_blur,
               color_strength, smoothing, watermark, watermark_text, watermark_position, watermark_font, device,
               fmt, encoder, matte, start, end, quality, detect_every, detail=detail)
    res = _run(lambda: swap_video(proj, job, target, out, progress=_progress_bar()))
    for w in res.warnings:
        typer.secho(f"warning: {w}", fg=typer.colors.YELLOW, err=True)
    fps = res.frames / res.seconds if res.seconds else 0
    typer.echo(f"{res.frames} frames ({res.frames_with_faces} with swaps) -> {out}  [{fps:.1f} fps]")
    if res.matte_output:
        typer.echo(f"matte -> {res.matte_output}")


@app.command()
def preview(
    project: ProjectOpt,
    identity: IdentityOpt,
    target: Annotated[Path, typer.Option(help="Video or still image.")],
    out: Annotated[Path, typer.Option(help="Preview PNG.")],
    frame: Annotated[int, typer.Option(help="Frame number (video).")] = 0,
    compare: Annotated[bool, typer.Option(help="Write original | result side by side.")] = True,
    select: SelectOpt = "all",
    only_person: OnlyPersonOpt = None,
    ref_threshold: RefThresholdOpt = 0.4,
    mask: MaskOpt = "box",
    enhance: EnhanceOpt = None,
    enhance_blend: EnhanceBlendOpt = 0.8,
    detail: DetailOpt = 256,
    mask_blur: MaskBlurOpt = 0.12,
    color_strength: ColorOpt = 0.5,
    watermark: WatermarkOpt = False,
    watermark_text: WatermarkTextOpt = "AI face-swapped",
    watermark_position: WatermarkPosOpt = "bottom-right",
    watermark_font: WatermarkFontOpt = None,
    device: DeviceOpt = "auto",
) -> None:
    """Check the result on one frame before rendering."""
    import numpy as np

    from .imageio import encode_image
    from .jobs import preview as run_preview
    from .project import Project
    from .safety.provenance import provenance_record, tag_image_bytes

    proj = Project.load(project)
    job = _job(identity, select, only_person, ref_threshold, mask, enhance, enhance_blend, mask_blur,
               color_strength, 0.0, watermark, watermark_text, watermark_position, watermark_font, device, detail=detail)
    original, result = _run(lambda: run_preview(proj, job, target, frame))
    img = np.hstack([original, result.frame]) if compare else result.frame
    out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tag_image_bytes(encode_image(img, ".png"), ".png", provenance_record(preview=True)))
    typer.echo(f"{len(result.faces)} face(s) replaced on frame {frame} -> {out}")


@app.command()
def batch(
    project: ProjectOpt,
    identity: IdentityOpt,
    input_dir: Annotated[Path, typer.Option("--input-dir", help="Folder of clips / stills.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", help="Where results and batch_report.json go.")],
    fmt: FormatOpt = "h264",
    encoder: EncoderOpt = None,
    quality: QualityOpt = "standard",
    matte: MatteOpt = None,
    select: SelectOpt = "all",
    only_person: OnlyPersonOpt = None,
    ref_threshold: RefThresholdOpt = 0.4,
    mask: MaskOpt = "box",
    enhance: EnhanceOpt = None,
    enhance_blend: EnhanceBlendOpt = 0.8,
    detail: DetailOpt = 256,
    mask_blur: MaskBlurOpt = 0.12,
    color_strength: ColorOpt = 0.5,
    smoothing: SmoothingOpt = 0.5,
    detect_every: Annotated[int, typer.Option(help="Full face detection every N frames (1 = every frame, slower).")] = 3,
    watermark: WatermarkOpt = False,
    watermark_text: WatermarkTextOpt = "AI face-swapped",
    watermark_position: WatermarkPosOpt = "bottom-right",
    watermark_font: WatermarkFontOpt = None,
    device: DeviceOpt = "auto",
) -> None:
    """Process every clip in a folder with the same settings. Failed clips are skipped and reported."""
    import sys

    from .jobs import batch as run_batch
    from .project import Project

    proj = Project.load(project)
    if not input_dir.is_dir():
        _fail(f"not a folder: {input_dir}")
    job = _job(identity, select, only_person, ref_threshold, mask, enhance, enhance_blend, mask_blur,
               color_strength, smoothing, watermark, watermark_text, watermark_position, watermark_font, device,
               fmt, encoder, matte, quality=quality, detect_every=detect_every, detail=detail)

    def progress(i, n, done, total):
        if done == total or done % 10 == 0:
            sys.stderr.write(f"\r  clip {i + 1}/{n}: {done}/{total} frames")
            if done == total:
                sys.stderr.write("\n")

    report = _run(lambda: run_batch(proj, job, input_dir, output_dir, progress=progress))
    ok = sum(r.ok for r in report)
    for r in report:
        mark = "ok  " if r.ok else "FAIL"
        typer.echo(f"{mark} {Path(r.source).name}" + (f"  ({r.error})" if r.error else ""))
    typer.echo(f"{ok}/{len(report)} succeeded; report: {output_dir / 'batch_report.json'}")
    if ok < len(report):
        raise typer.Exit(2)


@app.command()
def ui(
    project_root: Annotated[Path, typer.Option("--projects", help="Folder that holds projects.")] = Path("projects"),
    port: Annotated[int, typer.Option()] = 7860,
    device: DeviceOpt = "auto",
    classic: Annotated[bool, typer.Option("--classic", help="Use the older Gradio UI.")] = False,
    no_browser: Annotated[bool, typer.Option("--no-browser", help="Do not open a browser window.")] = False,
) -> None:
    """Start the local studio UI in your browser (bound to 127.0.0.1 only)."""
    if classic:
        from .ui.app import launch as launch_classic

        launch_classic(project_root, port=port, device=device)
        return
    from .web.server import launch

    launch(project_root, port=port, device=device, open_browser=not no_browser)


@app.command()
def bench(
    target: Annotated[Path, typer.Option(help="A video to measure with (use your own footage).")],
    frames: Annotated[int, typer.Option(help="How many frames to process.")] = 48,
    start: Annotated[int, typer.Option(help="First frame.")] = 0,
    mask: MaskOpt = "box",
    enhance: EnhanceOpt = None,
    detail: DetailOpt = 256,
    detect_every: Annotated[int, typer.Option(help="Full face detection every N frames (1 = every frame).")] = 3,
    fmt: FormatOpt = "h264",
    encoder: EncoderOpt = None,
    device: DeviceOpt = "auto",
) -> None:
    """Measure speed on this machine: fps and time per stage. The output is deleted afterwards.

    The first face in the first frame is swapped with itself, so no identity or project is needed.
    """
    from .bench import run_bench, summary_text

    if not target.is_file():
        _fail(f"target not found: {target}")
    result = _run(lambda: run_bench(target, frames, start, mask, enhance, detect_every, fmt, encoder, device,
                                    swap_size=detail, progress=_progress_bar()))
    typer.echo("\n" + summary_text(result))


@log_app.command("verify")
def log_verify(project: ProjectOpt) -> None:
    """Check the consent log hash chain."""
    from .project import Project

    log = Project.load(project).consent_log
    problems = log.verify()
    if problems:
        _fail("; ".join(problems))
    typer.echo(f"ok: {len(log.entries())} entries, chain intact")


if __name__ == "__main__":
    app()
