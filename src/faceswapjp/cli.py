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
    models_list()


@models_app.command("list")
def models_list() -> None:
    """List models with license and status."""
    for spec in registry.load_manifest().values():
        problems = registry.verify(spec)
        status = "ok" if not problems else ("missing" if all(p.startswith("missing") for p in problems) else "CORRUPT")
        commercial = {True: "commercial OK", False: "NON-COMMERCIAL", None: "unknown"}[spec.commercial_use]
        typer.echo(f"{spec.name:16} {spec.kind:14} {status:8} {commercial:15} {spec.license}")


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


@identity_app.command("add")
def identity_add(
    project: ProjectOpt,
    label: Annotated[str, typer.Option("--label", help="Display name for this source face.")],
    images: Annotated[list[Path], typer.Option("--image", "-i", help="Reference image (repeatable).")],
    person_name: Annotated[str, typer.Option(help="Full name of the person (required).")],
    consent_date: Annotated[str, typer.Option(help="Date consent was given, YYYY-MM-DD (required).")],
    consent_doc: Annotated[Path, typer.Option(help="Path to the signed consent document (required).")],
    source_type: Annotated[str, typer.Option(help="consented_person | self | synthetic")],
    identity_id: Annotated[Optional[str], typer.Option("--id")] = None,
    device: DeviceOpt = "auto",
) -> None:
    """Register a source face. Consent information is mandatory."""
    from .engine import build_engine
    from .identity import ConsentError, register_identity, validate_consent
    from .project import Project

    proj = Project.load(project)
    try:
        consent = validate_consent(person_name, consent_date, consent_doc, source_type)
    except ConsentError as exc:
        _fail(str(exc))
    for img in images:
        if not img.is_file():
            _fail(f"image not found: {img}")
    try:
        engine = build_engine(proj, device=device, swapper_name=None)
        ident = register_identity(proj, engine.analyzer, label, images, consent, engine.analyzer_name, identity_id)
    except (registry.ModelError, ValueError, FileExistsError) as exc:
        _fail(str(exc))
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
    identity: Annotated[str, typer.Option("--identity", help="Identity id (see `identity list`).")],
    target: Annotated[Path, typer.Option(help="Target still image.")],
    out: Annotated[Path, typer.Option(help="Output image (.png or .jpg).")],
    faces: Annotated[str, typer.Option(help="all | largest")] = "all",
    mask_blur: Annotated[float, typer.Option(help="Mask feather, fraction of the face crop.")] = 0.12,
    color_strength: Annotated[float, typer.Option(help="Color match strength 0..1.")] = 0.5,
    matte: Annotated[Optional[Path], typer.Option(help="Also write a 16-bit matte PNG.")] = None,
    device: DeviceOpt = "auto",
) -> None:
    """Swap faces in a still image."""
    from .engine import build_engine
    from .identity import load_identity
    from .models.registry import sha256_file
    from .pipeline.frame import FrameOptions, FrameProcessor
    from .pipeline.image import swap_image
    from .project import Project

    proj = Project.load(project)
    if out.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        _fail("output must be .png or .jpg (metadata tagging is required)")
    if not target.is_file():
        _fail(f"target not found: {target}")
    try:
        ident, embedding = load_identity(proj, identity)
        engine = build_engine(proj, device=device)
    except (FileNotFoundError, registry.ModelError) as exc:
        _fail(str(exc))
    processor = FrameProcessor(engine.analyzer, engine.swapper, FrameOptions(faces=faces, mask_blur=mask_blur, color_strength=color_strength))
    try:
        result = swap_image(
            processor,
            embedding,
            target,
            out,
            provenance={"models": [engine.analyzer_name, engine.swapper_name], "identity": ident.id},
            matte_output=matte,
        )
    except ValueError as exc:
        _fail(str(exc))
    proj.consent_log.append(
        "render",
        kind="image",
        identity_id=ident.id,
        target=str(target.resolve()),
        target_sha256=sha256_file(target),
        output=str(out.resolve()),
        output_sha256=sha256_file(out),
        faces_replaced=len(result.faces),
        models=[engine.analyzer_name, engine.swapper_name],
        providers=engine.provider_names,
    )
    typer.echo(f"replaced {len(result.faces)} face(s) -> {out}  [{', '.join(engine.provider_names)}]")


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
