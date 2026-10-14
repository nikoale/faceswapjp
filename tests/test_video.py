import json
import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from faceswapjp.media.ffpipe import FrameReader, FrameWriter
from faceswapjp.media.presets import get_preset
from faceswapjp.media.probe import frames_to_tc, offset_timecode, probe, tc_to_frames
from faceswapjp.pipeline.frame import FrameOptions, FrameProcessor
from faceswapjp.pipeline.video import RenderSettings, TrackedProcessor, TrackingOptions, render_video
from faceswapjp.safety.watermark import Watermark

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.mark.parametrize(
    "tc,fps,offset,expected",
    [
        ("01:00:00:00", Fraction(24000, 1001), 24, "01:00:01:00"),
        ("00:59:59:23", Fraction(24), 1, "01:00:00:00"),
        ("00:00:59;29", Fraction(30000, 1001), 1, "00:01:00;02"),  # drop frame skips ;00 and ;01
        ("00:09:59;29", Fraction(30000, 1001), 1, "00:10:00;00"),  # ...except every 10th minute
        ("10:00:00:00", Fraction(25), 0, "10:00:00:00"),
    ],
)
def test_timecode_offsets(tc, fps, offset, expected):
    assert offset_timecode(tc, fps, offset) == expected


def test_timecode_roundtrip_drop_frame():
    fps = Fraction(60000, 1001)
    for n in (0, 1, 3595, 3600, 35964, 107892, 123456):
        assert tc_to_frames(frames_to_tc(n, fps, drop=True), fps) == n


def _make_clip(path, rate="24000/1001", frames=24, size="320x240", tc="01:00:00:00", audio=True):
    dur = float(Fraction(rate) ** -1 * frames)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={dur}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={dur}", "-c:a", "pcm_s24le"]
    cmd += ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-timecode", tc,
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", str(path)]
    subprocess.run(cmd, check=True)
    return path


@needs_ffmpeg
def test_reader_seek_is_frame_accurate(tmp_path):
    info = probe(_make_clip(tmp_path / "src.mov"))
    assert info.nb_frames == 24 and info.fps == Fraction(24000, 1001) and info.bit_depth == 10
    full = list(FrameReader(info))
    part = list(FrameReader(info, start=7, count=3))
    assert len(full) == 24 and len(part) == 3 and part[0].dtype == np.uint16
    assert all(np.array_equal(full[7 + i], part[i]) for i in range(3))


@needs_ffmpeg
@pytest.mark.parametrize("fmt", ["prores422hq", "h264"])
def test_writer_keeps_fps_timecode_audio_and_tags(tmp_path, fmt):
    src = _make_clip(tmp_path / "src.mov")
    info = probe(src)
    out = tmp_path / ("out" + get_preset(fmt).ext)
    with FrameWriter(out, info, get_preset(fmt), "bgr48le", timecode=info.timecode, audio_source=src,
                     metadata={"comment": "AI face-swapped"}) as w:
        for f in FrameReader(info):
            w.write(f)
    o = probe(out)
    assert (o.fps, o.nb_frames, o.timecode) == (info.fps, 24, "01:00:00:00")
    assert o.has_audio and abs(o.duration - info.duration) < 0.05
    assert o.color_space == "bt709"
    tags = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "csv=p=0", str(out)],
                          capture_output=True, text=True).stdout
    assert "AI face-swapped" in tags


@needs_ffmpeg
def test_prores_roundtrip_is_close(tmp_path):
    src = _make_clip(tmp_path / "src.mov", frames=2)
    info = probe(src)
    out = tmp_path / "o.mov"
    frames = list(FrameReader(info))
    with FrameWriter(out, info, get_preset("prores422hq"), "bgr48le") as w:
        for f in frames:
            w.write(f)
    again = list(FrameReader(probe(out)))
    err = np.abs(frames[1].astype(np.int64) - again[1].astype(np.int64))
    assert np.median(err) < 300  # of 65535: untouched areas survive the 10-bit 4:2:2 round trip


@needs_ffmpeg
@pytest.mark.parametrize("matte", ["luma", "alpha"])
def test_render_video_partial_range_with_matte(tmp_path, fake_analyzer, solid_swapper, matte):
    src = _make_clip(tmp_path / "src.mov", frames=24)
    proc = FrameProcessor(fake_analyzer, solid_swapper, FrameOptions(color_strength=0))
    tracked = TrackedProcessor(proc, np.zeros(512, np.float32), 23.976)
    out = tmp_path / "out.mov"
    res = render_video(tracked, src, out, RenderSettings(format="prores422hq", matte=matte, start=12, end=20,
                                                         watermark=Watermark()))
    assert res.frames == 8 and res.frames_with_faces == 8 and not res.warnings
    o, m = probe(out), probe(res.matte_output)
    assert o.nb_frames == m.nb_frames == 8
    assert o.timecode == m.timecode == "01:00:00:12"
    assert abs(o.duration - 8 / float(Fraction(24000, 1001))) < 0.05
    assert m.has_alpha == (matte == "alpha")


@needs_ffmpeg
def test_render_rejects_wrong_extension(tmp_path, fake_analyzer, solid_swapper):
    src = _make_clip(tmp_path / "src.mov", frames=2, audio=False)
    tracked = TrackedProcessor(FrameProcessor(fake_analyzer, solid_swapper), np.zeros(512), 24)
    with pytest.raises(ValueError, match=".mov"):
        render_video(tracked, src, tmp_path / "x.mp4", RenderSettings(format="prores422hq"))


def test_tracked_processor_reference_selection(solid_swapper):
    from faceswapjp.analysis.face import Face

    a, b = np.eye(512, dtype=np.float32)[0], np.eye(512, dtype=np.float32)[1]

    class TwoFaces:
        def detect(self, image, with_embedding=True):
            kps = np.array([[30, 40], [60, 40], [45, 55], [33, 70], [57, 70]], np.float32)
            return [
                Face(np.array([10, 10, 80, 90], np.float32), kps, 0.9, a if with_embedding else None),
                Face(np.array([110, 10, 180, 90], np.float32), kps + [100, 0], 0.9, b if with_embedding else None),
            ]

        def embed(self, image, face):
            return a if face.bbox[0] < 100 else b

    proc = FrameProcessor(TwoFaces(), solid_swapper, FrameOptions(color_strength=0))
    tracked = TrackedProcessor(proc, np.zeros(512), 24, TrackingOptions(select="reference"), reference_embedding=b)
    frame = np.zeros((120, 200, 3), np.uint8)
    for i in range(15):
        res = tracked.process(frame, i)
        assert len(res.faces) == 1 and res.faces[0].bbox[0] > 100
    assert res.matte[:, :95].max() == 0 and res.matte[:, 100:].max() > 0.9
    assert len(tracked.tracker.tracks) == 2


@pytest.mark.parametrize("fmt", ["h265", "prores_proxy", "prores4444xq", "dnxhr_hq", "dnxhr_hqx"])
@needs_ffmpeg
def test_more_codecs(tmp_path, fmt):
    from faceswapjp.media.presets import PRESETS, available_encoders

    if not any(e.name in available_encoders() for e in PRESETS[fmt].encoders):
        pytest.skip(f"no encoder for {fmt}")
    src = _make_clip(tmp_path / "src.mov", frames=3, size="640x360")
    info = probe(src)
    out = tmp_path / ("o" + get_preset(fmt).ext)
    with FrameWriter(out, info, get_preset(fmt), "bgr48le", quality="light", timecode=info.timecode,
                     audio_source=src) as w:
        for f in FrameReader(info):
            w.write(f)
    o = probe(out)
    assert o.nb_frames == 3 and o.timecode == "01:00:00:00" and o.has_audio
    assert o.codec == {"h265": "hevc", "dnxhr_hq": "dnxhd", "dnxhr_hqx": "dnxhd"}.get(fmt, "prores")
    if fmt == "h265":
        assert o.bit_depth == 10  # 10-bit source stays 10-bit


def test_choose_encoder_modes(monkeypatch):
    from faceswapjp.media import presets

    monkeypatch.setattr(presets, "available_encoders", lambda: frozenset({"libx264", "h264_videotoolbox"}))
    h264 = presets.get_preset("h264")
    assert presets.choose_encoder(h264).name == "libx264"
    assert presets.choose_encoder(h264, "hardware").name == "h264_videotoolbox"
    assert presets.choose_encoder(h264, "software").name == "libx264"
    with pytest.raises(presets.FFmpegError):
        presets.choose_encoder(presets.get_preset("h265"), "hardware")
    with pytest.raises(ValueError):
        presets.choose_encoder(h264, "prores_ks")


@needs_ffmpeg
@pytest.mark.parametrize("fmt", ["prores422hq", "h264"])
def test_resumable_render_continues_after_interruption(tmp_path, fake_analyzer, solid_swapper, monkeypatch, fmt):
    import faceswapjp.pipeline.video as video
    import threading

    monkeypatch.setattr(video, "SEGMENT_SECONDS", 0.25)  # 6 frames per segment at 23.976 fps
    src = _make_clip(tmp_path / "src.mov", frames=24)
    ext = ".mov" if fmt.startswith("prores") else ".mp4"
    out = tmp_path / f"out{ext}"
    settings = RenderSettings(format=fmt, matte="luma", start=2, end=22)  # 20 frames, partial range
    proc = FrameProcessor(fake_analyzer, solid_swapper)

    stop = threading.Event()
    def progress(done, total):
        if done == 15:
            stop.set()  # interrupt in the middle of the third segment
    with pytest.raises(InterruptedError):
        render_video(TrackedProcessor(proc, np.zeros(512, np.float32), 24.0), src, out, settings,
                     progress=progress, cancel=stop, resume_key="k1")
    parts = video.segment_dir(out)
    assert not out.exists() and (parts / "state.json").exists()
    assert sorted(p.name for p in parts.glob("v*.mov")) == ["v00000.mov", "v00001.mov"]  # 12 frames kept

    seen = []
    res = render_video(TrackedProcessor(proc, np.zeros(512, np.float32), 24.0), src, out, settings,
                       resume_key="k1", on_resume=lambda d, t: seen.append((d, t)))
    assert seen == [(12, 20)] and res.resumed_frames == 12 and res.frames == 20 and not res.warnings
    assert not parts.exists()
    info, ref = probe(out), probe(src)
    assert info.nb_frames == 20 and info.fps == ref.fps
    assert info.timecode == offset_timecode(ref.timecode, ref.fps, 2)
    assert info.has_audio and abs(info.duration - 20 / float(ref.fps)) < 0.06
    tags = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "csv=p=0",
                           str(out)], capture_output=True, text=True).stdout
    assert "AI face-swapped" in tags
    m = probe(res.matte_output)
    assert m.nb_frames == 20 and m.timecode == info.timecode


@needs_ffmpeg
def test_resumable_render_restarts_when_settings_change(tmp_path, fake_analyzer, solid_swapper, monkeypatch):
    import faceswapjp.pipeline.video as video

    monkeypatch.setattr(video, "SEGMENT_SECONDS", 0.25)
    src = _make_clip(tmp_path / "src.mov", frames=12, audio=False)
    out = tmp_path / "out.mov"
    parts = video.segment_dir(out)
    parts.mkdir()
    (parts / "state.json").write_text(json.dumps({"key": "old", "seg_frames": 6,
                                                  "segments": [{"frames": 6, "with_faces": 6}]}))
    (parts / "v00000.mov").write_bytes(b"stale")
    res = render_video(TrackedProcessor(FrameProcessor(fake_analyzer, solid_swapper), np.zeros(512, np.float32), 24.0),
                       src, out, RenderSettings(format="prores422hq"), resume_key="new")
    assert res.resumed_frames == 0 and probe(out).nb_frames == 12
