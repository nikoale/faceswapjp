import numpy as np
import pytest

from faceswapjp.imageio import read_image
from faceswapjp.pipeline.frame import FrameOptions, FrameProcessor, select_faces
from faceswapjp.pipeline.image import swap_image
from faceswapjp.safety.provenance import read_jpeg_comments, read_png_text
from faceswapjp.analysis.face import Face


def test_frame_processor_replaces_face_region(fake_analyzer, solid_swapper):
    frame = np.full((200, 220, 3), 128, np.uint8)
    proc = FrameProcessor(fake_analyzer, solid_swapper, FrameOptions(color_strength=0))
    res = proc.process(frame, np.zeros(512, np.float32))
    assert len(res.faces) == 1
    assert res.frame[80, 100, 2] > 250 and res.frame[80, 100, 0] < 5  # solid red at face center
    assert (res.frame[0, 0] == 128).all()  # far away pixel untouched
    assert (frame == 128).all(), "input frame must not be modified"
    assert res.matte[80, 100] > 0.99 and res.matte[0, 0] == 0


def test_select_faces_largest_and_threshold():
    def face(size, score):
        return Face(np.array([0, 0, size, size], np.float32), np.zeros((5, 2), np.float32), score)

    faces = [face(10, 0.9), face(50, 0.9), face(80, 0.3)]
    assert select_faces(faces, "largest", 0.5)[0].area == 2500
    assert len(select_faces(faces, "all", 0.5)) == 2
    with pytest.raises(ValueError):
        select_faces(faces, "bogus", 0.5)


@pytest.mark.parametrize("ext", [".png", ".jpg"])
def test_swap_image_writes_metadata(tmp_path, fake_analyzer, solid_swapper, ext):
    import cv2

    target = tmp_path / "target.png"
    cv2.imwrite(str(target), np.full((200, 220, 3), 100, np.uint8))
    out = tmp_path / f"out{ext}"
    proc = FrameProcessor(fake_analyzer, solid_swapper)
    swap_image(proc, np.zeros(512, np.float32), target, out, provenance={"identity": "x"}, matte_output=tmp_path / "m.png")
    data = out.read_bytes()
    if ext == ".png":
        assert read_png_text(data)["Comment"] == "AI face-swapped"
        assert '"identity": "x"' in read_png_text(data)["faceswapjp"]
    else:
        assert read_jpeg_comments(data)[0].startswith("AI face-swapped")
    matte = cv2.imread(str(tmp_path / "m.png"), cv2.IMREAD_UNCHANGED)
    assert matte.dtype == np.uint16 and matte.max() == 65535


def test_swap_image_keeps_16bit_and_unicode_path(tmp_path, fake_analyzer, solid_swapper):
    import cv2

    target = tmp_path / "素材.png"
    ok, buf = cv2.imencode(".png", np.full((200, 220, 3), 30000, np.uint16))
    target.write_bytes(buf.tobytes())
    out = tmp_path / "出力.png"
    swap_image(FrameProcessor(fake_analyzer, solid_swapper), np.zeros(512, np.float32), target, out)
    img = read_image(out)
    assert img.dtype == np.uint16
    assert img[0, 0, 0] == 30000


def test_swap_image_without_face_fails(tmp_path, solid_swapper):
    import cv2

    class NoFaces:
        def detect(self, image, with_embedding=True):
            return []

    target = tmp_path / "t.png"
    cv2.imwrite(str(target), np.zeros((50, 50, 3), np.uint8))
    with pytest.raises(ValueError, match="no target face"):
        swap_image(FrameProcessor(NoFaces(), solid_swapper), np.zeros(512), target, tmp_path / "o.png")


def test_pixel_boost_tiles_at_model_size_and_reassembles():
    from faceswapjp.pipeline.frame import pixel_boost

    rng = np.random.default_rng(0)
    crop = rng.random((256, 256, 3), dtype=np.float32)
    seen = []

    def swap(tile):
        seen.append(tile.shape)
        return tile

    out = pixel_boost(swap, crop, 128)
    assert seen == [(128, 128, 3)] * 4
    np.testing.assert_array_equal(out, crop)
    assert pixel_boost(lambda t: t * 0 + 0.5, crop[:128, :128], 128).mean() == 0.5
