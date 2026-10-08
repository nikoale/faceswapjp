"""Write "AI face-swapped" provenance metadata into output files."""

from __future__ import annotations

import json
import struct
import zlib
from datetime import datetime, timezone

from .. import __version__

AI_TAG = "AI face-swapped"


def provenance_record(**extra: object) -> dict[str, object]:
    return {
        "ai_modified": True,
        "description": AI_TAG,
        "tool": f"faceswapjp {__version__}",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def add_png_text(png: bytes, text: dict[str, str]) -> bytes:
    """Insert iTXt chunks (UTF-8) right after IHDR."""
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    ihdr_end = 8 + 12 + struct.unpack(">I", png[8:12])[0]
    chunks = b"".join(
        _png_chunk(b"iTXt", k.encode("latin-1") + b"\x00\x00\x00\x00\x00" + v.encode("utf-8")) for k, v in text.items()
    )
    return png[:ihdr_end] + chunks + png[ihdr_end:]


def read_png_text(png: bytes) -> dict[str, str]:
    out, pos = {}, 8
    while pos < len(png):
        length = struct.unpack(">I", png[pos : pos + 4])[0]
        kind, data = png[pos + 4 : pos + 8], png[pos + 8 : pos + 8 + length]
        if kind == b"iTXt":
            key, rest = data.split(b"\x00", 1)
            # skip compression flag, method, language tag, translated keyword
            rest = rest[2:]
            _, rest = rest.split(b"\x00", 1)
            _, value = rest.split(b"\x00", 1)
            out[key.decode("latin-1")] = value.decode("utf-8")
        elif kind == b"tEXt":
            key, value = data.split(b"\x00", 1)
            out[key.decode("latin-1")] = value.decode("latin-1")
        pos += 12 + length
    return out


def add_jpeg_comment(jpeg: bytes, comment: str) -> bytes:
    """Insert a COM segment right after SOI."""
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    payload = comment.encode("utf-8")[:65533]
    return jpeg[:2] + b"\xff\xfe" + struct.pack(">H", len(payload) + 2) + payload + jpeg[2:]


def read_jpeg_comments(jpeg: bytes) -> list[str]:
    out, pos = [], 2
    while pos + 4 <= len(jpeg) and jpeg[pos] == 0xFF:
        marker = jpeg[pos + 1]
        if marker == 0xDA:  # start of scan
            break
        length = struct.unpack(">H", jpeg[pos + 2 : pos + 4])[0]
        if marker == 0xFE:
            out.append(jpeg[pos + 4 : pos + 2 + length].decode("utf-8", "replace"))
        pos += 2 + length
    return out


def tag_image_bytes(data: bytes, ext: str, record: dict[str, object]) -> bytes:
    ext = ext.lower()
    if ext == ".png":
        return add_png_text(
            data,
            {"Comment": AI_TAG, "Description": AI_TAG, "faceswapjp": json.dumps(record, ensure_ascii=False)},
        )
    if ext in (".jpg", ".jpeg"):
        return add_jpeg_comment(data, f"{AI_TAG} {json.dumps(record, ensure_ascii=False)}")
    raise ValueError(f"metadata tagging is not supported for {ext}; use .png or .jpg")
