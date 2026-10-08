"""Plain-language messages for errors that reach the user (shared by the UIs)."""

from __future__ import annotations

from .identity import ConsentError
from .media.probe import FFmpegError
from .models import registry
from .safety.nsfw import NSFWContentError


CONSENT_MESSAGES = {
    "person name is required": "氏名を入力してください。",
    "consent date must be": "日付は「年-月-日」の形式で入力してください。",
    "in the future": "日付が未来になっています。",
    "right to use this face": "「この顔を使う権利があることを確認しました」にチェックを入れてください。",
    "consent document not found": "同意書のファイルが見つかりません。",
    "source type must be": "誰の顔かを選んでください。",
}


def friendly(exc: Exception) -> str:
    """Turn internal errors into something a beginner can act on."""
    if isinstance(exc, NSFWContentError):
        return "この素材は処理できません。性的な内容が含まれている可能性があるため、処理を止めました。"
    if isinstance(exc, ConsentError):
        text = str(exc)
        for key, ja in CONSENT_MESSAGES.items():
            if key in text:
                return ja
        return f"同意の情報に不足があります：{text}"
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
