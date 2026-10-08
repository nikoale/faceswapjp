import json
from datetime import date, timedelta

import numpy as np
import pytest

from faceswapjp.identity import ConsentError, list_identities, load_identity, make_id, register_identity, validate_consent
from faceswapjp.project import Project
from faceswapjp.safety.consent import ConsentLog
from faceswapjp.safety.provenance import add_png_text, read_png_text


def test_consent_log_chain_detects_tampering(tmp_path):
    log = ConsentLog(tmp_path / "log.json")
    log.append("a", x=1)
    log.append("b", x=2)
    assert log.verify() == []
    data = json.loads(log.path.read_text())
    data["entries"][0]["data"]["x"] = 99
    log.path.write_text(json.dumps(data))
    assert any("modified" in p for p in log.verify())
    with pytest.raises(RuntimeError):
        log.append("c")


def test_consent_log_detects_deleted_entry(tmp_path):
    log = ConsentLog(tmp_path / "log.json")
    for i in range(3):
        log.append("e", i=i)
    data = json.loads(log.path.read_text())
    del data["entries"][1]
    log.path.write_text(json.dumps(data))
    assert log.verify()


@pytest.fixture
def consent_doc(tmp_path):
    p = tmp_path / "consent.pdf"
    p.write_bytes(b"%PDF signed")
    return p


def test_validate_consent_requires_everything(consent_doc):
    ok = validate_consent("山田 太郎", "2026-01-15", consent_doc, "consented_person")
    assert ok.person_name == "山田 太郎" and len(ok.consent_document_sha256) == 64
    with pytest.raises(ConsentError):
        validate_consent(" ", "2026-01-15", consent_doc, "self")
    with pytest.raises(ConsentError):
        validate_consent("A", "15/01/2026", consent_doc, "self")
    with pytest.raises(ConsentError):
        validate_consent("A", (date.today() + timedelta(days=1)).isoformat(), consent_doc, "self")
    with pytest.raises(ConsentError):
        validate_consent("A", "2026-01-15", consent_doc.parent / "missing.pdf", "self")
    with pytest.raises(ConsentError):
        validate_consent("A", "2026-01-15", consent_doc, "celebrity")


def test_make_id():
    assert make_id("Stunt Double A", set()) == "stunt-double-a"
    assert make_id("Stunt Double A", {"stunt-double-a"}) == "stunt-double-a-2"
    assert make_id("山田太郎", set()).startswith("id-")


def test_register_identity_writes_files_and_log(tmp_path, consent_doc, fake_analyzer):
    import cv2

    project = Project.init(tmp_path / "proj")
    ref = tmp_path / "ref.png"
    cv2.imwrite(str(ref), np.zeros((200, 220, 3), np.uint8))
    consent = validate_consent("Taro", "2026-01-15", consent_doc, "self")
    ident = register_identity(project, fake_analyzer, "Taro", [ref, ref], consent, "buffalo_l")
    loaded, emb = load_identity(project, ident.id)
    assert loaded.consent.person_name == "Taro"
    assert np.isclose(np.linalg.norm(emb), 1.0)
    assert [i.id for i in list_identities(project)] == [ident.id]
    events = [e["event"] for e in project.consent_log.entries()]
    assert events == ["project_created", "identity_registered"]
    assert project.consent_log.verify() == []


def test_png_text_roundtrip():
    import cv2

    ok, buf = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))
    tagged = add_png_text(buf.tobytes(), {"Comment": "AI face-swapped", "x": "日本語"})
    assert read_png_text(tagged) == {"Comment": "AI face-swapped", "x": "日本語"}
    assert cv2.imdecode(np.frombuffer(tagged, np.uint8), cv2.IMREAD_UNCHANGED).shape == (4, 4, 3)
