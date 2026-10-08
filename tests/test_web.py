"""Studio web API with a fake engine (no models needed)."""

import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from faceswapjp import jobs  # noqa: E402
from faceswapjp.engine import Engine  # noqa: E402
from faceswapjp.safety.nsfw import SafetyGate  # noqa: E402
from faceswapjp.web.server import create_app  # noqa: E402


class FakeClassifier:
    value = 0.01

    def score(self, image):
        return self.value


@pytest.fixture
def client(tmp_path, monkeypatch, fake_analyzer, solid_swapper):
    eng = Engine(fake_analyzer, solid_swapper, ["CPUExecutionProvider"], "buffalo_l", "inswapper_128",
                 SafetyGate(FakeClassifier()), model_names=["buffalo_l", "inswapper_128"])
    monkeypatch.setattr(jobs, "get_engine", lambda *a, **k: eng)
    app = create_app(tmp_path / "projects", device="cpu", token="tok")
    c = TestClient(app)
    c.headers["X-FSJ-Token"] = "tok"
    c.root = tmp_path
    return c


def _png(path, value=100):
    import cv2

    cv2.imwrite(str(path), np.full((200, 220, 3), value, np.uint8))
    return path


def test_requires_token_and_local_host(client):
    assert client.get("/api/state", headers={"X-FSJ-Token": "wrong"}).status_code == 403
    assert client.get("/api/state?t=tok", headers={"X-FSJ-Token": ""}).status_code == 200
    assert client.get("/api/state", headers={"host": "evil.example"}).status_code == 403
    page = client.get("/")
    assert page.status_code == 200 and 'content="tok"' in page.text


def test_state_lists_formats_and_default_project(client):
    s = client.get("/api/state").json()
    assert s["projects"] == ["はじめての作品"]
    assert {"h264", "prores422hq", "dnxhr_hq"} <= {f["name"] for f in s["formats"]}


def test_full_flow_on_a_still(client):
    project = "はじめての作品"
    ref = _png(client.root / "ref.png")
    r = client.post(f"/api/projects/{project}/identities",
                    files=[("images", ("ref.png", ref.read_bytes(), "image/png"))],
                    data={"label": "Taro", "person": "Taro", "consent_date": "2026-01-01",
                          "source_type": "self", "attested": "false"})
    assert r.status_code == 400 and "チェック" in r.json()["error"]
    r = client.post(f"/api/projects/{project}/identities",
                    files=[("images", ("ref.png", ref.read_bytes(), "image/png"))],
                    data={"label": "Taro", "person": "Taro", "consent_date": "2026-01-01",
                          "source_type": "self", "attested": "true"})
    assert r.status_code == 200, r.text
    iid = r.json()["id"]
    ids = client.get(f"/api/projects/{project}/identities").json()
    assert ids[0]["id"] == iid
    assert client.get(ids[0]["thumb"]).headers["content-type"] == "image/jpeg"

    target = client.post("/api/targets/path", json={"path": str(_png(client.root / "shot.png"))}).json()
    assert target["kind"] == "image" and target["width"] == 220
    assert client.get(f"/api/targets/{target['id']}/frame?n=0&w=100").headers["content-type"] == "image/jpeg"
    faces = client.post(f"/api/targets/{target['id']}/faces", json={"frame": 0, "project": project}).json()["faces"]
    assert len(faces) == 1 and len(faces[0]["emb"]) == 512

    body = {"project": project, "identity": iid, "target": target["id"], "frame": 0, "who": "pick",
            "picked": [faces[0]["emb"]], "codec": "h264"}
    prev = client.post("/api/preview", json=body).json()
    assert prev["faces"] == 1 and client.get(prev["after"]).status_code == 200

    job = client.post("/api/render", json=body).json()
    for _ in range(100):
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert state["status"] == "done", state
    out = state["result"]["output"]
    assert client.get("/api/file", params={"path": out}).status_code == 200
    assert client.get(f"/api/projects/{project}/log").json()["entries"][0]["event"] == "render"


def test_validation_messages(client):
    body = {"project": "はじめての作品", "identity": "x", "target": "nope", "who": "pick", "picked": []}
    assert "素材" in client.post("/api/preview", json=body).json()["error"]
    t = client.post("/api/targets/path", json={"path": str(_png(client.root / "s.png"))}).json()
    body["target"] = t["id"]
    assert "置き換える人" in client.post("/api/preview", json=body).json()["error"]


def test_file_access_is_limited_to_projects(client):
    outside = _png(client.root / "secret.png")
    assert client.get("/api/file", params={"path": str(outside)}).status_code == 403
    assert client.post("/api/projects", json={"name": "../x"}).status_code == 400
