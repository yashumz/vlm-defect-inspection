"""API tests with a fake model, so they run on a laptop without a GPU."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from defect_vlm.api import app, get_inspector
from defect_vlm.inference import DEFECT_TYPES, make_prompt, parse_json


class FakeInspector:
    def __init__(self):
        self.calls = []

    def inspect(self, image, category):
        self.calls.append((image.size, category))
        return {"defective": True, "defect_type": "scratch_head", "location": "top",
                "score": 0.93, "valid_json": True, "raw": "{}"}


@pytest.fixture
def client():
    fake = FakeInspector()
    app.dependency_overrides[get_inspector] = lambda: fake
    yield TestClient(app), fake
    app.dependency_overrides.clear()


def png_bytes(size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, "gray").save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok"}


def test_inspect_returns_structured_answer(client):
    c, fake = client
    r = c.post("/inspect", data={"category": "screw"},
               files={"image": ("part.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["defective"] is True and body["defect_type"] == "scratch_head"
    assert 0 <= body["score"] <= 1 and body["latency_ms"] >= 0
    assert fake.calls == [((64, 64), "screw")]


def test_unknown_category_rejected(client):
    c, fake = client
    r = c.post("/inspect", data={"category": "banana"},
               files={"image": ("part.png", png_bytes(), "image/png")})
    assert r.status_code == 400 and "Unknown category" in r.json()["detail"]
    assert fake.calls == []


def test_non_image_rejected(client):
    c, _ = client
    r = c.post("/inspect", data={"category": "screw"},
               files={"image": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_taxonomy_matches_mvtec():
    assert len(DEFECT_TYPES) == 15
    assert sum(len(v) for v in DEFECT_TYPES.values()) == 73


def test_prompt_matches_training_format():
    p = make_prompt("metal_nut")
    assert p.startswith("You are inspecting a metal nut on a factory line.")
    assert "Possible defect types: bent, color, flip, scratch." in p


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"defective": false}\n```') == {"defective": False}
    assert parse_json("no json") is None


def test_unconfigured_server_gives_clear_503(monkeypatch):
    from defect_vlm import api
    app.dependency_overrides.clear()
    monkeypatch.delenv("ADAPTER_DIR", raising=False)
    monkeypatch.setattr(api, "_inspector", None)
    r = TestClient(app).post("/inspect", data={"category": "screw"},
                             files={"image": ("part.png", png_bytes(), "image/png")})
    assert r.status_code == 503 and "ADAPTER_DIR" in r.json()["detail"]
