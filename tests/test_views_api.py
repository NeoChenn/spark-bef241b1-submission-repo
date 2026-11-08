import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rig.web.views_api import ViewStore, views_router

JPG = b"\xff\xd8views"


def inset(seq, cap="cam"):
    return {"jpg": base64.b64encode(JPG + bytes([seq])).decode(), "w": 256, "h": 192, "seq": seq, "caption": cap}


def header(seq):
    return {"type": "frame", "views": {"R1": {"fpv": inset(seq), "policy": inset(seq, "rust"), "info": {"role": "rust"}}}}


def test_store_strips_jpgs_and_keeps_sticky_stubs():
    s = ViewStore()
    h = s.take(header(1))
    stub = h["views"]["R1"]["fpv"]
    assert "jpg" not in stub and stub["url"] == "/api/views/R1/fpv.jpg?s=1" and stub["seq"] == 1
    assert h["views"]["R1"]["info"] == {"role": "rust"}
    assert h["fpv"]["R1"]["jpg"] == inset(1)["jpg"]
    later = s.take({"type": "frame"})
    assert later["views"]["R1"]["policy"]["seq"] == 1
    assert "fpv" in later
    assert s.jpg("R1", "policy") == JPG + bytes([1])


def test_store_without_inline_fpv():
    s = ViewStore(inline_fpv=False)
    h = s.take(header(2))
    assert "fpv" not in h


class Hub:
    def take_plume(self, header):
        return {**header, "plumed": True}


def test_router_wraps_hub_and_serves_jpgs():
    hub = Hub()
    app = FastAPI()
    app.include_router(views_router(hub))
    out = hub.take_plume(header(4))
    assert out["plumed"] and "jpg" not in out["views"]["R1"]["fpv"]
    c = TestClient(app)
    r = c.get("/api/views/R1/policy.jpg")
    assert r.status_code == 200 and r.content == JPG + bytes([4]) and r.headers["content-type"] == "image/jpeg"
    assert c.get("/api/views/R9/fpv.jpg").status_code == 404
    assert c.get("/api/views").json()["R1"]["fpv"]["seq"] == 4
