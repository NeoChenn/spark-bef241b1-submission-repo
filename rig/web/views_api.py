import base64
import binascii
import os

from fastapi import APIRouter, HTTPException, Response

KINDS = ("fpv", "policy", "bev")


class ViewStore:
    def __init__(self, inline_fpv: bool = os.environ.get("RIG_VIEWS_INLINE_FPV", "1") == "1"):
        self.inline_fpv = inline_fpv
        self.jpgs: dict[tuple[str, str], bytes] = {}
        self.stubs: dict[str, dict] = {}
        self.fpv: dict[str, dict] = {}

    def take(self, header: dict) -> dict:
        views = header.get("views")
        if isinstance(views, dict):
            for rid, kinds in views.items():
                if not isinstance(kinds, dict):
                    continue
                stub = self.stubs.setdefault(str(rid), {})
                for kind, v in kinds.items():
                    if isinstance(v, dict) and "jpg" in v:
                        try:
                            self.jpgs[(str(rid), kind)] = base64.b64decode(v["jpg"])
                        except (binascii.Error, ValueError):
                            continue
                        stub[kind] = {**{k: x for k, x in v.items() if k != "jpg"},
                                      "url": f"/api/views/{rid}/{kind}.jpg?s={v.get('seq', 0)}"}
                        if kind == "fpv":
                            self.fpv[str(rid)] = v
                    else:
                        stub[kind] = v
                if "boxes" in stub:
                    stub["policy"] = stub["boxes"]
        if not self.stubs:
            return header
        out = {**header, "views": self.stubs}
        if self.inline_fpv and self.fpv:
            out["fpv"] = {**(header.get("fpv") if isinstance(header.get("fpv"), dict) else {}), **self.fpv}
        return out

    def jpg(self, rid: str, kind: str) -> bytes | None:
        return self.jpgs.get((rid, kind))


def views_router(hub=None, store: ViewStore | None = None) -> APIRouter:
    store = store or ViewStore()
    if hub is not None:
        inner = hub.take_plume
        hub.take_plume = lambda header: store.take(inner(header))
        hub.views = store
    r = APIRouter()

    @r.get("/api/views")
    def index() -> dict:
        return store.stubs

    @r.get("/api/views/{rid}/{kind}.jpg")
    def image(rid: str, kind: str) -> Response:
        data = store.jpg(rid, kind)
        if data is None:
            raise HTTPException(404, f"no {kind} view for {rid}")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return r
