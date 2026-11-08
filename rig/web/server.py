import asyncio
import contextlib
import gzip
import hashlib
import json
import os
from collections import deque
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from rig.frames import encode, read_message_async
from rig.hud import check_event

STATIC = Path(__file__).parent / "static"
REPO = Path(__file__).resolve().parents[2]
MOCK_PATH = REPO / "runs/hud/mock_session.json"
FLEET_DIRS = (REPO / "runs/fleet", REPO / "runs_sample/fleet")
FLEET_SCHEMA = "fleet_state.v1"
SHOTS = {1: "sim/output/cine_storm_aerial.png", 2: "sim/output/preview_hero.png",
         3: "sim/output/preview_repair_closeup.png"}
LOG_KEEP = 200
REPLAY_EVENTS = 20
FPS_WINDOW_S = 2.0
STALE_AFTER_S = 2.0
BLACK_MEAN = 3.0
HEADER_FILES = ("headers.jsonl", "header.jsonl", "hud.jsonl", "frames.jsonl", "live_hud.jsonl", "live_hud_1hz.jsonl")
TIMELINE_FILES = ("timeline.jsonl", "live_timeline.jsonl")
VIDEO_DIRS = ("runs/fleet_isaac", "runs_sample/fleet_isaac")
SPAN_TOL_S = 1.0
TIMELINE_TYPES = ("finding", "argon_request", "argon_call", "command", "ticket", "vetoed", "object_report", "nav_event", "scenario")


def fleet_session(p: Path) -> dict | None:
    from rig.replay import load_timeline
    d = p if p.is_dir() else p.parent
    hp = p if p.is_file() else next((d / n for n in HEADER_FILES if (d / n).is_file()), None)
    if hp is None:
        return None
    rows = load_timeline(hp)
    heads = [h for h in rows if "event" not in h and h.get("type") in (None, "frame", "header")]
    if not any("robots" in h or "argon" in h for h in heads):
        return None
    live = hp.name.startswith("live_")
    tl = next((d / n for n in sorted(TIMELINE_FILES, key=lambda n: n.startswith("live_") != live) if (d / n).is_file()), None)
    extra = load_timeline(tl) if tl is not None and tl != hp else []
    marks = sorted((r for r in rows + extra if r.get("type") in TIMELINE_TYPES), key=lambda r: float(r["t"]))
    heads.sort(key=lambda h: float(h["t"]))
    t0, k = float(heads[0]["t"]), 0
    for h in heads:
        h.setdefault("pt", round(float(h["t"]) - t0, 3))
        n = k
        while k < len(marks) and float(marks[k]["t"]) <= float(h["t"]):
            k += 1
        if k > n:
            h["argon_rows"] = marks[n:k]
    return {"snapshots": heads, "duration": heads[-1]["pt"], "replay": True, "fleet_demo": True, "timeline_file": str(tl) if tl else None}


def _has_moov(f: Path) -> bool:
    with f.open("rb") as fh:
        if b"moov" in fh.read(1 << 16):
            return True
        fh.seek(max(0, f.stat().st_size - (1 << 20)))
        return b"moov" in fh.read()


def video_info(root: Path, mp4: Path) -> dict | None:
    hdr = mp4.parent / "fleet_isaac_header.jsonl"
    if not mp4.is_file() or not _has_moov(mp4):
        return None
    t0 = t1 = 0.0
    if hdr.is_file():
        with hdr.open("rb") as fh:
            t0 = float(json.loads(fh.readline())["t"])
            fh.seek(max(0, hdr.stat().st_size - (1 << 16)))
            t1 = float(json.loads(fh.read().splitlines()[-1])["t"])
    return {"url": "/api/video?path=" + str(mp4.relative_to(root)), "t0": t0, "t1": t1}


def _sha256(f: Path) -> str:
    return hashlib.sha256(f.read_bytes()).hexdigest()


def find_video(root: Path, timeline: str | None, t_end: float | None = None) -> dict | None:
    found, sha = [], None
    for vd in VIDEO_DIRS:
        for mp4 in (root / vd).glob("*/fleet_isaac.mp4"):
            d = mp4.parent
            if not (d / "fleet_isaac_header.jsonl").is_file():
                continue
            meta = d / "fleet_isaac.json"
            m = json.loads(meta.read_text()) if meta.is_file() else {}
            if timeline and m.get("timeline") and m["timeline"] != timeline:
                continue
            found.append((mp4, m.get("timeline_sha256")))
    for mp4, want in sorted(found, key=lambda f: f[0].stat().st_mtime, reverse=True):
        if want:
            if timeline is None:
                continue
            sha = sha or _sha256(root / timeline)
            if want != sha:
                continue
        info = video_info(root, mp4)
        if not info:
            continue
        if not want and (t_end is None or abs(info["t1"] - t_end) > SPAN_TOL_S):
            continue
        return info
    return None


class FrameHub:
    def __init__(self) -> None:
        self.latest: tuple[int, bytes, dict] | None = None
        self.frames = 0
        self.last_frame_at: float | None = None
        self.sim_connected = False
        self._seq = 0
        self._event = asyncio.Event()
        self._times: list[float] = []
        self.log: list[dict] = []
        self.log_base = 0
        self.fleet: dict | None = None

    @property
    def fps(self) -> float:
        now = time.monotonic()
        recent = [t for t in self._times if now - t <= FPS_WINDOW_S]
        return len(recent) / FPS_WINDOW_S

    def publish(self, jpeg: bytes, header: dict) -> None:
        now = time.monotonic()
        self._seq += 1
        self.frames += 1
        self.last_frame_at = now
        self.latest = (self._seq, jpeg, header)
        self.log.extend(header.get("events") or [])
        if isinstance(header.get("fleet"), dict):
            self.fleet = header["fleet"]
        if len(self.log) > LOG_KEEP:
            self.log_base += len(self.log) - LOG_KEEP
            self.log = self.log[-LOG_KEEP:]
        self._times = [t for t in self._times if now - t <= FPS_WINDOW_S]
        self._times.append(now)
        event, self._event = self._event, asyncio.Event()
        event.set()

    async def wait_newer(self, seq: int) -> tuple[int, bytes, dict]:
        while self.latest is None or self.latest[0] <= seq:
            await self._event.wait()
        return self.latest

    def events_since(self, idx: int) -> tuple[int, list[dict]]:
        end = self.log_base + len(self.log)
        return end, self.log[max(0, idx - self.log_base):]


def create_app(sim_host: str = "127.0.0.1", sim_port: int = 9000, mock_path: Path = MOCK_PATH,
               fleet_dirs=FLEET_DIRS, root: Path = REPO) -> FastAPI:
    root = Path(root).resolve()
    hub = FrameHub()
    current: dict[str, asyncio.StreamWriter | None] = {"writer": None}
    pending: deque = deque()

    def flush() -> None:
        writer = current["writer"]
        while writer is not None and pending:
            writer.write(encode({"type": "event", **pending.popleft()}))

    async def handle_sim(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        previous = current["writer"]
        current["writer"] = writer
        if previous is not None:
            previous.close()
        hub.sim_connected = True
        flush()
        try:
            while True:
                header, payload = await read_message_async(reader)
                if header.get("type") == "frame":
                    hub.publish(payload, header)
        except (ConnectionError, OSError):
            pass
        finally:
            if current["writer"] is writer:
                current["writer"] = None
                hub.sim_connected = False
            writer.close()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        server = await asyncio.start_server(handle_sim, sim_host, sim_port)
        app.state.sim_port = server.sockets[0].getsockname()[1]
        try:
            yield
        finally:
            server.close()
            if current["writer"] is not None:
                current["writer"].close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(server.wait_closed(), 1)

    app = FastAPI(lifespan=lifespan)
    app.state.hub = hub
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    async def health() -> dict:
        header = hub.latest[2] if hub.latest else {}
        age = None if hub.last_frame_at is None else time.monotonic() - hub.last_frame_at
        mean = header.get("mean")
        if not hub.sim_connected:
            state = "no_sim"
        elif age is None or age > STALE_AFTER_S:
            state = "stalled"
        elif mean is not None and mean < BLACK_MEAN:
            state = "warming"
        else:
            state = "live"
        return {
            "state": state,
            "sim_connected": hub.sim_connected,
            "frames": hub.frames,
            "fps": round(hub.fps, 1),
            "frame_age_s": None if age is None else round(age, 2),
            "w": header.get("w"),
            "h": header.get("h"),
            "events_pending": len(pending),
        }

    @app.post("/api/event")
    async def post_event(body: dict) -> dict:
        try:
            pending.append(check_event(body))
        except ValueError as e:
            raise HTTPException(400, str(e))
        queued = len(pending)
        flush()
        if current["writer"] is not None:
            await current["writer"].drain()
        return {"queued": queued, "delivered": not pending}

    def fleet_file(name: str) -> Path:
        for d in fleet_dirs:
            if (Path(d) / name).exists():
                return Path(d) / name
        raise HTTPException(404, f"no {name}")

    @app.get("/api/fleet")
    async def fleet() -> Response:
        if hub.fleet is not None:
            return JSONResponse(hub.fleet)
        return FileResponse(fleet_file("fleet_state.json"), media_type="application/json")

    @app.post("/api/fleet")
    async def post_fleet(body: dict) -> dict:
        if body.get("schema") != FLEET_SCHEMA:
            raise HTTPException(400, f"expected schema {FLEET_SCHEMA}")
        hub.fleet = body
        return {"t": body.get("t")}

    @app.get("/api/fleet/map")
    async def fleet_map() -> FileResponse:
        return FileResponse(fleet_file("fleet_map.json"), media_type="application/json")

    @app.get("/api/replay")
    async def replay(path: str, request: Request, video: str | None = None) -> Response:
        from rig.replay import load_timeline, timeline_session
        p = (root / path).resolve()
        if not any(p.is_relative_to(root / d) for d in ("runs", "runs_sample")):
            raise HTTPException(403, "replay files must live under runs/ or runs_sample/")
        if not p.exists():
            raise HTTPException(404, f"no {path}")
        fleet = await asyncio.to_thread(fleet_session, p)
        if fleet is not None:
            tf = fleet.pop("timeline_file", None)
            rel = str(Path(tf).relative_to(root)) if tf and Path(tf).is_relative_to(root) else None
            v = inside(video) if video else None
            t_end = float(fleet["snapshots"][0]["t"]) + float(fleet["duration"]) if fleet["snapshots"] else None
            fleet["video"] = await asyncio.to_thread(lambda: video_info(root, v) if v else find_video(root, rel, t_end))
            body = await asyncio.to_thread(lambda: json.dumps(fleet, separators=(",", ":")).encode())
            if "gzip" not in request.headers.get("accept-encoding", ""):
                return Response(body, media_type="application/json")
            body = await asyncio.to_thread(gzip.compress, body, 5)
            return Response(body, media_type="application/json", headers={"Content-Encoding": "gzip"})
        if not p.is_file():
            raise HTTPException(404, f"no replay in {path}")
        return await asyncio.to_thread(lambda: timeline_session(load_timeline(p)))

    def inside(rel: str) -> Path:
        v = (root / rel).resolve()
        if not any(v.is_relative_to(root / d) for d in ("runs", "runs_sample")):
            raise HTTPException(403, "videos must live under runs/ or runs_sample/")
        return v

    @app.get("/api/video")
    async def video_file(path: str) -> FileResponse:
        v = inside(path)
        if v.suffix != ".mp4" or not v.is_file():
            raise HTTPException(404, f"no {path}")
        return FileResponse(v, media_type="video/mp4")

    @app.get("/api/bench")
    async def bench() -> list:
        out = {}
        for d in ("runs_sample/bench", "runs/bench"):
            for f in sorted((root / d).glob("*.json")):
                try:
                    body = json.loads(f.read_text())
                except ValueError:
                    continue
                if isinstance(body, dict):
                    out[f.stem] = {"name": f.stem, **body}
        return list(out.values())

    @app.get("/api/mock")
    async def mock() -> Response:
        if not mock_path.exists():
            from rig.hud import record_mock
            await asyncio.to_thread(record_mock, mock_path, str(mock_path.parent / "decisions.jsonl"))
        return FileResponse(mock_path, media_type="application/json")

    @app.get("/api/mock/shot/{n}")
    async def mock_shot(n: int) -> FileResponse:
        if n not in SHOTS:
            raise HTTPException(404, "no such shot")
        return FileResponse(REPO / SHOTS[n], media_type="image/png")

    @app.get("/api/frame.jpg")
    async def frame_jpg() -> Response:
        if hub.latest is None:
            return JSONResponse({"detail": "no frames yet"}, status_code=503)
        return Response(hub.latest[1], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.websocket("/ws/video")
    async def ws_video(ws: WebSocket) -> None:
        await ws.accept()
        seq = 0
        try:
            while True:
                seq, jpeg, _ = await hub.wait_newer(seq)
                await ws.send_bytes(jpeg)
        except (WebSocketDisconnect, RuntimeError):
            pass

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket) -> None:
        await ws.accept()
        seq = 0
        idx = max(0, hub.log_base + len(hub.log) - REPLAY_EVENTS)
        try:
            while True:
                seq, _, header = await hub.wait_newer(seq)
                idx, events = hub.events_since(idx)
                await ws.send_json({**header, "events": events})
                await asyncio.sleep(0.08)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


app = create_app(os.environ.get("SIM_HOST", "127.0.0.1"), int(os.environ.get("SIM_PORT", "9000")))
