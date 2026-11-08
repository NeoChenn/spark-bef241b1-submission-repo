import asyncio
import base64
import binascii
import contextlib
import json
import os
from collections import deque
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from rig.frames import encode, read_message_async
from rig.hud import check_event, is_fleet_event
from rig.plume import Field

STATIC = Path(__file__).parent / "static"
REPO = Path(__file__).resolve().parents[2]
MOCK_PATH = REPO / "runs/hud/mock_session.json"
FLEET_DIRS = (REPO / "runs/fleet", REPO / "runs_sample/fleet")
FLEET_SCHEMA = "fleet_state.v1"
MOCK_SAMPLE = REPO / "runs_sample/hud/mock_session.json"
SHOTS = {n: f"runs_sample/hud/shot{n}.jpg" for n in (1, 2, 3)}
LOG_KEEP = 200
REPLAY_EVENTS = 20
FPS_WINDOW_S = 2.0
STALE_AFTER_S = 2.0
BLACK_MEAN = 3.0
FLEET_TICK_S = 0.25
FLEET_SIM_EVERY = 2


def plume_caption(p: dict) -> str:
    s, f, w = p.get("stats") or {}, p.get("flow") or {}, p.get("wind") or {}
    return (f"{p.get('flange', 'Leak')} · {f.get('kg_s', 0) * 1000:.1f} g/s ({f.get('sm3_h', 0):.0f} Sm³/h) · "
            f"cloud >10% LEL {s.get('vol_10lel_m3', 0):.1f} m³ · peak {s.get('max_lel_pct', 0):.0f}% LEL · "
            f"wind {w.get('speed_m_s', 0):.0f} m/s")


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
        self.plume: dict | None = None
        self.plume_jpg: bytes | None = None
        self.plume_field: Field | None = None
        self.plume_npz: bytes | None = None
        self.pip: dict | None = None
        self.fleet: dict | None = None

    @property
    def fps(self) -> float:
        now = time.monotonic()
        recent = [t for t in self._times if now - t <= FPS_WINDOW_S]
        return len(recent) / FPS_WINDOW_S

    def take_plume(self, header: dict) -> dict:
        p = header.get("plume")
        if not isinstance(p, dict):
            return header
        p = dict(p)
        jpg, field = p.pop("jpg", None), p.pop("field", None)
        seq = (self.plume or {}).get("seq", 0)
        try:
            if jpg:
                self.plume_jpg = base64.b64decode(jpg)
                seq += 1
                self.pip = {"jpg": jpg, "w": p.get("w"), "h": p.get("h"), "caption": plume_caption(p), "seq": seq}
            if field:
                self.plume_npz = base64.b64decode(field)
                self.plume_field = Field.from_bytes(self.plume_npz)
        except (binascii.Error, ValueError, OSError, KeyError):
            pass
        self.plume = {**(self.plume or {}), **p, "seq": seq, "image": self.plume_jpg is not None,
                      "field": self.plume_field is not None}
        return {**header, "plume": self.plume}

    def publish(self, jpeg: bytes, header: dict) -> None:
        header = self.take_plume(header)
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
               fleet_dirs=FLEET_DIRS, root: Path = REPO, fleet=None) -> FastAPI:
    root = Path(root).resolve()
    hub = FrameHub()
    fleet_seq = {"n": 0, "event": asyncio.Event()}
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

    def fleet_changed() -> None:
        fleet_seq["n"] += 1
        event, fleet_seq["event"] = fleet_seq["event"], asyncio.Event()
        event.set()

    async def fleet_loop() -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(FLEET_TICK_S)
            now = time.monotonic()
            await asyncio.to_thread(fleet.advance, now - last)
            last = now
            fleet_changed()
            writer = current["writer"]
            if writer is not None and fleet_seq["n"] % FLEET_SIM_EVERY == 0:
                writer.write(encode({"type": "event", "name": "fleet", "args": fleet.for_sim()}))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        server = await asyncio.start_server(handle_sim, sim_host, sim_port)
        app.state.sim_port = server.sockets[0].getsockname()[1]
        loop = asyncio.create_task(fleet_loop()) if fleet is not None else None
        try:
            yield
        finally:
            if loop is not None:
                loop.cancel()
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
            ev = check_event(body)
        except ValueError as e:
            raise HTTPException(400, str(e))
        out = {}
        if fleet is not None and is_fleet_event(ev["name"], ev["args"]):
            try:
                out["fleet"] = fleet.event(ev["name"], ev["args"])
            except (ValueError, TypeError) as e:
                raise HTTPException(400, str(e))
            fleet_changed()
        pending.append(ev)
        queued = len(pending)
        flush()
        if current["writer"] is not None:
            await current["writer"].drain()
        return {"queued": queued, "delivered": not pending, **out}

    def fleet_file(name: str) -> Path:
        for d in fleet_dirs:
            if (Path(d) / name).exists():
                return Path(d) / name
        raise HTTPException(404, f"no {name}")

    @app.get("/api/fleet")
    async def fleet_state() -> Response:
        if fleet is not None:
            return JSONResponse(fleet.state())
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
    async def fleet_map() -> Response:
        if fleet is not None:
            return JSONResponse(fleet.map)
        return FileResponse(fleet_file("fleet_map.json"), media_type="application/json")

    @app.websocket("/ws/fleet")
    async def ws_fleet(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while fleet is not None:
                await ws.send_json(fleet.state())
                await fleet_seq["event"].wait()
            await ws.close()
        except (WebSocketDisconnect, RuntimeError):
            pass

    @app.get("/api/replay")
    async def replay(path: str) -> dict:
        from rig.replay import load_timeline, timeline_session
        p = (root / path).resolve()
        if not any(p.is_relative_to(root / d) for d in ("runs", "runs_sample")):
            raise HTTPException(403, "replay files must live under runs/ or runs_sample/")
        if not p.is_file():
            raise HTTPException(404, f"no {path}")
        return await asyncio.to_thread(lambda: timeline_session(load_timeline(p)))

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
        return FileResponse(REPO / SHOTS[n], media_type="image/jpeg")

    @app.get("/api/frame.jpg")
    async def frame_jpg() -> Response:
        if hub.latest is None:
            return JSONResponse({"detail": "no frames yet"}, status_code=503)
        return Response(hub.latest[1], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    def plume_field() -> Field:
        if hub.plume_field is None:
            raise HTTPException(503, "no plume field yet")
        return hub.plume_field

    @app.get("/api/plume")
    async def plume() -> JSONResponse:
        if hub.plume is None:
            return JSONResponse({"detail": "no plume"}, status_code=404)
        return JSONResponse(hub.plume)

    @app.get("/api/plume.jpg")
    async def plume_jpg() -> Response:
        if hub.plume_jpg is None:
            return JSONResponse({"detail": "no plume image yet"}, status_code=503)
        return Response(hub.plume_jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/plume/field.npz")
    async def plume_npz() -> Response:
        plume_field()
        return Response(hub.plume_npz, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})

    @app.get("/api/plume/sample")
    async def plume_sample(x: float, y: float, z: float) -> dict:
        f = plume_field()
        ppm = f.sample_ppm(x, y, z)
        return {"ppm": round(ppm, 2), "lel_pct": round(ppm / 1e6 / f.lel * 100, 3), "t": f.t}

    @app.post("/api/plume/sample")
    async def plume_sample_many(points: list[list[float]] = Body(..., embed=True)) -> dict:
        f = plume_field()
        if not points or any(len(p) != 3 for p in points):
            raise HTTPException(400, "points must be a list of [x, y, z]")
        return {"ppm": [round(float(v), 2) for v in f.sample(points)], "t": f.t}

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
        seq = pip_seq = 0
        idx = max(0, hub.log_base + len(hub.log) - REPLAY_EVENTS)
        try:
            while True:
                seq, _, header = await hub.wait_newer(seq)
                idx, events = hub.events_since(idx)
                extra = {"plume": hub.plume} if hub.plume else {}
                if hub.pip and hub.pip["seq"] != pip_seq:
                    extra["pip"], pip_seq = hub.pip, hub.pip["seq"]
                await ws.send_json({**header, **extra, "events": events})
                await asyncio.sleep(0.08)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


def default_fleet():
    name = os.environ.get("RIG_FLEET", "judges")
    if name in ("", "0", "off"):
        return None
    try:
        from policy.fleet.live import FleetLive
        return FleetLive(name, float(os.environ.get("RIG_FLEET_SPEED", "4")))
    except (ImportError, OSError) as e:
        print(f"rig web: fleet planner off ({e})", flush=True)
        return None


app = create_app(os.environ.get("SIM_HOST", "127.0.0.1"), int(os.environ.get("SIM_PORT", "9000")), fleet=default_fleet())
