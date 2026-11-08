import asyncio
import contextlib
import os
from collections import deque
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from rig.frames import encode, read_message_async
from rig.hud import EXTRA, FLEET, HOOKS, is_fleet_event

STATIC = Path(__file__).parent / "static"
REPO = Path(__file__).resolve().parents[2]
MOCK_PATH = REPO / "runs/hud/mock_session.json"
SHOTS = {1: "sim/output/cine_storm_aerial.png", 2: "sim/output/preview_hero.png",
         3: "sim/output/preview_repair_closeup.png"}
LOG_KEEP = 200
REPLAY_EVENTS = 20
FPS_WINDOW_S = 2.0
STALE_AFTER_S = 2.0
BLACK_MEAN = 3.0
FLEET_TICK_S = 0.25
FLEET_SIM_EVERY = 2


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


def create_app(sim_host: str = "127.0.0.1", sim_port: int = 9000, mock_path: Path = MOCK_PATH, fleet=None) -> FastAPI:
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
        name = body.get("name")
        args = body.get("args") or {}
        if name not in HOOKS + EXTRA + FLEET:
            raise HTTPException(400, f"unknown event {name!r}; expected one of {HOOKS + EXTRA + FLEET}")
        out = {}
        if fleet is not None and is_fleet_event(name, args):
            try:
                out["fleet"] = fleet.event(name, args)
            except (ValueError, TypeError) as e:
                raise HTTPException(400, str(e))
            fleet_changed()
        pending.append({"name": name, "args": args})
        queued = len(pending)
        flush()
        if current["writer"] is not None:
            await current["writer"].drain()
        return {"queued": queued, "delivered": not pending, **out}

    @app.get("/api/fleet")
    async def fleet_state() -> dict:
        if fleet is None:
            raise HTTPException(404, "fleet planner not running")
        return fleet.state()

    @app.get("/api/fleet/map")
    async def fleet_map() -> dict:
        if fleet is None:
            raise HTTPException(404, "fleet planner not running")
        return fleet.map

    @app.websocket("/ws/fleet")
    async def ws_fleet(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while fleet is not None:
                await ws.send_json(fleet.state())
                await fleet_seq["event"].wait()
        except (WebSocketDisconnect, RuntimeError):
            pass

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
