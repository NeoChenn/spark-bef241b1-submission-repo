import asyncio
import contextlib
import hmac
import json
import os
from collections import deque
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from rig.frames import encode, read_message_async
from rig.hud import check_event
from rig.render.state import StateLog, split

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
HELLO_TIMEOUT_S = 5.0


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
        self.meta: dict = {}
        self.version = 0
        self._state_event = asyncio.Event()
        self.latency_ms: float | None = None

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
        if "state_ts" in header:
            self.latency_ms = round((time.time() - header["state_ts"]) * 1000, 1)
        self._absorb(header)
        self._times = [t for t in self._times if now - t <= FPS_WINDOW_S]
        self._times.append(now)
        event, self._event = self._event, asyncio.Event()
        event.set()
        self._bump()

    def publish_meta(self, meta: dict) -> None:
        self._absorb(meta)
        self.meta = {k: v for k, v in meta.items() if k != "events"}
        self._bump()

    def _absorb(self, header: dict) -> None:
        self.log.extend(header.get("events") or [])
        if isinstance(header.get("fleet"), dict):
            self.fleet = header["fleet"]
        if len(self.log) > LOG_KEEP:
            self.log_base += len(self.log) - LOG_KEEP
            self.log = self.log[-LOG_KEEP:]

    def _bump(self) -> None:
        self.version += 1
        event, self._state_event = self._state_event, asyncio.Event()
        event.set()

    def header(self) -> dict:
        return {**self.meta, **(self.latest[2] if self.latest else {})}

    async def wait_state(self, version: int) -> tuple[int, dict]:
        while self.version <= version:
            await self._state_event.wait()
        return self.version, self.header()

    async def wait_newer(self, seq: int) -> tuple[int, bytes, dict]:
        while self.latest is None or self.latest[0] <= seq:
            await self._event.wait()
        return self.latest

    def events_since(self, idx: int) -> tuple[int, list[dict]]:
        end = self.log_base + len(self.log)
        return end, self.log[max(0, idx - self.log_base):]


class RenderClient:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        self.log = StateLog()
        self.wake = asyncio.Event()
        self.fresh = True


def create_app(sim_host: str = "127.0.0.1", sim_port: int = 9000, mock_path: Path = MOCK_PATH,
               fleet_dirs=FLEET_DIRS, root: Path = REPO, lan_host: str | None = None, lan_port: int | None = None,
               token: str | None = None) -> FastAPI:
    root = Path(root).resolve()
    hub = FrameHub()
    current: dict[str, asyncio.StreamWriter | None] = {"writer": None}
    pending: deque = deque()
    renders: set[RenderClient] = set()
    scene_log = StateLog()
    sim_frame_at = [-1e9]

    def push_state(scene: dict) -> None:
        scene_log.merge(scene)
        for client in renders:
            client.log.merge(scene)
            client.wake.set()

    def flush() -> None:
        writer = current["writer"]
        while writer is not None and pending:
            writer.write(encode({"type": "event", **pending.popleft()}))

    def on_sim_message(header: dict, payload: bytes) -> None:
        kind = header.get("type")
        if kind == "frame":
            sim_frame_at[0] = time.monotonic()
            hub.publish(payload, header)
        elif kind == "state":
            scene, meta = split(header)
            hub.publish_meta(meta)
            push_state(scene)

    async def handle_sim(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, first=None) -> None:
        previous = current["writer"]
        current["writer"] = writer
        if previous is not None:
            previous.close()
        hub.sim_connected = True
        flush()
        try:
            if first is not None:
                on_sim_message(*first)
            while True:
                on_sim_message(*await read_message_async(reader))
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            if current["writer"] is writer:
                current["writer"] = None
                hub.sim_connected = False
            writer.close()

    async def pump(client: RenderClient) -> None:
        while True:
            await client.wake.wait()
            client.wake.clear()
            if client.fresh:
                client.fresh = False
                client.log = StateLog()
                msg = scene_log.message(True)
            else:
                msg = client.log.take()
            client.writer.write(encode(msg))
            await client.writer.drain()

    async def handle_render(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        client = RenderClient(writer)
        renders.add(client)
        if scene_log.head:
            client.wake.set()
        task = asyncio.create_task(pump(client))
        try:
            while True:
                header, payload = await read_message_async(reader)
                if header.get("type") == "frame" and time.monotonic() - sim_frame_at[0] > STALE_AFTER_S:
                    hub.publish(payload, {**header, "source": "render"})
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            renders.discard(client)
            task.cancel()
            writer.close()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, lan: bool) -> None:
        try:
            first = await asyncio.wait_for(read_message_async(reader), HELLO_TIMEOUT_S if lan else None)
        except (ConnectionError, OSError, ValueError, asyncio.TimeoutError):
            writer.close()
            return
        header = first[0]
        hello = header.get("type") == "hello"
        if lan and not (hello and token and hmac.compare_digest(str(header.get("token", "")), token)):
            writer.close()
            return
        if hello and header.get("role") == "render":
            await handle_render(reader, writer)
        else:
            await handle_sim(reader, writer, None if hello else first)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        server = await asyncio.start_server(lambda r, w: handle(r, w, False), sim_host, sim_port)
        app.state.sim_port = server.sockets[0].getsockname()[1]
        servers = [server]
        if lan_host and token:
            lan = await asyncio.start_server(lambda r, w: handle(r, w, True), lan_host,
                                             app.state.sim_port if lan_port is None else lan_port)
            app.state.lan_port = lan.sockets[0].getsockname()[1]
            servers.append(lan)
        elif lan_host:
            print(f"rig web: {lan_host} ingest disabled, INGEST_TOKEN is empty", flush=True)
        try:
            yield
        finally:
            for s in servers:
                s.close()
            for client in list(renders):
                client.writer.close()
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
            "render_connected": bool(renders),
            "source": header.get("source", "sim"),
            "latency_ms": hub.latency_ms,
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
        version = 0
        idx = max(0, hub.log_base + len(hub.log) - REPLAY_EVENTS)
        try:
            while True:
                version, header = await hub.wait_state(version)
                idx, events = hub.events_since(idx)
                await ws.send_json({**header, "events": events})
                await asyncio.sleep(0.08)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


app = create_app(os.environ.get("SIM_HOST", "127.0.0.1"), int(os.environ.get("SIM_PORT", "9000")),
                 lan_host=os.environ.get("SIM_HOST_LAN") or None, token=os.environ.get("INGEST_TOKEN") or None)
