import asyncio
import contextlib
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from rig.frames import read_message_async

STATIC = Path(__file__).parent / "static"
FPS_WINDOW_S = 2.0
STALE_AFTER_S = 2.0
BLACK_MEAN = 3.0


class FrameHub:
    def __init__(self) -> None:
        self.latest: tuple[int, bytes, dict] | None = None
        self.frames = 0
        self.last_frame_at: float | None = None
        self.sim_connected = False
        self._seq = 0
        self._event = asyncio.Event()
        self._times: list[float] = []

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
        self._times = [t for t in self._times if now - t <= FPS_WINDOW_S]
        self._times.append(now)
        event, self._event = self._event, asyncio.Event()
        event.set()

    async def wait_newer(self, seq: int) -> tuple[int, bytes, dict]:
        while self.latest is None or self.latest[0] <= seq:
            await self._event.wait()
        return self.latest


def create_app(sim_host: str = "127.0.0.1", sim_port: int = 9000) -> FastAPI:
    hub = FrameHub()
    current: dict[str, asyncio.StreamWriter | None] = {"writer": None}

    async def handle_sim(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        previous = current["writer"]
        current["writer"] = writer
        if previous is not None:
            previous.close()
        hub.sim_connected = True
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
        }

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

    return app


app = create_app(os.environ.get("SIM_HOST", "127.0.0.1"), int(os.environ.get("SIM_PORT", "9000")))
