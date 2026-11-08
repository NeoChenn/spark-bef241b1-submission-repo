import asyncio
import base64
import binascii
import contextlib
import hmac
import json
import os
from collections import deque
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from rig.frames import encode, read_message_async
from rig.hud import check_event, is_fleet_event
from rig.plume import Field
from rig.render.state import StateLog, split

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
HELLO_TIMEOUT_S = 5.0
FLEET_TICK_S = 0.25
CONTROL_TICK_S = 1.0
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
        self.control: dict | None = None
        self.meta: dict = {}
        self.mission: dict = {}
        self.version = 0
        self._state_event = asyncio.Event()
        self.latency_ms: float | None = None

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
        ts = header.get("state_ts")
        self.latency_ms = round((time.time() - ts) * 1000, 1) if ts else None
        self._absorb(header)
        self._times = [t for t in self._times if now - t <= FPS_WINDOW_S]
        self._times.append(now)
        event, self._event = self._event, asyncio.Event()
        event.set()
        self._bump()

    def publish_meta(self, meta: dict) -> None:
        meta = self.take_plume(meta)
        self._absorb(meta)
        self.meta = {k: v for k, v in meta.items() if k != "events"}
        self._bump()

    def publish_mission(self, h: dict) -> None:
        h = self.take_plume(h)
        self.log.extend(h.get("events") or [])
        self.mission = {k: v for k, v in h.items() if k != "events"}
        self._trim()
        self._bump()

    def _absorb(self, header: dict) -> None:
        self.log.extend(header.get("events") or [])
        if isinstance(header.get("fleet"), dict):
            self.fleet = header["fleet"]
        self._trim()

    def _trim(self) -> None:
        if len(self.log) > LOG_KEEP:
            self.log_base += len(self.log) - LOG_KEEP
            self.log = self.log[-LOG_KEEP:]

    def _bump(self) -> None:
        self.version += 1
        event, self._state_event = self._state_event, asyncio.Event()
        event.set()

    def header(self) -> dict:
        return {**self.meta, **(self.latest[2] if self.latest else {}), **self.mission}

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
               fleet_dirs=FLEET_DIRS, root: Path = REPO, fleet=None, lan_host: str | None = None,
               lan_port: int | None = None, token: str | None = None, control=None, mission=None,
               event_gap_s: float = 0.0) -> FastAPI:
    root = Path(root).resolve()
    last_event: dict[str, float] = {}
    hub = FrameHub()
    if mission is not None:
        fleet, control = mission.fleet, mission.control
        mission.leak.field = lambda: hub.plume_field if (hub.plume or {}).get("flange") == mission.leak.fid else None
        mission.sim_flange = lambda: (hub.plume or {}).get("flange")
    live = mission or fleet
    fleet_seq = {"n": 0, "event": asyncio.Event()}
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

    def fleet_changed() -> None:
        fleet_seq["n"] += 1
        event, fleet_seq["event"] = fleet_seq["event"], asyncio.Event()
        event.set()

    async def fleet_loop() -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(FLEET_TICK_S)
            now = time.monotonic()
            if mission is not None:
                await asyncio.to_thread(mission.advance, now - last)
                hub.publish_mission(await asyncio.to_thread(mission.header))
            else:
                await asyncio.to_thread(fleet.advance, now - last)
            last = now
            fleet_changed()
            writer = current["writer"]
            if writer is not None and fleet_seq["n"] % FLEET_SIM_EVERY == 0:
                writer.write(encode({"type": "event", "name": "fleet", "args": live.for_sim()}))

    async def control_loop() -> None:
        from policy.control.adapters import from_header

        while True:
            await asyncio.sleep(CONTROL_TICK_S)
            if mission is not None:
                try:
                    hub.control = await asyncio.to_thread(mission.control_tick)
                except Exception as e:
                    print(f"rig web: mission control tick failed: {type(e).__name__}: {e}", flush=True)
                hub._bump()
                continue
            st = fleet.state() if fleet is not None else (hub.fleet or {})
            h = {**hub.meta, "t": st.get("t", hub.meta.get("t")), "robots": st.get("robots") or [],
                 "tasks": st.get("tasks") or []}
            try:
                hub.control = control.tick(from_header(h))
            except Exception as e:
                print(f"rig web: control tick failed: {type(e).__name__}: {e}", flush=True)
            hub._bump()

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
        loop = asyncio.create_task(fleet_loop()) if fleet is not None else None
        ctl = asyncio.create_task(control_loop()) if control is not None else None
        try:
            yield
        finally:
            for task in (loop, ctl):
                if task is not None:
                    task.cancel()
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
    app.include_router(__import__("rig.web.views_api", fromlist=["views_router"]).views_router(hub))
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/interactive")
    async def interactive() -> FileResponse:
        return FileResponse(STATIC / "interactive.html")

    @app.get("/judge")
    async def judge() -> RedirectResponse:
        return RedirectResponse("/interactive")

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
    async def post_event(body: dict, request: Request):
        who = (request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
               or (request.client.host if request.client else "?"))
        now = time.monotonic()
        wait = event_gap_s - (now - last_event.get(who, -1e9))
        if wait > 0:
            return JSONResponse({"detail": "one change per second, please", "retry_s": round(wait, 2)}, status_code=429)
        last_event[who] = now
        if len(last_event) > 4096:
            last_event.clear()
        try:
            ev = check_event(body)
        except ValueError as e:
            raise HTTPException(400, str(e))
        out = {}
        if mission is not None:
            try:
                out.update(await asyncio.to_thread(mission.event, ev["name"], ev["args"]))
            except (ValueError, TypeError, KeyError) as e:
                raise HTTPException(400, str(e))
            fleet_changed()
        elif control is not None and ev["name"] == "cut_link":
            control.cut_link(bool(ev["args"].get("on", True)))
        elif control is not None and ev["name"] == "intent":
            control.intent(str(ev["args"].get("text") or ""))
        if mission is None and fleet is not None and is_fleet_event(ev["name"], ev["args"]):
            try:
                out["fleet"] = fleet.event(ev["name"], ev["args"])
            except (ValueError, TypeError) as e:
                raise HTTPException(400, str(e))
            fleet_changed()
        pending.append(ev)
        if mission is not None and hasattr(mission, "take_sim"):
            pending.extend(mission.take_sim())
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
        if live is not None:
            return JSONResponse(live.state())
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
            while live is not None:
                await ws.send_json(live.state())
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
            try:
                from rig.hud import record_mock
                await asyncio.to_thread(record_mock, mock_path, str(mock_path.parent / "decisions.jsonl"))
            except (ImportError, OSError):
                if not MOCK_SAMPLE.exists():
                    raise HTTPException(503, "no mock session")
                return FileResponse(MOCK_SAMPLE, media_type="application/json")
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
        version = pip_seq = 0
        idx = max(0, hub.log_base + len(hub.log) - REPLAY_EVENTS)
        try:
            while True:
                version, header = await hub.wait_state(version)
                idx, events = hub.events_since(idx)
                extra = {"plume": hub.plume} if hub.plume else {}
                if hub.control is not None:
                    extra["control"] = hub.control
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


def default_mission():
    if os.environ.get("RIG_MISSION", "1") in ("", "0", "off"):
        return None
    try:
        from rig.mission.live import MissionLive
        fleet = default_fleet()
        if fleet is None:
            return None
        return MissionLive(fleet, control=mission_control)
    except (ImportError, OSError) as e:
        print(f"rig web: mission brain off ({e})", flush=True)
        return None


def mission_control(port):
    mode = os.environ.get("RIG_CONTROL", "auto")
    if mode in ("", "0", "off"):
        return None
    from policy.control.centre import from_env
    from policy.gemini_verifier import load_api_key
    if mode == "auto" and not load_api_key():
        mode = "edge"
    return from_env(mode, log_path=REPO / "runs/control/calls.jsonl", sync=False, port=port,
                    interval_s=float(os.environ.get("RIG_CONTROL_INTERVAL", "10")))


def default_control():
    mode = os.environ.get("RIG_CONTROL", "auto")
    if mode in ("", "0", "off"):
        return None
    try:
        from policy.control.centre import from_env
        return from_env(mode, log_path=REPO / "runs/control/calls.jsonl", sync=False,
                        interval_s=float(os.environ.get("RIG_CONTROL_INTERVAL", "30")))
    except (ImportError, OSError) as e:
        print(f"rig web: control centre off ({e})", flush=True)
        return None


MISSION = default_mission()
app = create_app(os.environ.get("SIM_HOST", "127.0.0.1"), int(os.environ.get("SIM_PORT", "9000")),
                 fleet=None if MISSION else default_fleet(), control=None if MISSION else default_control(),
                 mission=MISSION, event_gap_s=float(os.environ.get("RIG_EVENT_GAP_S", "1")),
                 lan_host=os.environ.get("SIM_HOST_LAN") or None, token=os.environ.get("INGEST_TOKEN") or None)
