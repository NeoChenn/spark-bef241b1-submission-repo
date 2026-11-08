import queue
import socket
import threading

from rig.frames import encode, read_message


class FrameSender:
    def __init__(self, host: str, port: int, retry_s: float = 1.0) -> None:
        self.host, self.port, self.retry_s = host, port, retry_s
        self.connected = threading.Event()
        self.sent = 0
        self.dropped = 0
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._slot: tuple[dict, bytes] | None = None
        self.inbox: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="frame-sender", daemon=True)

    def start(self) -> "FrameSender":
        self._thread.start()
        return self

    def send(self, header: dict, payload: bytes) -> None:
        with self._lock:
            if self._slot is not None:
                self.dropped += 1
                lost = self._slot[0].get("events")
                if lost:
                    header = {**header, "events": lost + list(header.get("events") or [])}
                old = self._slot[0].get("plume")
                if old:
                    header = {**header, "plume": {**old, **(header.get("plume") or {})}}
            self._slot = (header, payload)
        self._ready.set()

    def stop(self) -> None:
        self._stop.set()
        self._ready.set()
        self._thread.join(timeout=3)

    def _take(self) -> tuple[dict, bytes] | None:
        with self._lock:
            item, self._slot = self._slot, None
            self._ready.clear()
        return item

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sock = socket.create_connection((self.host, self.port), timeout=2)
            except OSError:
                self._stop.wait(self.retry_s)
                continue
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connected.set()
            threading.Thread(target=self._listen, args=(sock,), name="frame-sender-rx", daemon=True).start()
            try:
                while not self._stop.is_set():
                    if not self._ready.wait(0.5):
                        continue
                    item = self._take()
                    if item is not None:
                        sock.sendall(encode(*item))
                        self.sent += 1
            except OSError:
                pass
            finally:
                self.connected.clear()
                sock.close()
            self._stop.wait(self.retry_s)

    def _listen(self, sock: socket.socket) -> None:
        try:
            while True:
                header, _ = read_message(sock)
                if header.get("type") == "event":
                    self.inbox.put({"name": header.get("name"), "args": header.get("args") or {}})
        except (ConnectionError, OSError, ValueError):
            pass
