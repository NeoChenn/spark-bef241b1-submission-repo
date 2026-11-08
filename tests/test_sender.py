import socket
import threading
import time

from rig.frames import read_message
from rig.sim.sender import FrameSender


def header(seq: int) -> dict:
    return {"type": "frame", "w": 2, "h": 1, "seq": seq, "t": 0.0}


class Listener:
    def __init__(self, port: int = 0) -> None:
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        self.received: list[dict] = []
        self.conn: socket.socket | None = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            self.conn, _ = self.sock.accept()
            while True:
                h, _ = read_message(self.conn)
                self.received.append(h)
        except (ConnectionError, OSError):
            pass

    def close(self) -> None:
        if self.conn:
            self.conn.close()
        self.sock.close()


def wait_until(cond, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_delivers_frames():
    srv = Listener()
    sender = FrameSender("127.0.0.1", srv.port).start()
    try:
        assert sender.connected.wait(2)
        for seq in range(3):
            sender.send(header(seq), b"\xff\xd8")
            time.sleep(0.05)
        assert wait_until(lambda: srv.received and srv.received[-1]["seq"] == 2, 2)
        assert sender.connected.is_set()
    finally:
        sender.stop()
        srv.close()


def test_send_without_server_does_not_block():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    sender = FrameSender("127.0.0.1", port, retry_s=0.05).start()
    try:
        t0 = time.monotonic()
        for seq in range(100):
            sender.send(header(seq), b"x")
        assert time.monotonic() - t0 < 0.05
        assert not sender.connected.is_set()
        assert sender.dropped >= 99
    finally:
        sender.stop()


def test_reconnects_after_server_restart():
    srv = Listener()
    port = srv.port
    sender = FrameSender("127.0.0.1", port, retry_s=0.05).start()
    try:
        assert sender.connected.wait(2)
        sender.send(header(0), b"x")
        assert wait_until(lambda: srv.received, 2)
        srv.close()
        srv2 = Listener(port)
        try:
            seq = 1
            deadline = time.monotonic() + 2
            while not srv2.received and time.monotonic() < deadline:
                sender.send(header(seq), b"x")
                seq += 1
                time.sleep(0.02)
            assert srv2.received
        finally:
            srv2.close()
    finally:
        sender.stop()


def test_dropped_frame_events_carry_over():
    sender = FrameSender("127.0.0.1", 1)
    sender.send({**header(0), "events": [{"to": "A"}]}, b"x")
    sender.send({**header(1), "events": [{"to": "B"}]}, b"y")
    sender.send(header(2), b"z")
    h, payload = sender._take()
    assert payload == b"z" and h["seq"] == 2
    assert [e["to"] for e in h["events"]] == ["A", "B"]


def test_receives_events_from_server():
    from rig.frames import encode
    srv = Listener()
    sender = FrameSender("127.0.0.1", srv.port).start()
    try:
        assert sender.connected.wait(2)
        assert wait_until(lambda: srv.conn is not None, 2)
        srv.conn.sendall(encode({"type": "event", "name": "set_glare", "args": {"on": True}}))
        assert wait_until(lambda: not sender.inbox.empty(), 2)
        assert sender.inbox.get() == {"name": "set_glare", "args": {"on": True}}
    finally:
        sender.stop()
        srv.close()
