import asyncio
import socket

import pytest

from rig.frames import MAX_HEADER, PREFIX, encode, read_message, read_message_async

HEADER = {"type": "frame", "w": 2, "h": 1, "seq": 7, "t": 1.5}
PAYLOAD = b"\xff\xd8xyz"


def feed(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


async def read_fed(data: bytes):
    return await read_message_async(feed(data))


def test_roundtrip_socketpair():
    a, b = socket.socketpair()
    with a, b:
        a.sendall(encode(HEADER, PAYLOAD))
        assert read_message(b) == (HEADER, PAYLOAD)


def test_roundtrip_async():
    assert asyncio.run(read_fed(encode(HEADER, PAYLOAD))) == (HEADER, PAYLOAD)


def test_eof_mid_message_raises():
    partial = encode(HEADER, PAYLOAD)[:10]
    a, b = socket.socketpair()
    with a, b:
        a.sendall(partial)
        a.shutdown(socket.SHUT_WR)
        with pytest.raises(ConnectionError):
            read_message(b)
    with pytest.raises(ConnectionError):
        asyncio.run(read_fed(partial))


def test_oversize_header_rejected():
    data = PREFIX.pack(MAX_HEADER + 1, 0)
    a, b = socket.socketpair()
    with a, b:
        a.sendall(data)
        a.shutdown(socket.SHUT_WR)
        with pytest.raises(ConnectionError):
            read_message(b)
    with pytest.raises(ConnectionError):
        asyncio.run(read_fed(data))
