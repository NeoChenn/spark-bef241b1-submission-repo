import asyncio
import json
import socket
import struct

PREFIX = struct.Struct("<II")
MAX_HEADER = 512 * 1024
MAX_PAYLOAD = 16 * 1024 * 1024


def encode(header: dict, payload: bytes = b"") -> bytes:
    h = json.dumps(header, separators=(",", ":")).encode()
    return PREFIX.pack(len(h), len(payload)) + h + payload


def _check(hlen: int, plen: int) -> None:
    if hlen > MAX_HEADER or plen > MAX_PAYLOAD:
        raise ConnectionError(f"message too large: header={hlen} payload={plen}")


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def read_message(sock: socket.socket) -> tuple[dict, bytes]:
    hlen, plen = PREFIX.unpack(_recv_exactly(sock, PREFIX.size))
    _check(hlen, plen)
    header = json.loads(_recv_exactly(sock, hlen))
    payload = _recv_exactly(sock, plen) if plen else b""
    return header, payload


async def read_message_async(reader: asyncio.StreamReader) -> tuple[dict, bytes]:
    try:
        hlen, plen = PREFIX.unpack(await reader.readexactly(PREFIX.size))
        _check(hlen, plen)
        header = json.loads(await reader.readexactly(hlen))
        payload = await reader.readexactly(plen) if plen else b""
    except asyncio.IncompleteReadError as e:
        raise ConnectionError("connection closed") from e
    return header, payload
