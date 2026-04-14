from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator

import httpx

_STREAM_DONE = object()


def default_webrtc_offer_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[: -len("/v1")]
    return f"{normalized}/webrtc/offer"


class WebRTCRPCClient:
    """Simple request/response RPC over a persistent WebRTC data channel."""

    def __init__(
        self,
        offer_url: str,
        channel_label: str,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.offer_url = offer_url
        self.channel_label = channel_label
        self.timeout_s = timeout_s

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

        self._pc: Any | None = None
        self._channel: Any | None = None
        self._open_event = asyncio.Event()
        self._ready_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()

        self._pending_unary: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._pending_streams: dict[str, asyncio.Queue[Any]] = {}

    async def _ensure_ready(self) -> None:
        if self._channel is not None and self._open_event.is_set():
            return

        async with self._ready_lock:
            if self._channel is not None and self._open_event.is_set():
                return
            await self._connect()

    async def _connect(self) -> None:
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
        except Exception as exc:
            raise RuntimeError(
                "WebRTC transport requires aiortc. Install with: pip install aiortc"
            ) from exc

        self._open_event = asyncio.Event()
        pc = RTCPeerConnection()
        channel = pc.createDataChannel(self.channel_label)

        @channel.on("open")
        def _on_open() -> None:
            self._open_event.set()

        @channel.on("message")
        def _on_message(message: Any) -> None:
            if isinstance(message, bytes):
                try:
                    message = message.decode("utf-8")
                except Exception:
                    return

            if not isinstance(message, str):
                return

            try:
                payload = json.loads(message)
            except Exception:
                return

            self._handle_incoming(payload)

        @pc.on("connectionstatechange")
        async def _on_connection_state_change() -> None:
            if pc.connectionState in {"failed", "disconnected", "closed"}:
                await self.close()

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        response = await self._client.post(
            self.offer_url,
            json={
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            },
        )
        response.raise_for_status()
        answer = response.json()

        await pc.setRemoteDescription(
            RTCSessionDescription(
                sdp=str(answer["sdp"]),
                type=str(answer["type"]),
            )
        )

        await asyncio.wait_for(self._open_event.wait(), timeout=self.timeout_s)
        self._pc = pc
        self._channel = channel

    def _handle_incoming(self, payload: dict[str, Any]) -> None:
        req_id = str(payload.get("id", "")).strip()
        if not req_id:
            return

        event = str(payload.get("event", "result")).strip().lower()

        if event == "result":
            result = payload.get("payload", {})
            future = self._pending_unary.pop(req_id, None)
            if future is not None and not future.done():
                if isinstance(result, dict):
                    future.set_result(result)
                else:
                    future.set_result({"result": result})
            return

        if event == "chunk":
            queue = self._pending_streams.get(req_id)
            if queue is not None:
                queue.put_nowait(payload.get("payload", {}))
            return

        if event == "done":
            queue = self._pending_streams.pop(req_id, None)
            if queue is not None:
                queue.put_nowait(_STREAM_DONE)
            return

        if event == "error":
            exc = RuntimeError(str(payload.get("error", "WebRTC RPC error")))
            future = self._pending_unary.pop(req_id, None)
            if future is not None and not future.done():
                future.set_exception(exc)

            queue = self._pending_streams.pop(req_id, None)
            if queue is not None:
                queue.put_nowait(exc)

    async def _send_json(self, message: dict[str, Any]) -> None:
        await self._ensure_ready()
        if self._channel is None:
            raise RuntimeError("WebRTC data channel is not ready")

        serialized = json.dumps(message, ensure_ascii=True)
        async with self._send_lock:
            self._channel.send(serialized)

    async def call_unary(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        req_id = uuid.uuid4().hex
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_unary[req_id] = future

        await self._send_json(
            {
                "id": req_id,
                "op": op,
                "stream": False,
                "payload": payload,
            }
        )

        try:
            return await asyncio.wait_for(future, timeout=self.timeout_s)
        except Exception:
            self._pending_unary.pop(req_id, None)
            raise

    async def call_stream(self, op: str, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        req_id = uuid.uuid4().hex
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self._pending_streams[req_id] = queue

        await self._send_json(
            {
                "id": req_id,
                "op": op,
                "stream": True,
                "payload": payload,
            }
        )

        try:
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=self.timeout_s)
                if item is _STREAM_DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                if isinstance(item, dict):
                    yield item
                else:
                    yield {"value": item}
        finally:
            self._pending_streams.pop(req_id, None)

    async def close(self) -> None:
        for req_id, future in list(self._pending_unary.items()):
            if not future.done():
                future.set_exception(RuntimeError("WebRTC client closed"))
            self._pending_unary.pop(req_id, None)

        for req_id, queue in list(self._pending_streams.items()):
            queue.put_nowait(RuntimeError("WebRTC client closed"))
            self._pending_streams.pop(req_id, None)

        if self._pc is not None:
            await self._pc.close()
            self._pc = None
            self._channel = None
            self._open_event.clear()

        if self._owns_client:
            await self._client.aclose()
