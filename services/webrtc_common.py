from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import HTTPException
from pydantic import BaseModel

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription

    AIORTC_AVAILABLE = True
except Exception:
    RTCPeerConnection = Any  # type: ignore[assignment]
    RTCSessionDescription = Any  # type: ignore[assignment]
    AIORTC_AVAILABLE = False


class RTCOffer(BaseModel):
    sdp: str
    type: str


RPCResult = dict[str, Any]
RPCStream = AsyncIterator[RPCResult]
RPCHandler = Callable[[str, dict[str, Any], bool], Awaitable[RPCResult | RPCStream]]


def _send_json(channel: Any, payload: dict[str, Any]) -> None:
    channel.send(json.dumps(payload, ensure_ascii=True))


def create_offer_handler(peers: set[Any], handler: RPCHandler):
    async def offer_endpoint(offer: RTCOffer) -> dict[str, str]:
        if not AIORTC_AVAILABLE:
            raise HTTPException(status_code=503, detail="aiortc is not installed")

        pc = RTCPeerConnection()
        peers.add(pc)

        @pc.on("connectionstatechange")
        async def _on_connection_state_change() -> None:
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await pc.close()
                peers.discard(pc)

        @pc.on("datachannel")
        def _on_datachannel(channel: Any) -> None:
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
                    request = json.loads(message)
                    req_id = str(request.get("id", "")).strip()
                    op = str(request.get("op", "")).strip()
                    stream = bool(request.get("stream", False))
                    payload = request.get("payload", {})
                    if not isinstance(payload, dict):
                        payload = {}
                except Exception as exc:
                    _send_json(channel, {"id": "", "event": "error", "error": f"bad_request:{exc}"})
                    return

                async def _process() -> None:
                    try:
                        response = await handler(op=op, payload=payload, stream=stream)
                        if stream:
                            assert hasattr(response, "__aiter__")
                            async for part in response:  # type: ignore[union-attr]
                                _send_json(channel, {"id": req_id, "event": "chunk", "payload": part})
                            _send_json(channel, {"id": req_id, "event": "done", "payload": {}})
                            return

                        assert isinstance(response, dict)
                        _send_json(channel, {"id": req_id, "event": "result", "payload": response})
                    except Exception as exc:
                        _send_json(channel, {"id": req_id, "event": "error", "error": str(exc)})

                asyncio.create_task(_process())

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": str(pc.localDescription.sdp),
            "type": str(pc.localDescription.type),
        }

    return offer_endpoint
