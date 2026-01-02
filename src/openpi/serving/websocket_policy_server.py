import asyncio
import http
import logging
import time
import traceback

import numpy as np

import os
import jax

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        trace_dir = os.environ.get("OPENPI_TRACE_DIR", "/tmp/openpi_trace")
        did_trace = False

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                # action = self._policy.infer(obs)

                if not did_trace:
                    did_trace = True

                    # Warm up (important for JIT)
                    for _ in range(2):
                        warm = self._policy.infer(obs)
                        # Force sync so warmup actually executes
                        jax.tree.map(
                            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                            warm,
                        )

                    # Trace one inference
                    jax.profiler.start_trace(trace_dir)
                    action = self._policy.infer(obs)
                    jax.tree.map(
                        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                        action,
                    )
                    jax.profiler.stop_trace()

                    logger.info("Wrote JAX trace to %s", trace_dir)
                else:
                    action = self._policy.infer(obs)

                infer_time = time.monotonic() - infer_time

                # Collect component timings from model profiler
                component_timings = {}
                
                # Try to get the inner model
                inner_policy = self._policy
                # Unwrap recorder if present
                if hasattr(inner_policy, '_policy'):
                    inner_policy = inner_policy._policy
                
                # Get model from policy
                if hasattr(inner_policy, '_model'):
                    model = inner_policy._model
                    print(f"SERVER: Trying to get timings from model...")
                    print(f"SERVER: Model type: {type(model)}")
                    print(f"SERVER: Has profiler? {hasattr(model, 'profiler')}")
                    # Get component timings if profiler exists
                    if hasattr(model, 'profiler'):
                        timings = model.profiler.get_timings()
                        print(f"SERVER: Got timings: {timings}")
                        component_timings = timings
                    else:
                        print(f"SERVER: Model has NO profiler attribute")
                

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }

                action["component_timings"] = component_timings


                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
