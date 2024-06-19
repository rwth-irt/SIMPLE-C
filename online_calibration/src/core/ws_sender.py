import asyncio
import json
import logging
import threading

import numpy as np
import websockets

from .transformation import Transformation

logging.basicConfig(level=logging.INFO)

_clients: list[websockets.WebSocketServerProtocol] = []


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# counter to only send every n-th message
msg_index = 0


def broadcast(
        frames: dict[str, tuple[np.ndarray, np.ndarray, int]],
        transformations: list[tuple[str, str, Transformation]],
        every_nth_point=1,
        every_nth_message=4,
):
    """

    :param frames: dict{topic -> (points, clustering)}
    :param transformations: list[(from, to, Transformation)]
    :return:
    """
    global msg_index
    msg_index += 1
    if msg_index % every_nth_message != 0:
        return

    data = {
        "frames": [
            {
                "topic": topic,
                "points": data[::every_nth_point, :3].round(3).flatten(),
                "colors": color_index[::every_nth_point],
                "marker_index": marker_index if marker_index is not None else -10  # (-10 is no valid color index)
            } for topic, (data, color_index, marker_index) in frames.items()
        ],
        "transformations": [
            {
                "from": a,
                "to": b,
                "trafo": {
                    "t": t.t,
                    "R_quat": t.R_quat
                }
            } for a, b, t in transformations
        ]
    }
    msg = json.dumps(data, cls=_NumpyEncoder)
    # queue this for execution in ws server thread
    asyncio.run_coroutine_threadsafe(_broadcast(msg), loop)


async def _broadcast(msg: str):
    for c in _clients:
        await c.send(msg)


async def _on_connection(socket: websockets.WebSocketServerProtocol):
    _clients.append(socket)
    try:
        logging.info("New client connection")
        async for _message in socket:
            pass
        # loop terminates on disconnection
    except websockets.ConnectionClosedOK or websockets.ConnectionClosedError or websockets.ConnectionClosed:
        pass
    finally:
        _clients.remove(socket)
        logging.info("Client left")


async def _ws_thread_main():
    async with websockets.serve(_on_connection, "0.0.0.0", 6789):
        print("WS server listening")
        await asyncio.Future()  # run forever
    print("WS server stopped")


# Event loop for ws server
loop = asyncio.new_event_loop()


def main():
    # Start ws server in new Thread
    t = threading.Thread(target=loop.run_until_complete, args=[_ws_thread_main()])
    t.start()
