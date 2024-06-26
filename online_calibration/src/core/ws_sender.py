import asyncio
import json
import logging
import threading

import numpy as np
import websockets

from .frame import Frame
from .reflector_location import ReflectorLocation
from .transformation import Transformation

logging.basicConfig(level=logging.INFO)

_clients: list[websockets.WebSocketServerProtocol] = []


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


frame_send_counter = {
    "offset": 0
}


def broadcast_frame(
        frame: Frame,
        reflector_cluster_index: int = None,
        every_nth_point=3,
        every_nth_message=3,
):
    # called from PairCalibrator when new frame has been analyzed

    # check whether to skip this message due to every_nth_message
    if frame.topic not in frame_send_counter:
        frame_send_counter[frame.topic] = frame_send_counter["offset"]
        frame_send_counter["offset"] += 1
    frame_send_counter[frame.topic] += 1
    if frame_send_counter[frame.topic] % every_nth_message != 0:
        return

    colors_edited = frame.clustering[::every_nth_point].astype(np.int8)
    colors_edited[colors_edited >= 0] = 0  # avoid unused multi-digit numbers
    colors_edited[colors_edited == reflector_cluster_index] = -3  # websocket-specific special value
    data = {
        "type": "pointcloud",
        "topic": frame.topic,
        "data": {
            "points": frame.data[::every_nth_point, :3].flatten().astype(np.float64).round(2),
            "colors": colors_edited,
        }
    }
    # Sending this in a more compressed/binary format may be good...
    # However, the ws server *should* at least compress data.
    _broadcast_internal(data)


def broadcast_metadata(
        topic: str,
        reflector_locations: list[ReflectorLocation],
        transformation: Transformation
):
    # called from OnlineCalibrator after calculating absolute transformations
    locations = np.array([rl.centroid[:3] for rl in reflector_locations]).flatten().astype(np.float64).round(2)
    data = {
        "type": "metadata",
        "topic": topic,
        "data": {
            "reflector_locations": locations,
            "transformation": {
                "t": transformation.t,
                "R_quat": transformation.R_quat
            }
        }
    }
    _broadcast_internal(data)


# TODO add separate tracking_status message if required, send it when available (i.e. from pairCalibrator)


def _broadcast_internal(data: dict):
    """Send a data dict to all clients, thread-safe (can be called from any thread)"""
    msg = json.dumps(data, cls=_NumpyEncoder, indent=0, separators=(',', ':'), allow_nan=False)  # no whitespace in JSON
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
