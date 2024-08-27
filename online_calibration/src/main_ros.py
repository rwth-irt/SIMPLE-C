#!/usr/bin/env python
import logging

import rclpy

from .core import websocket_server
from .online_calibrator import OnlineCalibrator

logger = logging.getLogger(__name__)


def main(args=None):
    logging.basicConfig(level=logging.INFO)
    rclpy.init(args=args)
    calibrator = OnlineCalibrator()
    stop_server = websocket_server.main(calibrator.reset)  # start up websocket server in new Thread
    try:
        rclpy.spin(calibrator)
    except KeyboardInterrupt:
        logger.info("Got KeyboardInterrupt, stopping.")
    stop_server()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
