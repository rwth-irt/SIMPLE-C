#!/usr/bin/env python
import rclpy

from .online_calibrator import OnlineCalibrator
from .core import websocket_server


def main(args=None):
    rclpy.init(args=args)
    calibrator = OnlineCalibrator()
    stop_server = websocket_server.main(calibrator.reset)  # start up websocket server in new Thread
    try:
        rclpy.spin(calibrator)
    except KeyboardInterrupt:
        print("Got KeyboardInterrupt, stopping.")
    stop_server()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
