#!/usr/bin/env python
import rclpy

from .online_calibrator import OnlineCalibrator
from .core import ws_sender


def main(args=None):
    ws_sender.main()  # start up websocket server in new Thread
    rclpy.init(args=args)
    calibrator = OnlineCalibrator()
    try:
        rclpy.spin(calibrator)
    except KeyboardInterrupt:
        print("Got KeyboardInterrupt, stopping.")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
