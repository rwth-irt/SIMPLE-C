#!/usr/bin/env python
import rclpy

from .online_calibrator import OnlineCalibrator


def main(args=None):
    rclpy.init(args=args)
    calibrator = OnlineCalibrator()
    try:
        rclpy.spin(calibrator)
    except KeyboardInterrupt:
        print("Got KeyboardInterrupt, stopping.")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
