#!/bin/bash
set -e

# setup ros environment, this is copied from the entrypoint script of the normal ROS Dockerfile
source "/opt/ros/$ROS_DISTRO/setup.bash" --
# also, source the robosense sdk
source "/robosense/devel/setup.bash"

exec "$@"
