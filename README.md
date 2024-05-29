# TraBaDyFiCa

> Trajectory Based Dynamic Filtering for Calibration

## Running in Docker

To run in a docker container with all dependencies installed, the package built and everything properly sourced
(this is the case when running inside the docker container built from the definitions in the
[docker repository](https://git-ce.rwth-aachen.de/g-nav-mob-irt/projects/galileonautic2plus/calibration/robosense_docker),
run the following command:

`ros2 run lidar_align main --ros-args --params-file /PATH/TO/parameters.yaml -p sensor_pairs:="<PAIRS>"`

Make sure that the parameter file is accessible inside the docker container for ROS to read it.

Replace `<PAIRS>` with a sensor pair definition using the following syntax: `topicA,topicB;topicB,topicC`.
Pairs are separated using semicolons, in a single pair, the two sensors are comma-separated. **There must be no
spaces in this definition.**