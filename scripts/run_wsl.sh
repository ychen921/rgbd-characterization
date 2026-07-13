#!/usr/bin/env bash
set -e

IMAGE_NAME="ros2-humble:latest"
CONTAINER_NAME="ros2-humble-rgbd-wsl"
WS="$HOME/dev/rgbd-characterization"

docker run -it --rm \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  -e ROS_DOMAIN_ID=30 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e ROS_LOCALHOST_ONLY=0 \
  -v "${WS}:/workspaces/rgbd-characterization" \
  -v "${WS}/bags:/bags" \
  -v "${WS}/data:/data" \
  -v "${WS}/results:/results" \
  -w /workspaces/rgbd-characterization \
  "${IMAGE_NAME}" \
  bash
