#!/usr/bin/env bash
set -e

IMAGE_NAME="ros2-humble:latest"
CONTAINER_NAME="ros2-humble-orbbec-wsl"
WS="$HOME/dev/rgbd-characterization"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_USER="$(id -un)"
HOST_HOSTNAME="$(hostname)"

docker run -it --rm \
  --name "${CONTAINER_NAME}" \
  --hostname "${HOST_HOSTNAME}" \
  --user "${HOST_UID}:${HOST_GID}" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  --network host \
  --ipc host \
  -e USER="${HOST_USER}" \
  -e LOGNAME="${HOST_USER}" \
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
