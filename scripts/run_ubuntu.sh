#!/usr/bin/env bash
set -e

IMAGE_NAME="ros2-humble:latest"
CONTAINER_NAME="ros2-humble-orbbec-ubuntu"
WS="$HOME/dev/rgbd-characterization"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_USER="$(id -un)"
CONTAINER_HOME="$HOME/.docker-homes/${CONTAINER_NAME}"

mkdir -p "${CONTAINER_HOME}"

docker run -it --rm \
  --name "${CONTAINER_NAME}" \
  --user "${HOST_UID}:${HOST_GID}" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  --network host \
  --ipc host \
  --privileged \
  -v /dev:/dev \
  -e DISPLAY="${DISPLAY}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e USER="${HOST_USER}" \
  -e LOGNAME="${HOST_USER}" \
  -e HOME="/home/${HOST_USER}" \
  -e ROS_DOMAIN_ID=30 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e ROS_LOCALHOST_ONLY=0 \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/orbbec_ws/config/fastdds_udp_only.xml \
  -e FASTDDS_DEFAULT_PROFILES_FILE=/workspaces/orbbec_ws/config/fastdds_udp_only.xml \
  -v "${WS}:/workspaces/orbbec_ws" \
  -v "${WS}/bags:/bags" \
  -v "${WS}/data:/data" \
  -v "${WS}/results:/results" \
  -v "${CONTAINER_HOME}:/home/${HOST_USER}" \
  -w /workspaces/orbbec_ws \
  "${IMAGE_NAME}" \
  bash
