#!/usr/bin/env bash
set -e

IMAGE_NAME="ros2-humble:latest"
CONTAINER_NAME="ros2-humble-orbbec-ubuntu"
ORBBEC_WS="$HOME/dev/orbbec_ws"

docker run -it --rm \
  --name "${CONTAINER_NAME}" \
  --user $(id -u):$(id -g) \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  --network host \
  --ipc host \
  --privileged \
  -v /dev:/dev \
  -e ROS_DOMAIN_ID=30 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e ROS_LOCALHOST_ONLY=0 \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/orbbec_ws/config/fastdds_udp_only.xml \
  -e FASTDDS_DEFAULT_PROFILES_FILE=/workspaces/orbbec_ws/config/fastdds_udp_only.xml \
  -v "${ORBBEC_WS}:/workspaces/orbbec_ws" \
  -v "${ORBBEC_WS}/bags:/bags" \
  -v "${ORBBEC_WS}/data:/data" \
  -v "${ORBBEC_WS}/results:/results" \
  -w /workspaces/orbbec_ws \
  "${IMAGE_NAME}" \
  bash