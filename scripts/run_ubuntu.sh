#!/usr/bin/env bash
set -e

IMAGE_NAME="ros2-humble:latest"
CONTAINER_NAME="rgbd-characterization-ubuntu"
RGBD_CHARACTERIZATION_ROOT="${RGBD_CHARACTERIZATION_ROOT:-$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
)}"
CONTAINER_WORKSPACE="/workspaces/rgbd-characterization"
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
  -e FASTRTPS_DEFAULT_PROFILES_FILE="${CONTAINER_WORKSPACE}/config/fastdds_udp_only.xml" \
  -e FASTDDS_DEFAULT_PROFILES_FILE="${CONTAINER_WORKSPACE}/config/fastdds_udp_only.xml" \
  -v "${RGBD_CHARACTERIZATION_ROOT}:${CONTAINER_WORKSPACE}" \
  -v "${RGBD_CHARACTERIZATION_ROOT}/bags:/bags" \
  -v "${RGBD_CHARACTERIZATION_ROOT}/data:/data" \
  -v "${RGBD_CHARACTERIZATION_ROOT}/results:/results" \
  -v "${CONTAINER_HOME}:/home/${HOST_USER}" \
  -w "${CONTAINER_WORKSPACE}" \
  "${IMAGE_NAME}" \
  bash
