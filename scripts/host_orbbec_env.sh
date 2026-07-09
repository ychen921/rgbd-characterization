#!/usr/bin/env bash

source /opt/ros/humble/setup.bash

export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0

export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/dev/orbbec_ws/config/fastdds_udp_only.xml
export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/dev/orbbec_ws/config/fastdds_udp_only.xml