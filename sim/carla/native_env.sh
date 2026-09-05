#!/usr/bin/env bash
# Common environment for CARLA 0.9.16 native ROS 2 on Ubuntu 22.04.
# CARLA 0.9.16 native ROS 2 uses Fast DDS; keep every ROS terminal identical.
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# Work around Fast DDS shared-memory lock/stale-port failures seen after crashes.
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
