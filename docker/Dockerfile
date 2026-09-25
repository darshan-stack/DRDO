# syntax=docker/dockerfile:1.7

FROM ros:humble-ros-base-jammy

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ENV FFEM_TORCH_THREADS=4
ENV FFEM_CHECKPOINT=/models/checkpoints/semanticposs_range_model.pt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3-pip \
       python3-dev \
       build-essential \
       git \
       curl \
       ca-certificates \
       libgl1 \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ffem

COPY pyproject.toml README.md ./
COPY src ./src
COPY ros2/package.xml ros2/setup.py ros2/setup.cfg ros2/resource ./ros2/
COPY ros2/ffem_lidar_mapping ./ros2/ffem_lidar_mapping
COPY ros2/launch ./ros2/launch
COPY ros2/rviz ./ros2/rviz

COPY scripts ./scripts
COPY docs ./docs
COPY sim/carla/native_env.sh ./sim/carla/native_env.sh

RUN python3 -m pip install --no-cache-dir --break-system-packages -U pip setuptools wheel \
    && python3 -m pip install --no-cache-dir --break-system-packages "numpy>=1.26" "pyyaml>=6.0" "rerun-sdk>=0.22" "pillow>=10.0" \
    && python3 -m pip install --no-cache-dir --break-system-packages "torch>=2.2"

RUN source /opt/ros/humble/setup.bash \
    && colcon build --symlink-install --base-paths /opt/ffem/ros2 --build-base /opt/ffem/build --install-base /opt/ffem/install \
    && rm -rf /opt/ffem/build /opt/ffem/log

RUN mkdir -p /models/checkpoints /opt/ffem/outputs

COPY docker/entrypoint.sh /usr/local/bin/ffem-entrypoint
RUN chmod +x /usr/local/bin/ffem-entrypoint

ENTRYPOINT ["/usr/local/bin/ffem-entrypoint"]
CMD ["ros2", "launch", "ffem_lidar_mapping", "ffem_integrated.launch.py", "enable_rviz:=false"]
