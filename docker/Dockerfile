FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV ROS_DOMAIN_ID=30
ENV ROS_LOCALHOST_ONLY=0

# ------------------------------------------------------------
# Base system tools + ROS2 + OrbbecSDK_ROS2 dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic dev tools
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    tree \
    tmux \
    less \
    unzip \
    zip \
    pkg-config \
    software-properties-common \
    # Network / USB / debug tools
    iproute2 \
    net-tools \
    usbutils \
    udev \
    mesa-utils \
    libgl1 \
    # Python / colcon
    python3-pip \
    python3-venv \
    python3-argcomplete \
    python3-colcon-common-extensions \
    # OrbbecSDK_ROS2 common dependencies
    libgflags-dev \
    libgoogle-glog-dev \
    nlohmann-json3-dev \
    libdw-dev \
    libssl-dev \
    libusb-1.0-0-dev \
    # ROS2 core
    ros-humble-rclcpp \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs \
    ros-humble-visualization-msgs \
    ros-humble-diagnostic-msgs \
    ros-humble-statistics-msgs \
    # ROS2 TF
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-tf2-sensor-msgs \
    ros-humble-tf2-geometry-msgs \
    # ROS2 image / camera
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-compressed-image-transport \
    ros-humble-image-publisher \
    ros-humble-camera-info-manager \
    # ROS2 diagnostics / launch / xacro
    ros-humble-diagnostic-updater \
    ros-humble-xacro \
    ros-humble-backward-ros \
    # ROS2 rosbag
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-default-plugins \
    ros-humble-rosbag2-compression-zstd \
    ros-humble-rosbag2-storage-mcap \
    # ROS2 PCL
    ros-humble-pcl-ros \
    ros-humble-pcl-conversions \
    ros-humble-message-filters \
    # DDS
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rmw-fastrtps-cpp \
    # Demo nodes for communication test
    ros-humble-demo-nodes-py \
    ros-humble-demo-nodes-cpp \
    # OpenCV / PCL C++ / Python
    libopencv-dev \
    libpcl-dev \
    pcl-tools \
    python3-opencv \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-yaml \
    python3-tqdm \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python perception / data analysis packages
# ------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    scikit-learn \
    pandas \
    open3d

# ------------------------------------------------------------
# Shell setup
# ------------------------------------------------------------
SHELL ["/bin/bash", "-c"]

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-30}" >> /root/.bashrc && \
    echo "export ROS_LOCALHOST_ONLY=\${ROS_LOCALHOST_ONLY:-0}" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=\${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}" >> /root/.bashrc

# ------------------------------------------------------------
# Workspace
# ------------------------------------------------------------
WORKDIR /workspaces/orbbec_ws

CMD ["bash"]