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
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Isolated Python environment for project tools
#
# Keep project packages separate from Ubuntu Python extensions compiled
# against the NumPy 1.x ABI. ROS Python packages remain available through
# the paths configured by /opt/ros/humble/setup.bash.
# ------------------------------------------------------------
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python3 -m venv "${VIRTUAL_ENV}" && \
    "${VIRTUAL_ENV}/bin/python3" -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    "${VIRTUAL_ENV}/bin/python3" -m pip install --no-cache-dir \
        "numpy==1.26.4" \
        "matplotlib==3.8.4" \
        "opencv-python==4.10.0.84" \
        "scipy==1.13.1" \
        "scikit-learn==1.5.2" \
        "pandas==2.2.3" \
        "open3d==0.18.0" \
        "PyYAML==6.0.2" \
        "tqdm==4.67.1" && \
    "${VIRTUAL_ENV}/bin/python3" -c "import cv2, matplotlib, numpy; print(numpy.__version__, matplotlib.__version__, cv2.__version__)"

# ------------------------------------------------------------
# Shell setup
# ------------------------------------------------------------
SHELL ["/bin/bash", "-c"]

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-30}" >> /root/.bashrc && \
    echo "export ROS_LOCALHOST_ONLY=\${ROS_LOCALHOST_ONLY:-0}" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=\${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}" >> /root/.bashrc

CMD ["bash"]
