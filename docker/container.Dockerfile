# Dockerfile for creating a development container with GPU support and user-space tooling
# This container is designed for robotics/computer vision development with NVIDIA GPU access

# Base image argument - typically an NVIDIA CUDA image for GPU support
ARG BASE_IMAGE=nvidia/cuda:12.6.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# User configuration arguments - these are passed from build.sh to match host user
ARG USER_NAME=<user-name>
ARG USER_ID=1000

# Prevent anything requiring user input during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

ENV TCNN_CUDA_ARCHITECTURES=86

# Set timezone to avoid prompts during package installation
ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install basic development and utility packages
RUN apt-get -y update \
    && apt-get -y install \
      python3-pip \
      sudo \
      vim \
      wget \
      curl \
      software-properties-common \
      doxygen \
      git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -y update && apt-get -y install \
    build-essential \
    ninja-build \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install computer vision and graphics libraries (OpenGL, rendering, 3D processing)
RUN apt-get -y update \
    && apt-get -y install \
        libglew-dev \
        libassimp-dev \
        libboost-all-dev \
        libgtk-3-dev \
        libglfw3-dev \
        libavdevice-dev \
        libavcodec-dev \
        libeigen3-dev \
        libxxf86vm-dev \
        libembree-dev \
    && rm -rf /var/lib/apt/lists/*

# Install build tools
RUN apt-get -y update \
    && apt-get -y install \ 
        cmake \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade "pip<25" setuptools
RUN python3 -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126
RUN python3 -m pip install --ignore-installed blinker \
    numpy matplotlib jupyterlab open3d trimesh scikit-learn scikit-image pandas plotly

RUN python3 -m pip install --no-build-isolation \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV TCNN_CUDA_ARCHITECTURES=86
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"

RUN python3 -m pip install --no-build-isolation \
    git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e \
    git+https://github.com/nerfstudio-project/nerfacc \
    git+https://github.com/harry7557558/fused-bilagrid@90f9788e57d3545e3a033c1038bb9986549632fe \
    git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157

# install from the requirements.txt that is in this folder
COPY ./requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Create a non-root user matching the host user ID for seamless file permissions
# This prevents permission issues when accessing mounted volumes
RUN useradd -m -l -u ${USER_ID} -s /bin/bash ${USER_NAME} \
    && usermod -aG video ${USER_NAME} \
    && export PATH=$PATH:/home/${USER_NAME}/.local/bin

# Grant passwordless sudo access to the user for convenience
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to non-root user for all subsequent commands and container runtime
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

# Ensure the user owns their home directory
RUN sudo chown -R ${USER_NAME} /home/${USER_NAME}

# Copy and set up the entrypoint script that runs when the container starts
COPY ./entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]