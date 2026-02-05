#!/usr/bin/env bash
# Script to run, restart, or attach to a Docker container with GPU and X11 support
# This script intelligently handles container lifecycle: create, start, or attach as needed

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
IMAGE_TAG=$(whoami)_sonarsplat                      # Must match the tag used in build.sh
CONTAINER_NAME=sonarsplat_dev                       # Unique name for this container instance
CONTAINER_NAME="$CONTAINER_NAME_$(whoami)"          # Append username for uniqueness

# NVIDIA GPU capabilities (unused but kept for reference)
capabilities_str=\""capabilities=compute,utility,graphics,display\""

# Build Docker run options for GUI support, GPU access, and volume mounts
DOCKER_OPTIONS=""
DOCKER_OPTIONS+="-it "                                                                              # Interactive terminal
DOCKER_OPTIONS+="-e DISPLAY=$DISPLAY "                                                              # Forward X11 display
DOCKER_OPTIONS+="-v /tmp/.X11-unix:/tmp/.X11-unix "                                                 # Mount X11 socket for GUI apps
DOCKER_OPTIONS+="-v $SCRIPT_DIR/../:/home/$(whoami)/$(basename $(readlink -f $SCRIPT_DIR/../)) "    # Mount project directory
DOCKER_OPTIONS+="-v $HOME/.Xauthority:/home/$(whoami)/.Xauthority "                                 # X11 authentication
DOCKER_OPTIONS+="-v /etc/group:/etc/group:ro "                                                      # Mount group info (read-only) for proper permissions
DOCKER_OPTIONS+="-v /mnt/:/mnt/hostmnt "                                                            # Mount host's /mnt for accessing external drives
DOCKER_OPTIONS+="-v /frog-data:/frog-data "                                                         # Mount host's /mnt for accessing external drives
DOCKER_OPTIONS+="-v /frog-users:/frog-users "                                                         # Mount host's /mnt for accessing external drives
DOCKER_OPTIONS+="--name $CONTAINER_NAME "                                                           # Name the container for easy reference
DOCKER_OPTIONS+="--privileged "                                                                     # Grant extended privileges for device access
DOCKER_OPTIONS+="--gpus=all "                                                                       # Enable access to all GPUs
DOCKER_OPTIONS+="-e NVIDIA_DRIVER_CAPABILITIES=all "                                                # Enable all NVIDIA driver capabilities
DOCKER_OPTIONS+="--net=host "                                                                       # Use host network stack (simplifies networking)
DOCKER_OPTIONS+="--runtime=nvidia "                                                                 # Use NVIDIA container runtime
DOCKER_OPTIONS+="-e SDL_VIDEODRIVER=x11 "                                                           # Set SDL to use X11 for graphics
DOCKER_OPTIONS+="-u $(id -u):$(id -g) "                                                             # Run as current user/group for proper file permissions
DOCKER_OPTIONS+="--shm-size 32G "                                                                   # Increase shared memory for large data processing  # Increase shared memory for large data processing

# Mount all video devices (webcams, capture cards, etc.) into the container
for cam in /dev/video*; do
    DOCKER_OPTIONS+="--device=${cam} "
done

echo $CONTAINER_NAME

# Handle different container lifecycle scenarios based on arguments and state
if [ ${1:-""} == "restart" ]; then
    # Force restart: remove existing container and create a fresh one
    echo "Restarting Container"
    docker rm -f $CONTAINER_NAME
    docker run $DOCKER_OPTIONS $IMAGE_TAG /bin/bash
    # Check if container is currently running
    elif [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then                                      # Container isn't running
    
    # Check if container exists but is stopped
    if [  "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        # Container exists but is stopped - restart it
        echo "Resuming Container"
        docker start $CONTAINER_NAME
        docker exec -it $CONTAINER_NAME /entrypoint.sh
    else
        # Container doesn't exist - create and run a new one
        echo "Running Container"
        docker run $DOCKER_OPTIONS $IMAGE_TAG:latest
    fi
else
    # Container is already running - just attach to it
    echo "Attaching to existing container"
    docker exec -it $CONTAINER_NAME /entrypoint.sh
fi