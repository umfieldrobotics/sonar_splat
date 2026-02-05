#!/usr/bin/env bash
# Script to build the Docker image with proper user configuration
# This passes the current user's ID and name to avoid permission issues with mounted volumes

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Create a unique image tag prefixed with the current username
IMAGE_TAG="$(whoami)_sonarsplat"

# Build Docker options string
DOCKER_OPTIONS=""
DOCKER_OPTIONS+="-t $IMAGE_TAG:latest "                                           # Tag the image
DOCKER_OPTIONS+="-f $SCRIPT_DIR/container.Dockerfile "                             # Specify Dockerfile location
DOCKER_OPTIONS+="--build-arg USER_ID=$(id -u) --build-arg USER_NAME=$(whoami) "  # Pass host user info

# Construct and execute the docker build command
DOCKER_CMD="docker build $DOCKER_OPTIONS $SCRIPT_DIR"
echo $DOCKER_CMD
exec $DOCKER_CMD                                                                   # Replace this shell with the docker build process