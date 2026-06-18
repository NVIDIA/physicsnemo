#!/bin/bash

# General info about new container / image
CONTAINER_NAME=modulus/ressim
CONTAINER_SHORTCUT_NAME=ressim
SUBDIR_NAME=project
PORT_HOST=8890
TAG=latest

# Start container. Note that sudo is not necessary, but 
# the user won't have root rights anyway
# --rm \

#xhost +local:root

docker run \
    -it \
    --rm \
    --runtime=nvidia \
    --privileged \
    --gpus all \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="$HOME/.Xauthority:/home/developer/.Xauthority:rw" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v /dev:/dev \
    -p ${PORT_HOST}:8888 \
    -v ${PWD}:/workspace/${SUBDIR_NAME} \
    --name ${CONTAINER_SHORTCUT_NAME} \
    ${CONTAINER_NAME}:${TAG}
