#!/bin/bash

# General info about new container / image
CONTAINER_NAME=modulus/NorneRessim
TAG=latest

# Specs of the current user. These will be arguments to Dockerfile
THIS_UID=`id -u`
THIS_GID=`id -g`
THIS_USER=$USER

echo "Starting $CONTAINER_NAME:$TAG container..."
echo "User: $THIS_USER, UID: $THIS_UID, GID: $THIS_GID"

# Build a container
docker build \
--build-arg MYUID=$THIS_UID \
--build-arg MYGID=$THIS_GID \
--build-arg MYUSER=$THIS_USER \
-t $CONTAINER_NAME .

