#!/bin/bash

UNAME=tc
DOCKER_BUILDKIT=1 docker build . -t tc-temp --build-arg UID=$(id -u) --build-arg UNAME=$UNAME
WORKDIR=/home/$UNAME

docker run \
    -p 5000:5000 \
    -v "$(pwd)/database:/app/database" \
    lucasspangher/social_game_api_image:latesp
