#!/bin/bash

# Use  ./run.bash [version]

IMAGENAME=stagepersondetection

VERSION=latest
if [ ! "$1" == "" ]; then
  VERSION=$1
fi

SPD_DIR=`pwd | gawk '{ print gensub(/\/docker/, "", 1) }'`

echo "Running image $IMAGENAME:$VERSION ..."

docker run -it \
    --name stagepersondetection --rm \
    --privileged \
    --net=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:/home/robot/.Xauthority:rw \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v $SPD_DIR:/home/robot/src/stageperson_detection \
    $IMAGENAME:$VERSION

