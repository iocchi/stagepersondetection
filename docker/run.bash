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
    -v $SPD_DIR:/home/robot/src/stageperson_detection \
    -v $HOME/playground/images:/home/robot/images \
    $IMAGENAME:$VERSION

