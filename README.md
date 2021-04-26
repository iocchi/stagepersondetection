# Stage person detection

Dataset, models and app for Stage person detection

## Docker setup

    cd docker
    ./build.bash

Note: to build an image with a specific version of tensorflow or additional libraries and tools, follow these steps (do not edit ```Dockerfile``` directly)

    cd docker
    cp Dockerfile Dockerfile.<version>
    edit Dockerfile.<version>
    ./build.bash Dockerfile.<version> <version>

## Run

    cd docker
    ./run.bash

To run a specific image version

    ./run.bash <version>


## Use

Run these commands in the container.

* Train

      python stageperson_net.py -modelname <new_modelname> --train

    Example:

      python stageperson_net.py -modelname stageperson5_NEW --train

* Test


      python stageperson_net.py -modelname <saved_modelname> --test 

    Example:

      python stageperson_net.py -modelname stageperson5_v3 --test 


* Predict 

      python stageperson_net.py -modelname <saved_modelname> -predict <imagefile>

    Example:

      python stageperson_net.py -modelname stageperson5_v3 -predict dataset/test/red/20210425-220733-photo.jpg

* Start server mode

      python stageperson_net.py -modelname <saved_modelname> --server -server_port <server_port>

    Example:

      python stageperson_net.py -modelname stageperson5_v3 --server -server_port 9250

