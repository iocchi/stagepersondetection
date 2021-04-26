# Stage person detection

Dataset, models and app for Stage person detection

## Docker setup

    cd docker
    ./build.bash

## Run

    cd docker
    ./run.bash

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

