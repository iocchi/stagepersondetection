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

* Client

Send string message 

      EVAL <imagefile>

and get return string with prediction

      <class> <probability>

Example:

      echo "EVAL dataset/test/yellow/20210425-201650-photo.jpg" | netcat -w 3 localhost 9250
      yellow 1.000


Note: image file must be accessible from the docker container running the server. Make sure to share some volume between the image acquisition process and the Stage person detection server.



