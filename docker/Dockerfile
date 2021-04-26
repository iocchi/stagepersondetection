# Docker file for Stage person detection

FROM tensorflow/tensorflow

#
# USER root
#

USER root

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y -qq --no-install-recommends \
        tmux sudo nano htop wget less \
        iputils-ping net-tools \
        cmake g++ git \
        python3-dev python3-pip python3-tk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Create user: robot (password: robot) with sudo power

RUN useradd -ms /bin/bash robot && echo "robot:robot" | chpasswd && adduser robot sudo

# Python modules

RUN pip install \
    numpy pandas scipy matplotlib sklearn keras \
    jupyter notebook

#
# USER robot
#

USER robot

# Configuration

RUN echo "set -g mouse on" > $HOME/.tmux.conf 

WORKDIR /home/robot/src/stageperson_detection

CMD ["/usr/bin/tmux" ]

