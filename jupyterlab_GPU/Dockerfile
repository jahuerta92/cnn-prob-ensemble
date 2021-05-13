FROM tensorflow/tensorflow:latest-gpu
FROM vrodriguezf/jupyterlab-cuda:latest

# INSTALL CUDNN8
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8 fonts-powerline \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*
    
# PYTHON PACKAGES with pip 
RUN pip3 install scikit-learn fastgpu nbdev pandas transformers tensorflow-addons \
     tensorflow torch pymongo torch emoji python-dotenv 

RUN apt update

RUN apt -y install zsh wget sudo nano

# Defining ARG variables passed from docker-compose
ARG UID
ARG GID
ARG USER

RUN echo ${UID}

RUN echo ${GID}

RUN echo ${USER}

RUN addgroup --gid $GID $USER 
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $USER 

# Giving root permissions to $USER
RUN adduser $USER sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER $USER

WORKDIR /home/$USER/work

ENV SHELL=/bin/zsh
