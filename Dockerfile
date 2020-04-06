FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

USER root

RUN apt update
RUN apt install -y python3
RUN apt install -y python3-pip
RUN apt-get install -y python3-dev

RUN pip3 install scipy pymongo paramiko scp opencv-python seaborn scikit-image keras-vis pillow jupyter keras tensorflow-gpu sklearn 

RUN pip3 install jupyterlab imblearn plotly ipywidgets

RUN ln -s /usr/local/cuda-10.0 /usr/local/nvidia

RUN apt-get install -y screen zsh wget git

RUN sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

#RUN chsh -s $(which zsh)

RUN apt-get update

RUN apt-get install -y npm


RUN apt install -y curl
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
RUN apt install -y nodejs

RUN npm init -y

# npm install file-loader url-loader --saveDev


# user id for jupyter
ARG user_id=1000

# MAKE JUPYTER USER
RUN useradd -ms /bin/bash user
RUN usermod -u $user_id user
RUN groupmod -g $user_id user

USER root


#Plotly
RUN export NODE_OPTIONS=--max-old-space-size=4096
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager@2.0
#RUN jupyter labextension install jupyterlab-plotly
#RUN jupyter labextension install plotlywidget --no-build
RUN jupyter lab build --dev-build=False --minimize=False
RUN unset NODE_OPTIONS

RUN pip3 install pandoc pydot graphviz

RUN apt install -y unzip graphviz ruby-graphviz python-pydot python3-pydot python-pygraphviz python3-pygraphviz

RUN pip3 install --upgrade pip

RUN pip3 install tensorflow-gpu==1.15

USER user
WORKDIR /home/user

# MAKE DEAFULT CONFIG
RUN jupyter notebook --generate-config
RUN mkdir host_data

RUN python3.6 -m ipykernel install --user
