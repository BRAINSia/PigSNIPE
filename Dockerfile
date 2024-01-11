#Download base image ubuntu 20.04
FROM ubuntu:20.04

# LABEL about the custom image
LABEL maintainer="michal-brzus@uiowa.edu"
LABEL version="0.1"
LABEL description="This is custom Docker Image for Minipig Image Processing"

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
RUN apt update

# Install needed packages from ubuntu repository
RUN apt install -y python3.8-venv python3-pip python-is-python3 && \
    rm -rf /var/lib/apt/lists/* && \
    apt clean

RUN mkdir -p /opt
# move model parameters to correct directories
COPY DL_MODEL_PARAMS/gwc_model.ckpt                /opt/src/segmentation/model_Params/gwc_model.ckpt 
COPY DL_MODEL_PARAMS/high_res_brainmask_model.ckpt /opt/src/segmentation/model_Params/high_res_brainmask_model.ckpt 
COPY DL_MODEL_PARAMS/icv_model.ckpt                /opt/src/segmentation/model_Params/icv_model.ckpt 
COPY DL_MODEL_PARAMS/low_res_brainmask_model.ckpt  /opt/src/segmentation/model_Params/low_res_brainmask_model.ckpt 
COPY DL_MODEL_PARAMS/seg_model.ckpt                /opt/src/segmentation/model_Params/seg_model.ckpt 
COPY DL_MODEL_PARAMS/primary_lmk_05mm_model.pt     /opt/src/landmarks/RL_Params/prim_05mm/model.pt 
COPY DL_MODEL_PARAMS/primary_lmk_1mm_model.pt      /opt/src/landmarks/RL_Params/prim_1mm/model.pt
COPY DL_MODEL_PARAMS/secondary_lmk_model.pt        /opt/src/landmarks/RL_Params/sec/model.pt
COPY DL_MODEL_PARAMS/tertiary_lmk_model.pt         /opt/src/landmarks/RL_Params/ter/model.pt

COPY BRAINSToolsBinaries /opt/BRAINSToolsBinaries
COPY REQUIREMENTS.txt /temp/REQUIREMENTS.txt

ENV PATH $PATH:/opt/BRAINSToolsBinaries
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/BRAINSToolsBinaries

RUN pip install -U pip \
        && pip install -r /temp/REQUIREMENTS.txt --no-cache-dir \
        && rm /temp/REQUIREMENTS.txt

COPY src /opt/src
COPY pigsnipe /opt/pigsnipe

ENTRYPOINT ["/opt/pigsnipe"]
CMD ["/opt/pigsnipe", "--help"]
