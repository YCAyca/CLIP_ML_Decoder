#FROM continuumio/miniconda3
ARG CUDA="10.1"
ARG CUDNN="7"
ARG PYTORCH="1.5"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

WORKDIR CLIP_ML_Decoder

RUN apt-get -y update

RUN export PYTHONPATH="$PYTHONPATH:$PWD"
RUN pip install ftfy
RUN pip install regex
RUN pip install fitz
RUN pip install wilds
RUN pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
RUN pip install matplotlib
