FROM nvcr.io/nvidia/dli/dli-nano-ai:v2.0.2-r32.7.1
MAINTAINER eunaif

# Update packages and install required ones (segmentation-models, matplotlib...)
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install segmentation-models-pytorch
RUN python3 -m pip install matplotlib