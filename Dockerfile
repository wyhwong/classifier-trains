FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
RUN TZ=Asia/Hong_Kong

RUN apt-get update && apt-get install -y tzdata
RUN pip3 install pyyaml overrides matplotlib opencv-python

WORKDIR /workspace
