FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y tzdata ffmpeg libsm6 libxext6
RUN pip3 install pyyaml==6.0 numpy==1.22.3 pandas==2.0.3 matplotlib==3.7.2 \
    seaborn==0.12.2 opencv-python==4.8.0.76 onnx==1.14.1 scikit-learn==1.3.0

ARG USERNAME
ARG USER_ID
ARG GROUP_ID
ARG TZ
ENV TZ=${TZ}
RUN groupadd --gid ${GROUP_ID} ${USERNAME} && \
    adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ${USERNAME}

USER ${USERNAME}
COPY ./TCPyTorch /home/${USERNAME}/workspace
WORKDIR /home/${USERNAME}/workspace
CMD ["python3", "main.py"]
