FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
ENV TZ=Asia/Hong_Kong

RUN apt-get update && apt-get install -y tzdata ffmpeg libsm6 libxext6
RUN pip3 install pyyaml matplotlib pandas seaborn opencv-python onnx

ARG USERNAME
ARG USER_ID
ARG GROUP_ID
RUN groupadd --gid ${GROUP_ID} ${USERNAME} && \
    adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ${USERNAME}

USER ${USERNAME}
COPY . /home/${USERNAME}/workspace
WORKDIR /home/${USERNAME}/workspace
CMD ["python3", "main.py"]
