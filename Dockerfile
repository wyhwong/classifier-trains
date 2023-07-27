FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y tzdata ffmpeg libsm6 libxext6
RUN pip3 install pyyaml numpy matplotlib pandas seaborn opencv-python onnx scikit-learn

ARG USERNAME
ARG USER_ID
ARG GROUP_ID
ARG TZ
ENV TZ=${TZ}
RUN groupadd --gid ${GROUP_ID} ${USERNAME} && \
    adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}/workspace
CMD ["python3", "main.py"]
