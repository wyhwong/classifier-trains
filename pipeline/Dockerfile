# STAGE 1
FROM python:3.11-slim-buster AS base

# Convert poetry.lock to requirements.txt
RUN pip3 install poetry
COPY ./pyproject.toml ./poetry.lock /
RUN poetry export -f requirements.txt -o requirements.txt --without-hashes --without dev

# STAGE 2
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS last

RUN apt-get update && \
    apt-get install -y tzdata ffmpeg libsm6 libxext6 && \
    apt-get clean

# Install dependencies from requirements.txt
COPY --from=base /requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

# Create user
ARG USERNAME
ARG USER_ID
ARG GROUP_ID
ARG TZ
ENV TZ=${TZ}
RUN groupadd --gid ${GROUP_ID} ${USERNAME} && \
    adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ${USERNAME}
USER ${USERNAME}

# Set working directory and freeze scripts
COPY ./src /home/${USERNAME}/src
WORKDIR /home/${USERNAME}/src

CMD ["python3", "main.py"]
