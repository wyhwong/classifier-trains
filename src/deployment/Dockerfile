# STAGE 1
FROM python:3.11-slim-buster AS base

# Convert poetry.lock to requirements.txt
RUN pip3 install poetry
COPY ./pyproject.toml ./poetry.lock /
RUN poetry export -f requirements.txt -o requirements.txt \
    --without-hashes \
    --without torch \
    --without dev
RUN poetry export -f requirements.txt -o requirements-dev.txt \
    --without-hashes \
    --only dev

# STAGE 2
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime AS main

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
COPY ./pipeline /home/${USERNAME}/pipeline
WORKDIR /home/${USERNAME}/pipeline

CMD ["python3", "main.py"]

# [Optional] Setup test environment
FROM main AS test

COPY --from=base /requirements-dev.txt /tmp/requirements-dev.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-dev.txt
COPY ./tests/ /home/${USERNAME}/tests/
