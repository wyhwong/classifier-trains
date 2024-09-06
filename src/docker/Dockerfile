# STAGE 1 - Build requirements.txt
FROM python:3.11-slim-buster AS base

# Convert poetry.lock to requirements.txt
RUN pip3 install poetry
COPY ./pyproject.toml ./poetry.lock /
RUN poetry export -o requirements.txt \
    -f requirements.txt \
    --without-hashes \
    --without torch \
    --without dev && \
    poetry export -o requirements-dev.txt \
    -f requirements.txt \
    --without-hashes \
    --only dev

# STAGE 2 - Build main image
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime AS main

# Install system dependencies
RUN apt-get update && \
    apt-get install -y tzdata ffmpeg libsm6 libxext6 && \
    apt-get clean

# Create user
ARG USERNAME
ARG USER_ID
ARG GROUP_ID
ARG TZ
ENV TZ=${TZ}
RUN groupadd --gid ${GROUP_ID} ${USERNAME} && \
    adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ${USERNAME}
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Install dependencies from requirements.txt
COPY --chown=${USERNAME} --from=base /requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY --chown=${USERNAME} ./classifier_trains/ /usr/local/lib/classifier_trains/
ENV PYTHONPATH=/usr/local/lib

# # [Optional] Setup test environment
FROM main AS test

COPY --chown=${USERNAME} --from=base /requirements-dev.txt /tmp/requirements-dev.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-dev.txt
COPY ./tests/ /home/${USERNAME}/tests/
