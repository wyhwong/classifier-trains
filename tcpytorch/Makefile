export DOCKER_BUILDKIT=1

# For Docker mount
export DATA_DIR?=./dataset
export CONFIG_PATH?=./train.yml
export OUTPUT_DIR?=./results

# For Docker image
export VERSION?=devel
export USERNAME?=$(shell whoami)
export USER_ID?=$(shell id -u)
export GROUP_ID?=$(shell id -g)

# Environment variables
export TZ?=Asia/Hong_Kong
export LOGLEVEL?=10
export DEVICE?=cuda

# Build Docker image
build:
	mkdir -p ${OUTPUT_DIR}
	docker build -t tcpytorch:${VERSION} \
	--build-arg USERNAME=${USERNAME} \
	--build-arg USER_ID=${USER_ID} \
	--build-arg GROUP_ID=${GROUP_ID} \
	--build-arg TZ=${TZ} \
	--no-cache .

# Train classifier
train:
	docker run --rm -it --name tcpytorch \
	--gpus all \
	-v ${OUTPUT_DIR}:/results \
	-v ${DATASET}:/dataset \
	-v ${CONFIG}:/train.yml \
	--env TZ=${TZ} \
	--env LOGLEVEL=${LOGLEVEL} \
	--env DEVICE=${DEVICE} \
	tcpytorch:${VERSION}

# Install Python dependencies in poetry environment
local:
	poetry install
	poetry run pre-commit install -f

# Run code analysis
analyze:
	poetry install
	poetry run pre-commit install -f
	poetry run black src --check
	poetry run bandit -r src -c pyproject.toml
	poetry run pylint src --rcfile ../.pylintrc
	poetry run isort src --check-only
	poetry run mypy src --show-error-codes

# Upate Python dependencies in poetry environment
update:
	poetry update

# Format src using black, isort
format:
	poetry run black src
	poetry run isort src