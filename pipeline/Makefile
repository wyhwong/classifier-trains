export DOCKER_BUILDKIT=1

# For Docker mount
export DATA_DIR?=./dataset/
export CONFIG_PATH?=./setting.yml
export OUTPUT_DIR?=./results/

# For Docker image
export VERSION?=devel
export USERNAME?=$(shell whoami)
export USER_ID?=$(shell id -u)
export GROUP_ID?=$(shell id -g)

# Environment variables
export TZ?=Asia/Hong_Kong
export LOGLEVEL?=10
export DEVICE?=cuda
export RANDOM_SEED?=2024

# For Code analysis
export TEST_REPORT_DIR?=./test_report

# Build Docker image
build:
	mkdir -p ${OUTPUT_DIR}
	docker build -t classifier-pipeline:${VERSION} \
	--build-arg USERNAME=${USERNAME} \
	--build-arg USER_ID=${USER_ID} \
	--build-arg GROUP_ID=${GROUP_ID} \
	--build-arg TZ=${TZ} \
	--no-cache .

# Train classifier
train:
	mkdir -p ${OUTPUT_DIR}
	docker run --rm -it --name classifier-pipeline \
	--gpus all \
	-v ${OUTPUT_DIR}:/home/${USERNAME}/results/ \
	-v ${DATA_DIR}:/home/${USERNAME}/dataset/ \
	-v ${CONFIG_PATH}:/home/${USERNAME}/setting.yml \
	--env TZ=${TZ} \
	--env LOGLEVEL=${LOGLEVEL} \
	--env DEVICE=${DEVICE} \
	--env RANDOM_SEED=${RANDOM_SEED} \
	classifier-pipeline:${VERSION}

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
	poetry run radon cc src
	poetry run pytest --disable-warnings --cov-report=html:${TEST_REPORT_DIR} --cov=src/core src
	open ${TEST_REPORT_DIR}/index.html

# Upate Python dependencies in poetry environment
update:
	poetry update

# Format src using black, isort
format:
	poetry run black src
	poetry run isort src

# Run test cases in poetry environment and open coverage report
test:
	poetry run pytest --disable-warnings --cov-report=html:${TEST_REPORT_DIR} --cov=src/core src
	open ${TEST_REPORT_DIR}/index.html
