export DOCKER_BUILDKIT=1

# For Docker mount
export DATA_DIR?=./tests/test_dataset
export CONFIG_PATH?=pipeline_setting.yml
export OUTPUT_DIR?=results

# For Docker image
export VERSION?=0.1.0
export USERNAME?=$(shell whoami)
export USER_ID?=$(shell id -u)
export GROUP_ID?=$(shell id -g)

# Fixed variables
export APP=classifier_trains
export UML_DIR=uml-diagram
export DOCKERFILE=docker/Dockerfile
export TEST_DIR=tests

# For test
export TEST_REPORT_DIR?=test_report

# Build Docker image
build:
	@echo "Building Docker image (MAIN)..."
	docker build \
		--build-arg VERSION=${VERSION} \
		--build-arg USERNAME=${USERNAME} \
		--build-arg USER_ID=${USER_ID} \
		--build-arg GROUP_ID=${GROUP_ID} \
		-t ${USERNAME}/${APP}:${VERSION} \
		-f ${DOCKERFILE} \
		--no-cache \
		--target main .
	@echo "Building Docker image (TEST)..."
	docker build \
		--build-arg VERSION=${VERSION} \
		--build-arg USERNAME=${USERNAME} \
		--build-arg USER_ID=${USER_ID} \
		--build-arg GROUP_ID=${GROUP_ID} \
		-t ${USERNAME}/${APP}-test:${VERSION} \
		-f ${DOCKERFILE} \
		--target test .

# Install Python dependencies in poetry environment
install:
	@echo "Installing Python dependencies..."
	poetry install
	poetry run pre-commit install -f

# Install Python dependencies in poetry environment
dev-install:
	@echo "Installing Python dependencies (dev mode)..."
	poetry install --with dev
	poetry run pre-commit install -f

run:
	@echo "Running pipeline..."
	poetry run python3 -m ${APP} run -c ${CONFIG_PATH}  -o ${OUTPUT_DIR}

profile:
	@echo "Running pipeline in profile mode..."
	poetry run python3 -m ${APP} profile -c ${CONFIG_PATH}  -o ${OUTPUT_DIR}

compute-mean-and-std:
	@echo "Computing mean and std..."
	poetry run python3 -m ${APP} compute-mean-and-std -d ${DATA_DIR}

get-output-mapping:
	@echo "Getting output mapping..."
	poetry run python3 -m ${APP} get-output-mapping -d ${DATA_DIR}

# Upate Python dependencies in poetry environment
update:
	@echo "Updating Python dependencies..."
	poetry update

# Format pipeline using black, isort
format:
	@echo "Formatting source code..."
	poetry run black ${APP}
	poetry run isort ${APP}
	poetry run black ${TEST_DIR}
	poetry run isort ${TEST_DIR}

# Run static code analysis
static-code-analysis:
	@echo "Running static code analysis..."
	poetry run black ${APP} --check
	poetry run bandit -r ${APP} -c pyproject.toml
	poetry run pylint ${APP} --rcfile ../.pylintrc
	poetry run isort ${APP} --check-only
	poetry run mypy ${APP} --show-error-codes
	poetry run radon cc ${APP}
	mkdir -p ${TEST_REPORT_DIR}
	poetry run pytest \
		--disable-warnings \
		--cov=${APP} .

# Generate test report
test-report:
	mkdir -p ${TEST_REPORT_DIR}
	poetry run pytest \
		--disable-warnings \
		--cov-report=html:${TEST_REPORT_DIR} \
		--cov=${APP} .
	open ${TEST_REPORT_DIR}/index.html

# Generate UML diagram
uml:
	@echo "Generating UML diagram..."
	mkdir -p ${UML_DIR}
	poetry run pyreverse ${APP} -o png -d ${UML_DIR}

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  build: Build Docker image"
	@echo "    - VERSION: Version of Docker image"
	@echo "    - USERNAME: Username of Docker image"
	@echo "    - USER_ID: User ID of Docker image"
	@echo "    - GROUP_ID: Group ID of Docker image"
	@echo "  install: Install Python dependencies in poetry environment"
	@echo "  dev-install: Install Python dependencies in poetry environment (dev mode)"
	@echo "  run: Run pipeline"
	@echo "    - CONFIG_PATH: Path to pipeline setting file"
	@echo "    - OUTPUT_DIR: Path to output directory"
	@echo "  profile: Run pipeline in profile mode"
	@echo "    - CONFIG_PATH: Path to pipeline setting file"
	@echo "    - OUTPUT_DIR: Path to output directory"
	@echo "  compute-mean-and-std: Compute mean and std"
	@echo "    - DATA_DIR: Path to dataset directory"
	@echo "  get-output-mapping: Get output mapping"
	@echo "    - DATA_DIR: Path to dataset directory"
	@echo "  update: Update Python dependencies in poetry environment"
	@echo "  format: Format pipeline using black, isort"
	@echo "  static-code-analysis: Run static code analysis"
	@echo "    - TEST_REPORT_DIR: Path to test report directory"
	@echo "  test-report: Generate test report"
	@echo "    - TEST_REPORT_DIR: Path to test report directory"
	@echo "  uml: Generate UML diagram"
	@echo "  help: Show this help message and exit"
