export DOCKER_BUILDKIT=1

# Load environment variables (for app)
export ENV_FILE?=.pipeline-env
include ${ENV_FILE}

# For Docker mount
export DATA_DIR?=dataset/
export CONFIG_PATH?=pipeline_setting.yml
export OUTPUT_DIR?=results/

# For Docker image
export VERSION?=0.1.0
export USERNAME?=$(shell whoami)
export USER_ID?=$(shell id -u)
export GROUP_ID?=$(shell id -g)
export APP_DIR?=pipeline
export UML_DIR?=uml-diagram
export DOCKERFILE?=deployment/Dockerfile

# For Code analysis
export TEST_DIR?=tests
export TEST_REPORT_DIR?=test_report

# Build Docker image
build:
	mkdir -p ${OUTPUT_DIR}
	@echo "Building Docker image (MAIN)..."
	docker build \
		--build-arg VERSION=${VERSION} \
		--build-arg USERNAME=${USERNAME} \
		--build-arg USER_ID=${USER_ID} \
		--build-arg GROUP_ID=${GROUP_ID} \
		-t ${USERNAME}/${APP_DIR}:${VERSION} \
		-f ${DOCKERFILE} \
		--no-cache \
		--target main .
	@echo "Building Docker image (TEST)..."
	docker build \
		--build-arg VERSION=${VERSION} \
		--build-arg USERNAME=${USERNAME} \
		--build-arg USER_ID=${USER_ID} \
		--build-arg GROUP_ID=${GROUP_ID} \
		-t ${USERNAME}/${APP_DIR}-test:${VERSION} \
		-f ${DOCKERFILE} \
		--target test .

# Train classifier
train:
	@echo "Training classifier..."
	mkdir -p ${OUTPUT_DIR}
	docker run --rm -it --name ${USERNAME}-${APP_DIR} \
		--gpus all \
		-v ${OUTPUT_DIR}:/home/${USERNAME}/results/ \
		-v ${DATA_DIR}:/home/${USERNAME}/dataset/ \
		-v ${CONFIG_PATH}:/home/${USERNAME}/pipeline_setting.yml \
		--env-file ${ENV_FILE} \
		${USERNAME}/${APP_DIR}:${VERSION}

clean:
	@echo "Removing Docker container..."
	docker stop ${USERNAME}-${APP_DIR}

# Install Python dependencies in poetry environment
install:
	@echo "Installing Python dependencies..."
	poetry install
	poetry run pre-commit install -f

run:
	@echo "Running pipeline..."
	poetry run python3 -m ${APP_DIR} run -c ${CONFIG_PATH}  -o ${OUTPUT_DIR}

compute-mean-and-std:
	@echo "Computing mean and std..."
	poetry run python3 -m ${APP_DIR} compute-mean-and-std -d ${DATA_DIR}

get-output-mapping:
	@echo "Getting output mapping..."
	poetry run python3 -m ${APP_DIR} get-output-mapping -d ${DATA_DIR}

# Upate Python dependencies in poetry environment
update:
	@echo "Updating Python dependencies..."
	poetry update

# Format pipeline using black, isort
format:
	@echo "Formatting source code..."
	poetry run black ${APP_DIR}
	poetry run isort ${APP_DIR}
	poetry run black ${TEST_DIR}
	poetry run isort ${TEST_DIR}

# Run test cases in poetry environment and open coverage report
test:
	@echo "Running test cases..."
	poetry run pytest \
		--disable-warnings \
		--cov-report=html:${TEST_REPORT_DIR} \
		--cov=${APP_DIR} .

# Run static code analysis
static-code-analysis:
	@echo "Running static code analysis..."
	poetry run black ${APP_DIR} --check
	poetry run bandit -r ${APP_DIR} -c pyproject.toml
	poetry run pylint ${APP_DIR} --rcfile ../.pylintrc
	poetry run isort ${APP_DIR} --check-only
	poetry run mypy ${APP_DIR} --show-error-codes
	poetry run radon cc ${APP_DIR}
	mkdir -p ${TEST_REPORT_DIR}
	poetry run pytest \
		--disable-warnings \
		--cov=${APP_DIR} .
	rmdir ${TEST_REPORT_DIR}

# Generate test report
test-report:
	mkdir -p ${TEST_REPORT_DIR}
	poetry run pytest \
		--disable-warnings \
		--cov-report=html:${TEST_REPORT_DIR} \
		--cov=${APP_DIR} .
	open ${TEST_REPORT_DIR}/index.html

# Generate UML diagram
uml:
	@echo "Generating UML diagram..."
	mkdir -p ${UML_DIR}
	poetry run pyreverse ${APP_DIR} -o png -d ${UML_DIR}

# Show environment variables
app-info:
	@echo "Showing environment variables (used in the project)..."
	@while read -r line; do \
		echo $$line; \
	done < ${ENV_FILE}

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  build: Build Docker image"
	@echo "  train: Train classifier"
	@echo "  clean: Remove Docker container"
	@echo "  install: Install Python dependencies in poetry environment"
	@echo "  update: Update Python dependencies in poetry environment"
	@echo "  format: Format pipeline using black, isort"
	@echo "  test: Run test cases in poetry environment and open coverage report"
	@echo "  static-code-analysis: Run static code analysis"
	@echo "  test-report: Generate test report"
	@echo "  uml-diagram: Generate UML diagram"
	@echo "  app-info: Show environment variables"
	@echo "  help: Show this help message"
