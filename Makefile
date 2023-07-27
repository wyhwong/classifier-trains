export DOCKER_BUILDKIT=1

DATASET?=${PWD}/dataset
CONFIG?=${PWD}/configs/train.yml
OUTPUT_DIR?=${PWD}/results
VERSION?=devel
LOGLEVEL?=20
TZ?=Asia/Hong_Kong

build:
	mkdir -p ./results
	docker build -t tcpytorch:${VERSION} \
				 --build-arg USERNAME=$(shell whoami) \
				 --build-arg USER_ID=$(shell id -u) \
				 --build-arg GROUP_ID=$(shell id -g) \
				 --build-arg TZ=${TZ} .

train:
	docker run --rm -it --name tcpytorch \
			   --gpus all \
			   -v ${OUTPUT_DIR}:/results \
			   -v ${DATASET}:/dataset \
			   -v ${CONFIG}:/home/${USERNAME}/workspace/configs/train.yml \
			   -v ./:/home/${USERNAME}/workspace \
			   --env LOGLEVEL=${LOGLEVEL} \
			   tcpytorch:${VERSION}
