export DOCKER_BUILDKIT=1

DATASET?=./dataset
CONFIG?=./train.yml
OUTPUT_DIR?=./results
VERSION?=1.1.0
DEVICE?=cuda
LOGLEVEL?=20

TZ?=Asia/Hong_Kong
USERNAME?=$(shell whoami)
USER_ID?=$(shell id -u)
GROUP_ID?=$(shell id -g)

build:
	mkdir -p ${OUTPUT_DIR}
	docker build -t tcpytorch:${VERSION} \
				 --build-arg USERNAME=${USERNAME} \
				 --build-arg USER_ID=${USER_ID} \
				 --build-arg GROUP_ID=${GROUP_ID} \
				 --build-arg TZ=${TZ} .

train:
	docker run --rm -it --name tcpytorch \
			   --gpus all \
			   -v ${OUTPUT_DIR}:/results \
			   -v ${DATASET}:/dataset \
			   -v ${CONFIG}:/train.yml \
			   --env LOGLEVEL=${LOGLEVEL} \
			   --env DEVICE=${DEVICE} \
			   tcpytorch:${VERSION}
