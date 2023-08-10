export DOCKER_BUILDKIT=1

DATASET?=${PWD}/dataset
CONFIG?=${PWD}/configs/train.yml
OUTPUT_DIR?=${PWD}/results
VERSION?=devel
DEVICE?=cuda
LOGLEVEL?=20

TZ?=Asia/Hong_Kong
USERNAME?=$(shell whoami)
USER_ID?=$(shell id -u)
GROUP_ID?=$(shell id -g)

build:
	mkdir -p ./results
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
			   -v ${CONFIG}:/home/${USERNAME}/workspace/configs/train.yml \
			   -v ./:/home/${USERNAME}/workspace \
			   --env LOGLEVEL=${LOGLEVEL} \
			   --env DEVICE=${DEVICE} \
			   tcpytorch:${VERSION}
