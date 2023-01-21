export DOCKER_BUILDKIT=1

dataset ?= ${PWD}/dataset
config ?= ${PWD}/config/config.yml

build:
	docker build -t wyhwong/tao-like-pytorch-classifier .

train:
	docker run --rm -it --name Pytorch-classifier-training \
			   --gpus all \
			   -v ${dataset}:/dataset \
			   -v ${config}:/workspace/config/config.yml \
			   --env-file .env \
			   wyhwong/tao-like-pytorch-classifier
