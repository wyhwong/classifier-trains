export DOCKER_BUILDKIT=1

build:
	docker build -t wyhwong/tao-like-pytorch-classifier .

train:
	docker run --rm --name Pytorch-classifier-training wyhwong/tao-like-pytorch-classifier
