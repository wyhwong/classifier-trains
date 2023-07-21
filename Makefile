export DOCKER_BUILDKIT=1

dataset?=${PWD}/dataset
config?=${PWD}/configs/train.yml
outputDir?=${PWD}/results
shmSize?=8gb
version?=devel
loglevel?=20

build:
	mkdir -p ./results
	docker build -t local/TCPytorch:${version} \
				 --build-arg USERNAME=$(shell whoami) \
			     --build-arg USER_ID=$(shell id -u) \
			     --build-arg GROUP_ID=$(shell id -g) .

train:
	docker run --rm -it --name tcpytorch \
			   --gpus all \
			   -v ${outputDir}:/results \
			   -v ${dataset}:/dataset \
			   -v ${config}:/home/${USERNAME}/workspace/configs/train.yml \
			   --shm-size={shmSize} \
			   --env LOGLEVEL=${loglevel} \
			   local/TCPytorch:${version}
