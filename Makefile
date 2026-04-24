# Makefile to build the project

# Set the global variables
IMAGE_NAME = sumo-rl
CONTAINER_NAME = sumo-rl-container
USER_ID = $(shell id -u)
GROUP_ID = $(shell id -g)

# Import the .env file to get the DISPLAY variable for different OS
-include .env
export

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .


# run the Docker container without GUI
run:
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		--network=host \
		--mount type=bind,source=$(CURDIR),target=/workspace \
		--user $(USER_ID):$(GROUP_ID) \
		$(IMAGE_NAME) bash

# run the Docker container with GUI support
run-gui:
	xhost +local:root
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		--network=host \
		--mount type=bind,source=$(CURDIR),target=/workspace/ \
		--user $(USER_ID):$(GROUP_ID) \
		$(GUI_MOUNT) \
		-e DISPLAY=$(DISPLAY) \
		-e QT_X11_NO_MITSHM=1 \
		$(IMAGE_NAME) bash
	xhost -local:root

stop:
	docker stop $(CONTAINER_NAME) || true

# attach to the running container
attach:
	docker exec -it $(CONTAINER_NAME) bash

# Clean up the Docker container and image
clean:
	docker rm -f $(CONTAINER_NAME) || true
	docker rmi -f $(IMAGE_NAME) || true


		



