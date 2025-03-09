# Makefile for the Kurtis Project
# This Makefile provides targets for training the model, starting a chat session,
# installing dependencies, Docker operations, and cleaning up output files.

# Variables
PYTHON = python3
MODULE = kurtis
OUTPUT_DIR = output
IMAGE_NAME = kurtis:latest
DOCKER_REGISTRY = ethicalabs/kurtis
CONFIG_MODULE = kurtis.config.default

# Default target
.PHONY: all
all: help

# Help target to display available commands
help:
	@echo "Available commands:"
	@echo "  make preprocessing  - Preprocess the data using a pre-trained LLM."
	@echo "  make train          - Train the model."
	@echo "  make chat           - Start a prompt session with the model."
	@echo "  make install        - Install project dependencies using uv."
	@echo "  make eval_model     - Evaluate model."
	@echo "  make docker_build   - Build the Docker image for the project."
	@echo "  make docker_push    - Push the Docker image to the registry."
	@echo "  make docker_run     - Run the Docker container with output mounted."
	@echo "  make docker_train   - Run the training script inside the Docker container."
	@echo "  make docker_chat    - Start a prompt session inside the Docker container."


# Preprocess the dataset
.PHONY: preprocessing
preprocessing:
	uv run $(PYTHON) -m $(MODULE) --preprocessing --config-module $(CONFIG_MODULE) --debug

# Train the model
.PHONY: train
train:
	uv run $(PYTHON) -m $(MODULE) --train --config-module $(CONFIG_MODULE)

# Eval the model
.PHONY: eval_model
eval_model:
	uv run $(PYTHON) -m $(MODULE) --eval-model --config-module $(CONFIG_MODULE)


# Start a chat session
.PHONY: chat
chat:
	uv run $(PYTHON) -m $(MODULE) --chat --config-module $(CONFIG_MODULE)

# Install dependencies
.PHONY: install
install:
	uv sync --frozen --no-install-project;

# Cleanup target to remove files in the output directory with confirmation
.PHONY: cleanup
cleanup:
	@read -p "Are you sure you want to remove all files in the output directory? (y/n) " confirm; \
	if [ "$$confirm" = "y" ]; then \
		rm -rf $(OUTPUT_DIR)/*; \
		echo "Cleanup complete."; \
	else \
		echo "Cleanup aborted."; \
	fi

# Clean target (optional, can be used to remove build artifacts if any)
.PHONY: clean
clean:
	@echo "Cleaning up..."
	# Add commands to clean up build artifacts if necessary

# Docker build
.PHONY: docker_build
docker_build:
	docker build -t $(IMAGE_NAME) .

# Docker push
.PHONY: docker_push
docker_push:
	docker tag $(IMAGE_NAME) $(DOCKER_REGISTRY)
	docker push $(DOCKER_REGISTRY)

# Docker run with output mounted
.PHONY: docker_run
docker_run:
	docker run --rm -v $(PWD)/$(OUTPUT_DIR):/app/output $(IMAGE_NAME)

# Docker train inside the container
.PHONY: docker_train
docker_train:
	docker run --rm -v $(PWD)/$(OUTPUT_DIR):/app/output $(IMAGE_NAME) --train --config-module $(CONFIG_MODULE)
# Docker chat session inside the container
.PHONY: docker_chat
docker_chat:
	docker run --rm -v $(PWD)/$(OUTPUT_DIR):/app/output $(IMAGE_NAME) --chat --config-module $(CONFIG_MODULE)
