.PHONY: help build build-all run run-docker up down logs deploy-k8s

help:
	@echo ""
	@echo "BPLA Person Detection - Build Orchestration"
	@echo "============================================"
	@echo "
	@echo "Build Targets:"
	@echo "  build           - Build GPU-only Docker image"
	@echo "
	@echo "Component Builds:"
	@echo "  rust            - Build Rust core library"
	@echo "  train           - Training components"
	@echo "  models          - Model management"
	@echo "
	@echo "Runtime Targets:"
	@echo "  run             - Run locally (Julia)"
	@echo "  run-docker      - Run using Docker Compose"
	@echo "  up              - Start services via docker-compose"
	@echo "  down            - Stop services"
	@echo "  logs            - View logs"
	@echo "
	@echo "Deployment:"
	@echo "  deploy-k8s      - Deploy to Kubernetes"
	@echo ""

build:
	@echo "Building GPU-only Docker image..."
	@cd bpla_person_detection/src && docker compose build

build-all: build
	@echo "All images built successfully!"

rust:
	@echo "Building R core library..."
	@cd src/core && $(MAKE) all

train:
	@echo "Training..."
	@cd bpla_person_detection/src/train_project && $(MAKE) train

models:
	@echo "Models..."
	@cd bpla_person_detection/src/models && $(MAKE) download

run-docker:
	@echo "Running with Docker Compose..."
	@cd bpla_person_detection/src && docker compose up

up:
	@echo "Starting BPLA services..."
	@cd bpla_person_detection/src && docker compose up -d

down:
	@echo "Stopping BPLA services..."
	@cd bpla_person_detection/src && docker compose down

logs:
	@echo "Viewing logs..."
	@cd bpla_person_detection/src && docker compose logs -f

run:
	@echo "Running locally with Julia..."
	@cd bpla_person_detection/src && julia --project=. -e "using BPLADetection; BPLADetection.run_serve()"

deploy-k8s:
	@echo "Deploying to Kubernetes..."
	@kubectl apply -k deploy/kubernetes/
	@echo "Deployment complete."
