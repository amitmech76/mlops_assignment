#!/bin/bash

# Local Deployment Script for ML Prediction API
# This script deploys the API locally using Docker

set -e  # Exit on any error

# Configuration
IMAGE_NAME="ml-prediction-api"
CONTAINER_NAME="ml-api-local"
PORT=8000
MLRUNS_PATH="./mlruns"
LOGS_PATH="./logs"

echo "Starting local deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Stop and remove existing container if it exists
echo "Stopping existing container..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Clean up any dangling containers
docker container prune -f 2>/dev/null || true

# Build the Docker image if it doesn't exist or force rebuild
if [[ "$1" == "--build" ]] || ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME .
else
    echo "Using existing Docker image..."
fi

# Ensure directories exist
mkdir -p "$MLRUNS_PATH" "$LOGS_PATH"

# Check if mlruns directory has models
if [ ! -d "$MLRUNS_PATH" ] || [ -z "$(ls -A $MLRUNS_PATH 2>/dev/null)" ]; then
    echo "WARNING: mlruns directory is empty. Models may not load properly."
    echo "Please ensure you have trained models in the mlruns directory."
    echo "You can train models by running:"
    echo "cd src && python train_housing.py && python train_iris.py"
fi

# Run the container
echo "Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -p $PORT:8000 \
    -v "$(pwd)/$MLRUNS_PATH:/app/mlruns" \
    -v "$(pwd)/$LOGS_PATH:/app/logs" \
    -e MLFLOW_TRACKING_URI=file:./mlruns \
    -e PYTHONUNBUFFERED=1 \
    $IMAGE_NAME

# Wait for container to start
echo "Waiting for container to start..."
for i in {1..30}; do
    if docker ps | grep -q $CONTAINER_NAME; then
        echo "Container is running..."
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Container failed to start"
        docker logs $CONTAINER_NAME
        exit 1
    fi
    sleep 2
done

# Wait for API to be ready
echo "Waiting for API to be ready..."
for i in {1..60}; do
    if curl -f http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "SUCCESS: API is healthy and running!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: API failed to start properly"
        echo "Container logs:"
        docker logs $CONTAINER_NAME
        exit 1
    fi
    sleep 2
done

# Display deployment information
echo ""
echo "Local deployment completed successfully!"
echo ""
echo "Deployment Information:"
echo "   Container Name: $CONTAINER_NAME"
echo "   Port: $PORT"
echo "   API Documentation: http://localhost:$PORT/docs"
echo "   Health Check: http://localhost:$PORT/health"
echo "   Models Info: http://localhost:$PORT/models/info"
echo "   Metrics: http://localhost:$PORT/metrics/prometheus"
echo ""
echo "Usage examples:"
echo "   curl -X GET http://localhost:$PORT/health"
echo "   curl -X GET http://localhost:$PORT/models/info"
echo ""
echo "Management commands:"
echo "   Stop API: docker stop $CONTAINER_NAME"
echo "   View logs: docker logs $CONTAINER_NAME"
echo "   Follow logs: docker logs -f $CONTAINER_NAME"
echo "   Remove: docker rm $CONTAINER_NAME" 