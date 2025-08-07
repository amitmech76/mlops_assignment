#!/bin/bash

# Local Deployment Script for ML Prediction API
# This script deploys the API locally using Docker

set -e  # Exit on any error

# Configuration
IMAGE_NAME="ml-prediction-api"
CONTAINER_NAME="ml-api-local"
PORT=8000
MLRUNS_PATH="./mlruns"

echo "🚀 Starting local deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Stop and remove existing container if it exists
echo "🛑 Stopping existing container..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME .

# Check if mlruns directory exists
if [ ! -d "$MLRUNS_PATH" ]; then
    echo "⚠️  Warning: mlruns directory not found. Models may not load properly."
    echo "   Please ensure you have trained models in the mlruns directory."
fi

# Run the container
echo "🐳 Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -p $PORT:8000 \
    -v "$(pwd)/$MLRUNS_PATH:/app/mlruns" \
    -e MLFLOW_TRACKING_URI=file:./mlruns \
    $IMAGE_NAME

# Wait for container to start
echo "⏳ Waiting for container to start..."
sleep 10

# Health check
echo "🏥 Performing health check..."
if curl -f http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "✅ API is healthy and running!"
    echo "📊 API Documentation: http://localhost:$PORT/docs"
    echo "🔍 Health Check: http://localhost:$PORT/health"
    echo "📋 Models Info: http://localhost:$PORT/models/info"
else
    echo "❌ Health check failed. Container may not be running properly."
    echo "📋 Container logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi

echo "🎉 Local deployment completed successfully!"
echo ""
echo "📝 Usage examples:"
echo "   curl -X GET http://localhost:$PORT/health"
echo "   curl -X GET http://localhost:$PORT/models/info"
echo ""
echo "🔧 To stop the API: docker stop $CONTAINER_NAME"
echo "🔧 To view logs: docker logs $CONTAINER_NAME" 