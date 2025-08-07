#!/bin/bash

# EC2 Deployment Script for ML Prediction API
# This script deploys the API to an EC2 instance using Docker

set -e  # Exit on any error

# Configuration
DOCKER_IMAGE="${DOCKERHUB_USERNAME:-your-username}/ml-prediction-api:latest"
CONTAINER_NAME="ml-api-ec2"
PORT=8000
MLRUNS_PATH="/opt/mlruns"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting EC2 deployment...${NC}"

# Check if running on EC2
if [ ! -f /sys/hypervisor/uuid ] && [ ! -f /proc/device-tree/model ]; then
    echo -e "${YELLOW}⚠️  Warning: This script is designed for EC2 instances.${NC}"
fi

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Installing Docker...${NC}"
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo -e "${GREEN}✅ Docker installed successfully${NC}"
fi

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Login to Docker Hub (if credentials are provided)
if [ ! -z "$DOCKERHUB_USERNAME" ] && [ ! -z "$DOCKERHUB_PASSWORD" ]; then
    echo -e "${GREEN}🔐 Logging into Docker Hub...${NC}"
    echo "$DOCKERHUB_PASSWORD" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
fi

# Stop and remove existing container if it exists
echo -e "${YELLOW}🛑 Stopping existing container...${NC}"
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Pull the latest image
echo -e "${GREEN}📥 Pulling latest Docker image...${NC}"
docker pull $DOCKER_IMAGE

# Create mlruns directory if it doesn't exist
sudo mkdir -p $MLRUNS_PATH
sudo chown $USER:$USER $MLRUNS_PATH

# Run the container
echo -e "${GREEN}🐳 Starting container...${NC}"
docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -p $PORT:8000 \
    -v "$MLRUNS_PATH:/app/mlruns" \
    -e MLFLOW_TRACKING_URI=file:./mlruns \
    $DOCKER_IMAGE

# Wait for container to start
echo -e "${YELLOW}⏳ Waiting for container to start...${NC}"
sleep 15

# Health check
echo -e "${GREEN}🏥 Performing health check...${NC}"
if curl -f http://localhost:$PORT/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API is healthy and running!${NC}"
    echo -e "${GREEN}📊 API Documentation: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):$PORT/docs${NC}"
    echo -e "${GREEN}🔍 Health Check: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):$PORT/health${NC}"
    echo -e "${GREEN}📋 Models Info: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):$PORT/models/info${NC}"
else
    echo -e "${RED}❌ Health check failed. Container may not be running properly.${NC}"
    echo -e "${YELLOW}📋 Container logs:${NC}"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Set up firewall rules (if using ufw)
if command -v ufw &> /dev/null; then
    echo -e "${GREEN}🔥 Configuring firewall...${NC}"
    sudo ufw allow $PORT/tcp
    sudo ufw reload
fi

echo -e "${GREEN}🎉 EC2 deployment completed successfully!${NC}"
echo ""
echo -e "${GREEN}📝 Usage examples:${NC}"
echo "   curl -X GET http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):$PORT/health"
echo "   curl -X GET http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):$PORT/models/info"
echo ""
echo -e "${GREEN}🔧 Management commands:${NC}"
echo "   docker stop $CONTAINER_NAME"
echo "   docker logs $CONTAINER_NAME"
echo "   docker restart $CONTAINER_NAME" 