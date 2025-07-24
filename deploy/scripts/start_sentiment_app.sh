#!/bin/bash
# Login to AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 890742587077.dkr.ecr.us-east-1.amazonaws.com

# Pull the latest image
docker pull 890742587077.dkr.ecr.us-east-1.amazonaws.com/mlops-small-project:latest

# Check if the container 'sentiment-analysis-app' is running
if [ "$(docker ps -q -f name=sentiment-analysis-app)" ]; then
    # Stop the running container
    docker stop sentiment-analysis-app
fi

# Check if the container 'sentiment-analysis-app' exists (stopped or running)
if [ "$(docker ps -aq -f name=sentiment-analysis-app)" ]; then
    # Remove the container if it exists
    docker rm sentiment-analysis-app
fi

# Run a new container
sudo docker run -d --name sentiment-analysis-app -p 80:8501 -e DAGSHUB_PAT=21683b18059dbd6da4652dd1e4b4f66182e712d8 890742587077.dkr.ecr.us-east-1.amazonaws.com/mlops-small-project:latest