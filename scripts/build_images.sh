#!/bin/bash
# Build and push Docker images to Google Artifact Registry
set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
REGION="${GCP_REGION:-us-central1}"
REPOSITORY="${ARTIFACT_REGISTRY_REPO:-distributed-kv-cache}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGION}-docker.pkg.dev"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Building and Pushing Docker Images${NC}"
echo -e "${GREEN}================================================${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Create Artifact Registry repository if it doesn't exist
echo -e "\n${YELLOW}Checking Artifact Registry repository...${NC}"
if ! gcloud artifacts repositories describe ${REPOSITORY} --location=${REGION} --project=${PROJECT_ID} &>/dev/null; then
    echo -e "${YELLOW}Creating Artifact Registry repository: ${REPOSITORY}...${NC}"
    gcloud artifacts repositories create ${REPOSITORY} \
        --repository-format=docker \
        --location=${REGION} \
        --description="Distributed KV Cache container images" \
        --project=${PROJECT_ID}
    echo -e "${GREEN}✓ Repository created${NC}"
else
    echo -e "${GREEN}✓ Repository exists${NC}"
fi

# Configure Docker to use gcloud as a credential helper for Artifact Registry
echo -e "\n${YELLOW}Configuring Docker authentication...${NC}"
gcloud auth configure-docker ${REGISTRY}

# Confirm project ID
echo -e "\n${YELLOW}Project ID: ${PROJECT_ID}${NC}"
read -p "Is this correct? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Please set GCP_PROJECT_ID environment variable${NC}"
    echo "Example: export GCP_PROJECT_ID=your-gcp-project-id"
    exit 1
fi

# Build options
read -p "Enable GPU support for worker? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ENABLE_GPU="true"
    echo -e "${GREEN}Building worker image with GPU support${NC}"
else
    ENABLE_GPU="false"
    echo -e "${GREEN}Building worker image for CPU only${NC}"
fi

# Build and push coordinator
echo -e "\n${YELLOW}[1/3] Building coordinator image...${NC}"
docker build \
    --platform linux/amd64 \
    -t ${REGISTRY}/${PROJECT_ID}/${REPOSITORY}/coordinator:${IMAGE_TAG} \
    -f services/coordinator/Dockerfile \
    services/coordinator

echo -e "${GREEN}Pushing coordinator image...${NC}"
docker push ${REGISTRY}/${PROJECT_ID}/${REPOSITORY}/coordinator:${IMAGE_TAG}

# Build and push gateway
echo -e "\n${YELLOW}[2/3] Building gateway image...${NC}"
docker build \
    --platform linux/amd64 \
    -t ${REGISTRY}/${PROJECT_ID}/${REPOSITORY}/gateway:${IMAGE_TAG} \
    -f services/gateway/Dockerfile \
    services/gateway

echo -e "${GREEN}Pushing gateway image...${NC}"
docker push ${REGISTRY}/${PROJECT_ID}/${REPOSITORY}/gateway:${IMAGE_TAG}

# Build and push worker
echo -e "\n${YELLOW}[3/3] Building worker image...${NC}"
docker build \
    --platform linux/amd64 \
    --build-arg ENABLE_CUDA=${ENABLE_GPU} \
    -t ${REGISTRY}/${PROJECT_ID}/${REPOSITORY}/worker:${IMAGE_TAG} \
    -f services/worker/Dockerfile \
    services/worker

echo -e "${GREEN}Pushing worker image...${NC}"
docker push ${REGISTRY}/${PROJECT_ID}/${REPOSITORY}/worker:${IMAGE_TAG}

# Summary
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "Images pushed to Artifact Registry:"
echo -e "  - ${REGISTRY}/${PROJECT_ID}/${REPOSITORY}/coordinator:${IMAGE_TAG}"
echo -e "  - ${REGISTRY}/${PROJECT_ID}/${REPOSITORY}/gateway:${IMAGE_TAG}"
echo -e "  - ${REGISTRY}/${PROJECT_ID}/${REPOSITORY}/worker:${IMAGE_TAG} (GPU: ${ENABLE_GPU})"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  1. Run: cd infra && terraform init"
echo -e "  2. Run: terraform apply"
echo -e "  3. Run: cd .. && ./scripts/deploy_gke.sh"
