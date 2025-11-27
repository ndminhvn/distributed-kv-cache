#!/bin/bash
# Quick start script for GKE deployment
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Distributed KV Cache - GKE Quick Start        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"

# Check if running from project root
if [ ! -f "scripts/build_images.sh" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Step 1: Project configuration
echo -e "\n${BLUE}Step 1: Configure GCP Project${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

read -p "Enter your GCP Project ID: " PROJECT_ID
read -p "Enter GCP Region [us-central1]: " REGION
REGION=${REGION:-us-central1}

export GCP_PROJECT_ID=$PROJECT_ID
export GCP_REGION=$REGION

echo -e "\n${GREEN}✓ Configuration set${NC}"
echo -e "  Project: $PROJECT_ID"
echo -e "  Region: $REGION"

# Step 2: Enable APIs
echo -e "\n${BLUE}Step 2: Enable Required GCP APIs${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

gcloud config set project $PROJECT_ID

echo -e "${YELLOW}Enabling Container API...${NC}"
gcloud services enable container.googleapis.com

echo -e "${YELLOW}Enabling Compute API...${NC}"
gcloud services enable compute.googleapis.com

echo -e "${YELLOW}Enabling Artifact Registry API...${NC}"
gcloud services enable artifactregistry.googleapis.com

echo -e "${GREEN}✓ APIs enabled${NC}"

# Step 3: Configure Terraform
echo -e "\n${BLUE}Step 3: Configure Terraform${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd infra

if [ ! -f "terraform.tfvars" ]; then
    cp terraform.tfvars.example terraform.tfvars
    sed -i.bak "s/your-gcp-project-id/$PROJECT_ID/g" terraform.tfvars
    sed -i.bak "s/us-central1/$REGION/g" terraform.tfvars
    rm -f terraform.tfvars.bak
    echo -e "${GREEN}✓ Created terraform.tfvars${NC}"
else
    echo -e "${YELLOW}terraform.tfvars already exists${NC}"
fi

# Ask about GPU
read -p "Enable GPU for workers? (costs more, better performance) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sed -i.bak "s/enable_gpu = false/enable_gpu = true/g" terraform.tfvars
    rm -f terraform.tfvars.bak
    echo -e "${GREEN}✓ GPU enabled${NC}"
else
    sed -i.bak "s/enable_gpu = true/enable_gpu = false/g" terraform.tfvars
    sed -i.bak 's/worker_machine_type = "n1-standard-8"/worker_machine_type = "e2-standard-8"/g' terraform.tfvars
    rm -f terraform.tfvars.bak
    echo -e "${GREEN}✓ CPU-only mode${NC}"
fi

# Step 4: Create infrastructure
echo -e "\n${BLUE}Step 4: Create GKE Cluster${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}This will take 10-15 minutes...${NC}"

terraform init
terraform plan -out=tfplan

read -p "Apply this plan? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Deployment cancelled${NC}"
    exit 1
fi

terraform apply tfplan

echo -e "${GREEN}✓ GKE cluster created${NC}"

cd ..

# Step 5: Build images
echo -e "\n${BLUE}Step 5: Build and Push Docker Images${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

./scripts/build_images.sh

echo -e "${GREEN}✓ Images built and pushed${NC}"

# Step 6: Deploy to GKE
echo -e "\n${BLUE}Step 6: Deploy to GKE${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

./scripts/deploy_gke.sh

# Summary
echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Deployment Complete!                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"

GATEWAY_IP=$(kubectl get svc gateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")

echo -e "\n${BLUE}Your distributed KV cache is running at:${NC}"
echo -e "  http://$GATEWAY_IP"
echo -e "\n${BLUE}Test it:${NC}"
echo -e "  curl http://$GATEWAY_IP/health"
echo -e "  curl http://$GATEWAY_IP/stats"
echo -e "\n${BLUE}View logs:${NC}"
echo -e "  kubectl logs -f deployment/gateway"
echo -e "\n${YELLOW}For more information, see: infra/DEPLOYMENT.md${NC}"
