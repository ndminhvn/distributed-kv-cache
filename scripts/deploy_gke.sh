#!/bin/bash
# Deploy distributed KV cache system to GKE
set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
CLUSTER_NAME="${CLUSTER_NAME:-distributed-kv-cache}"
REPOSITORY="${ARTIFACT_REGISTRY_REPO:-distributed-kv-cache}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGION}-docker.pkg.dev"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Deploying to GKE${NC}"
echo -e "${GREEN}================================================${NC}"

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed${NC}"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed${NC}"
    exit 1
fi

# Confirm configuration
echo -e "\n${BLUE}Configuration:${NC}"
echo -e "  Project ID:    ${PROJECT_ID}"
echo -e "  Region:        ${REGION}"
echo -e "  Cluster:       ${CLUSTER_NAME}"
echo -e "  Image Tag:     ${IMAGE_TAG}"

read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Get cluster credentials
echo -e "\n${YELLOW}Getting cluster credentials...${NC}"
gcloud container clusters get-credentials ${CLUSTER_NAME} \
    --zone ${ZONE} \
    --project ${PROJECT_ID}

# Create temporary directory for processed manifests
TMP_DIR=$(mktemp -d)
echo -e "${YELLOW}Processing manifests...${NC}"

# Process coordinator manifest
sed "s|REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY|${REGISTRY}/${PROJECT_ID}/${REPOSITORY}|g" k8s/coordinator.yaml > ${TMP_DIR}/coordinator.yaml

# Process gateway manifest (ephemeral IP, no need for load balancer IP replacement)
sed "s|REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY|${REGISTRY}/${PROJECT_ID}/${REPOSITORY}|g" k8s/gateway.yaml > ${TMP_DIR}/gateway.yaml

# Process worker manifest
sed "s|REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY|${REGISTRY}/${PROJECT_ID}/${REPOSITORY}|g" k8s/worker.yaml > ${TMP_DIR}/worker.yaml

# Deploy services
echo -e "\n${YELLOW}Deploying coordinator...${NC}"
kubectl apply -f ${TMP_DIR}/coordinator.yaml

echo -e "${YELLOW}Waiting for coordinator to be ready...${NC}"
kubectl rollout status deployment/coordinator --timeout=300s

echo -e "\n${YELLOW}Deploying workers...${NC}"
kubectl apply -f ${TMP_DIR}/worker.yaml

echo -e "${YELLOW}Waiting for workers to be ready...${NC}"
kubectl rollout status statefulset/worker --timeout=600s

echo -e "\n${YELLOW}Deploying gateway...${NC}"
kubectl apply -f ${TMP_DIR}/gateway.yaml

echo -e "${YELLOW}Waiting for gateway to be ready...${NC}"
kubectl rollout status deployment/gateway --timeout=300s

# Clean up temp directory
rm -rf ${TMP_DIR}

# Get service information
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}================================================${NC}"

# Wait for LoadBalancer IP
echo -e "\n${YELLOW}Waiting for LoadBalancer IP...${NC}"
for i in {1..30}; do
    EXTERNAL_IP=$(kubectl get svc gateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    if [ -n "$EXTERNAL_IP" ]; then
        break
    fi
    echo -ne "  Attempt $i/30...\r"
    sleep 10
done

echo -e "\n${BLUE}Service Information:${NC}"
echo -e "  Gateway URL:     http://${EXTERNAL_IP}"
echo -e "  Health Check:    http://${EXTERNAL_IP}/health"

# Get pod status
echo -e "\n${BLUE}Pod Status:${NC}"
kubectl get pods -l app=coordinator
kubectl get pods -l app=gateway
kubectl get pods -l app=worker

# Get service status
echo -e "\n${BLUE}Service Status:${NC}"
kubectl get svc

# Get HPA status
echo -e "\n${BLUE}Horizontal Pod Autoscaler:${NC}"
kubectl get hpa

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Useful Commands:${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "  View logs:"
echo -e "    kubectl logs -f deployment/coordinator"
echo -e "    kubectl logs -f deployment/gateway"
echo -e "    kubectl logs -f statefulset/worker"
echo -e ""
echo -e "  Scale workers:"
echo -e "    kubectl scale statefulset/worker --replicas=5"
echo -e ""
echo -e "  Get cluster stats:"
echo -e "    curl http://${EXTERNAL_IP}/stats"
echo -e ""
echo -e "  Test generation:"
echo -e "    curl -X POST http://${EXTERNAL_IP}/generate \\"
echo -e "      -H 'Content-Type: application/json' \\"
echo -e "      -d '{\"prompt\": \"Hello world\", \"max_tokens\": 10}'"
echo -e ""
echo -e "  Delete deployment:"
echo -e "    kubectl delete -f k8s/"
