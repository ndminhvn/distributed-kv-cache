# GKE Deployment Guide

This guide walks you through deploying the distributed KV cache system to Google Kubernetes Engine (GKE).

## Architecture Overview

- **Gateway Pods** (1+ replicas): API entry point with load balancer
- **Coordinator Pod** (1 replica): Manages consistent hashing and worker registry
- **Worker Pods** (2+ replicas): Handle inference with KV cache, GPU-enabled

## Prerequisites

1. **Google Cloud SDK**

   ```bash
   # Install gcloud CLI
   # Visit: https://cloud.google.com/sdk/docs/install

   # Authenticate
   gcloud auth login
   gcloud auth application-default login
   ```

2. **Terraform** (>= 1.0)

   ```bash
   # macOS
   brew install terraform

   # Or download from: https://www.terraform.io/downloads
   ```

3. **kubectl**

   ```bash
   # macOS
   brew install kubectl

   # Or via gcloud
   gcloud components install kubectl
   ```

4. **Docker**
   ```bash
   # Install Docker Desktop
   # Visit: https://www.docker.com/products/docker-desktop
   ```

## Step 1: Configure GCP Project

```bash
# Set your GCP project ID
export GCP_PROJECT_ID="your-gcp-project-id"
export GCP_REGION="us-central1"

# Set project
gcloud config set project $GCP_PROJECT_ID

# Enable required APIs
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## Step 2: Configure Terraform

```bash
cd infra

# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your settings
# At minimum, set:
#   project_id = "your-gcp-project-id"
nano terraform.tfvars
```

### Configuration Options

**For CPU-only deployment** (cheaper, for testing):

```hcl
enable_gpu = false
worker_machine_type = "e2-standard-8"  # 8 vCPU, 32GB RAM
```

**For GPU deployment** (for production inference):

```hcl
enable_gpu = true
worker_machine_type = "n1-standard-8"  # 8 vCPU, 30GB RAM
gpu_type = "nvidia-tesla-t4"           # Cost-effective GPU
gpu_count = 1
```

## Step 3: Create GKE Cluster with Terraform

```bash
# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Create infrastructure
terraform apply

# This creates:
# - VPC network with subnets
# - GKE cluster with 3 node pools:
#   - Gateway pool (CPU-optimized, autoscaling)
#   - Coordinator pool (small, fixed size)
#   - Worker pool (GPU-enabled, autoscaling)
# - Firewall rules
# - External load balancer IP
```

**Note:** Cluster creation takes 10-15 minutes.

## Step 4: Build and Push Docker Images

```bash
cd ..  # Back to project root

# Set environment variables
export GCP_PROJECT_ID="your-gcp-project-id"
export IMAGE_TAG="latest"  # or version tag like "v1.0.0"

# Run build script
./scripts/build_images.sh

# When prompted:
# - Confirm project ID
# - Choose CPU or GPU support for workers
```

This builds and pushes three images:

- `gcr.io/PROJECT_ID/coordinator:latest`
- `gcr.io/PROJECT_ID/gateway:latest`
- `gcr.io/PROJECT_ID/worker:latest`

## Step 5: Deploy to GKE

```bash
# Deploy all services
./scripts/deploy_gke.sh

# This will:
# 1. Connect to GKE cluster
# 2. Deploy coordinator
# 3. Deploy workers
# 4. Deploy gateway with load balancer
# 5. Configure autoscaling
# 6. Display service endpoints
```

**Note:** Initial deployment takes 5-10 minutes (worker pods need to download models).

## Step 6: Verify Deployment

```bash
# Check pod status
kubectl get pods

# Expected output:
# NAME                           READY   STATUS    RESTARTS   AGE
# coordinator-xxx-yyy            1/1     Running   0          5m
# gateway-xxx-yyy                1/1     Running   0          4m
# gateway-xxx-zzz                1/1     Running   0          4m
# worker-0                       1/1     Running   0          4m
# worker-1                       1/1     Running   0          4m
# worker-2                       1/1     Running   0          4m

# Check services
kubectl get svc

# Get gateway external IP
kubectl get svc gateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

## Step 7: Test the System

```bash
# Get external IP
GATEWAY_IP=$(kubectl get svc gateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Health check
curl http://$GATEWAY_IP/health

# Check stats
curl http://$GATEWAY_IP/stats

# Test generation
curl -X POST http://$GATEWAY_IP/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of artificial intelligence is",
    "max_tokens": 20,
    "temperature": 0.7
  }'
```

## Monitoring and Management

### View Logs

```bash
# Coordinator logs
kubectl logs -f deployment/coordinator

# Gateway logs
kubectl logs -f deployment/gateway

# Worker logs (specific pod)
kubectl logs -f worker-0
kubectl logs -f worker-1

# All worker logs
kubectl logs -f -l app=worker
```

### Scale Workers

```bash
# Scale up to 5 workers
kubectl scale statefulset/worker --replicas=5

# Scale down to 2 workers
kubectl scale statefulset/worker --replicas=2
```

### Monitor Autoscaling

```bash
# Watch HPA status
kubectl get hpa -w

# Gateway should autoscale based on CPU/memory
```

### Port Forwarding (for local testing)

```bash
# Forward gateway port
kubectl port-forward svc/gateway 8080:80

# Forward coordinator port
kubectl port-forward svc/coordinator 8081:8081

# Test locally
curl http://localhost:8080/health
```

## Troubleshooting

### Pods not starting

```bash
# Describe pod to see events
kubectl describe pod worker-0

# Common issues:
# - GPU quota exceeded (reduce gpu_count or disable GPU)
# - Insufficient resources (check node capacity)
# - Image pull errors (check GCR permissions)
```

### Workers crashing during model load

```bash
# Check worker logs
kubectl logs worker-0

# If OOM (Out of Memory):
# - Increase worker memory limits in k8s/worker.yaml
# - Use smaller model (change to "gpt2-medium" or "gpt2")
```

### Load balancer not getting external IP

```bash
# Check service status
kubectl describe svc gateway

# Force service update
kubectl delete svc gateway
kubectl apply -f k8s/gateway.yaml
```

### GPU not detected

```bash
# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU nodes
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
```

## Cost Optimization

### Use Preemptible Nodes (60-80% cheaper)

Edit `infra/main.tf` node pool configurations:

```hcl
node_config {
  preemptible = true
  ...
}
```

### Use Spot VMs for Workers

Workers can recover from interruptions:

```hcl
node_config {
  spot = true
  ...
}
```

### CPU-Only Deployment

For testing/development, disable GPUs:

```hcl
# infra/terraform.tfvars
enable_gpu = false
worker_machine_type = "e2-standard-4"
```

## Cleanup

### Delete Kubernetes Resources

```bash
# Delete all deployments
kubectl delete -f k8s/

# Or delete specific resources
kubectl delete deployment gateway coordinator
kubectl delete statefulset worker
kubectl delete svc --all
```

### Destroy GKE Cluster

```bash
cd infra

# Destroy infrastructure
terraform destroy

# Confirm by typing "yes"
```

**Warning:** This deletes everything including the cluster and load balancer.

## Production Considerations

1. **SSL/TLS**: Add SSL certificate to load balancer
2. **Monitoring**: Set up Cloud Monitoring and Logging
3. **Backup**: Configure persistent volumes for model cache
4. **Security**: Use Workload Identity, limit network access
5. **Cost**: Set up billing alerts and quotas

## Future Enhancements

- Set up monitoring with Prometheus and Grafana
- Configure SSL certificates for HTTPS
- Implement authentication/authorization
- Set up CI/CD pipeline for automated deployments
- Configure persistent storage for model cache
