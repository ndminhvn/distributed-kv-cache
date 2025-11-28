# Distributed Inference Key-Value Cache

### Minh Nguyen, Zhenghui Gui

A high-performance distributed KV cache system for LLM inference, optimized for cloud deployment on Google Kubernetes Engine (GKE).

## Overview

This system accelerates LLM inference by distributing key-value cache across multiple worker nodes using consistent hashing for locality. It provides:

- **10-50% faster inference** through intelligent KV cache reuse
- **Horizontal scalability** with automatic worker routing
- **GPU acceleration** support for production workloads
- **Production-ready** Kubernetes deployment on GKE

## Architecture

<img width="3506" height="1199" alt="system-overview" src="https://github.com/user-attachments/assets/eac7b714-29e3-4c26-8097-4c91f244071d" />

### Components

- **Gateway**: FastAPI service handling external requests, routing via consistent hashing
- **Coordinator**: Manages worker registry and sequence-to-worker mapping
- **Workers**: StatefulSet pods running inference with distributed KV cache

## Quick Start

### Option 1: Using Dev Container (Recommended)

For a consistent, isolated development environment with all deployment tools pre-installed:

1. **Open in Dev Container**

   - Install "Dev Containers" extension in VS Code
   - Press `F1` → "Dev Containers: Reopen in Container"
   - Wait for container to build (2-5 minutes first time)

2. **Deploy to GKE**

   ```bash
   # Authenticate with GCP
   gcloud auth login

   # Run automated deployment
   ./scripts/quickstart_gke.sh
   ```

The dev container includes: `gcloud`, `terraform`, `kubectl`, `docker`, `uv`, and all necessary tools.

### Option 2: Automated Deployment (Local Environment)

If you have `gcloud`, `terraform`, and `kubectl` installed locally:

```bash
# One command deployment
./scripts/quickstart_gke.sh
```

This interactive script will:

1. Configure your GCP project
2. Enable required APIs
3. Create GKE cluster with Terraform
4. Build and push Docker images
5. Deploy all services

### Option 3: Manual Deployment

```bash
# Prerequisites: gcloud, terraform, kubectl installed
# (Or use dev container - see Option 1)

# 1. Configure GCP
export GCP_PROJECT_ID="your-project-id"
gcloud config set project $GCP_PROJECT_ID

# 2. Create infrastructure
cd infra
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings
terraform init
terraform apply

# 3. Build images
cd ..
./scripts/build_images.sh

# 4. Deploy to GKE
./scripts/deploy_gke.sh
```

See [infra/DEPLOYMENT.md](infra/DEPLOYMENT.md) for detailed instructions.

### Option 4: Local Development

```bash
# Run locally with Docker Compose (dev container or local)
./scripts/local_dev.sh
```

## Development Environment

### Using Dev Container (Recommended)

The dev container provides an isolated environment with all tools pre-installed:

- **Cloud Tools**: gcloud CLI, kubectl, Terraform
- **Python Tools**: Python 3.14, uv package manager
- **Container Tools**: Docker CLI for building images
- **VS Code Extensions**: Python, Kubernetes, Terraform, Cloud Code

**Get Started:**

1. Install "Dev Containers" extension in VS Code
2. Open project and select "Reopen in Container"
3. All deployment scripts work out of the box!

See [.devcontainer/README.md](.devcontainer/README.md) for full documentation.

## Performance Testing

The project includes comprehensive test suites:

```bash
cd tests

# Install dependencies
uv sync

# Run cache performance tests
uv run pytest test_cache_performance.py -v -s

# Run stress tests
uv run pytest test_stress.py -v -s

# Run all tests
uv run pytest -v -s
```

### Test Categories

1. **Routing Distribution** - Validates consistent hashing and load balancing
2. **Cache Locality** - Verifies KV cache append and reuse behavior
3. **Generation Flow** - End-to-end inference with streaming
4. **Cache Performance** - Measures speedup with cache vs no-cache
5. **Stress Testing** - High concurrency, sustained load, burst traffic

## Project Structure

```
distributed-kv-cache/
├── infra/                  # Terraform infrastructure
│   ├── main.tf                 # GKE cluster configuration
│   ├── variables.tf            # Infrastructure variables
│   └── DEPLOYMENT.md           # Detailed deployment guide
├── k8s/                    # Kubernetes manifests
│   ├── coordinator.yaml        # Coordinator deployment
│   ├── gateway.yaml            # Gateway + LoadBalancer + HPA
│   ├── worker.yaml             # Worker StatefulSet
│   └── namespace.yaml          # ConfigMap and namespace
├── services/               # Microservices
│   ├── coordinator/            # Consistent hashing coordinator
│   ├── gateway/                # API gateway
│   └── worker/                 # Inference worker with KV cache
├── scripts/                # Deployment scripts
│   ├── quickstart_gke.sh       # One-command deployment
│   ├── build_images.sh         # Build and push to GCR
│   ├── deploy_gke.sh           # Deploy to GKE
│   └── local_dev.sh            # Local development
└── tests/                 # Comprehensive test suite
    ├── test_cache_performance.py  # Cache vs no-cache
    ├── test_stress.py             # Load testing
    └── ...
```

## Configuration

### Infrastructure (Terraform)

Edit `infra/terraform.tfvars`:

```hcl
project_id   = "your-gcp-project-id"
region       = "us-central1"
cluster_name = "distributed-kv-cache"

# Gateway autoscaling
gateway_min_nodes = 1
gateway_max_nodes = 5

# Worker configuration
worker_node_count = 3
worker_min_nodes  = 2
worker_max_nodes  = 10

# GPU settings (set enable_gpu=false for CPU-only)
enable_gpu          = true
worker_machine_type = "n1-standard-8"
gpu_type            = "nvidia-tesla-t4"
gpu_count           = 1
```

### Kubernetes Resources

**Gateway**: 2 vCPU, 4GB RAM (autoscales 2-10 pods)
**Coordinator**: 1 vCPU, 2GB RAM (1 pod)
**Worker**: 4-8 vCPU, 16-24GB RAM, optional GPU (autoscales 2-10 pods)

## Monitoring

```bash
# View pod status
kubectl get pods

# Check autoscaling
kubectl get hpa

# View logs
kubectl logs -f deployment/gateway
kubectl logs -f statefulset/worker

# Get cluster stats
GATEWAY_IP=$(kubectl get svc gateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$GATEWAY_IP/stats
```

## Testing the Deployment

```bash
# Get gateway IP
GATEWAY_IP=$(kubectl get svc gateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Health check
curl http://$GATEWAY_IP/health

# Generate text
curl -X POST http://$GATEWAY_IP/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 20,
    "temperature": 0.7,
    "model_name": "gpt2"
  }'

# Check statistics
curl http://$GATEWAY_IP/stats | jq
```

## Cost Estimation

**CPU-Only Deployment** (testing):

- Gateway: 2x e2-standard-2 ≈ $50/month
- Coordinator: 1x e2-standard-2 ≈ $25/month
- Workers: 3x e2-standard-8 ≈ $300/month
- **Total**: ~$375/month

**GPU Deployment** (production):

- Gateway: 2x e2-standard-2 ≈ $50/month
- Coordinator: 1x e2-standard-2 ≈ $25/month
- Workers: 3x n1-standard-8 + T4 GPU ≈ $900/month
- **Total**: ~$975/month

Use preemptible/spot instances for 60-80% savings.

## Security

- Workload Identity for GCP service access
- Private GKE cluster option available
- Firewall rules for internal communication
- SSL/TLS termination at load balancer (configure separately)

## Cleanup

```bash
# Delete Kubernetes resources
kubectl delete -f k8s/

# Destroy infrastructure
cd infra
terraform destroy
```

## Documentation

- [Deployment Guide](infra/DEPLOYMENT.md) - Complete GKE deployment walkthrough
- [Test Suite](tests/README.md) - Testing documentation
- [Architecture Diagrams](docs/) - System architecture and design

## Contributing

This is a university project for COSC 6376 - Cloud Computing.

## License

Educational use only - University of Houston, Fall 2025
