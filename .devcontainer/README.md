# Development Container

This dev container provides a complete, isolated development environment with all tools needed for local development and GKE deployment.

## What's Included

### Core Tools

- **Python 3.14** - Runtime for all services (_note: updated to 3.13 in worker service to enable CUDA support_)
- **uv** - Fast Python package manager
- **Docker CLI** - For building container images
- **Git** - Version control

### Cloud Deployment Tools

- **gcloud CLI** - Google Cloud SDK for GCP operations
- **kubectl 1.34.2** - Kubernetes command-line tool
- **Terraform 1.14.0** - Infrastructure as Code
- **GKE Auth Plugin** - Authenticate kubectl to GKE clusters

### Utilities

- **jq** - JSON processor for parsing outputs
- **httpie** - User-friendly HTTP client for testing APIs
- **vim, less** - Text editors and pagers

### VS Code Extensions

- Python development (Pylance, Black, isort)
- Kubernetes tools
- Terraform syntax & validation
- Google Cloud Code
- Docker management
- GitHub Copilot
- GitLens & Git History
- YAML support

## Quick Start

### 1. Open in Dev Container

**Using VS Code:**

1. Install "Dev Containers" extension
2. Open this folder in VS Code
3. Press `F1` → "Dev Containers: Reopen in Container"
4. Wait for container to build (~2-5 minutes first time)

**Using Command Line:**

```bash
# From project root
devcontainer open .
```

### 2. Verify Installation

After the container starts, verify all tools are installed:

```bash
# Check deployment tools
gcloud version
terraform version
kubectl version --client
docker --version

# Check Python tools
python --version
uv --version
```

All tools should be available and show their versions.

## Common Workflows

### Local Development

```bash
# Start local services with docker compose
./scripts/local_dev.sh start

# View logs
./scripts/local_dev.sh logs

# Stop services
./scripts/local_dev.sh stop

# Access services
curl http://localhost:8080/health  # Gateway
# curl http://localhost:8081/health  # Coordinator
# curl http://localhost:8082/health  # Worker
```

### GKE Deployment

#### First-time Setup

```bash
# 1. Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# 2. Set your project
gcloud config set project YOUR_PROJECT_ID

# 3. Enable required APIs
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# 4. Configure Docker for Artifact Registry
gcloud auth configure-docker REGION-docker.pkg.dev
```

#### Deploy Infrastructure

Enable Docker access inside the dev container if you encounter build issues:
```bash
sudo chown root:docker /var/run/docker.sock
```

```bash
# Option 1: Quick deploy (recommended)
cd scripts
./quickstart_gke.sh

# Option 2: Step-by-step
# Create terraform.tfvars
cd infra
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project ID

# Deploy infrastructure
terraform init
terraform plan
terraform apply

# Build and push images
cd ../scripts
./build_images.sh

# Deploy to Kubernetes
./deploy_gke.sh
```

#### Manage Cluster

```bash
# Interactive management menu
./scripts/gke_helper.sh

# Or use kubectl directly
kubectl get pods
kubectl get services
kubectl logs -f deployment/gateway
kubectl logs -f deployment/coordinator
kubectl logs -f statefulset/worker

# Scale workers
kubectl scale statefulset worker --replicas=3

# Update deployment
kubectl rollout restart deployment/gateway
```

## Dev Container Features

<!-- ### Mounted Directories

- **Workspace** - Your project files are mounted at `/workspace`
- **Docker socket** - Access to host Docker daemon for building images

### Port Forwarding

When running locally, these ports are automatically forwarded:

- `8080` - Gateway service
- `8081` - Coordinator service
- `8082` - Worker service -->

### Shell Aliases

Convenient shortcuts are pre-configured:

- `k` → `kubectl`
- `tf` → `terraform`
- `kubectl` tab completion enabled

### Environment Setup

- Python interpreter at `/usr/local/bin/python`
- Auto-formatting on save (Black + isort)
- Organized imports on save
- Hidden cache directories (`__pycache__`, `.pytest_cache`, etc.)

## Customization

### Add More Extensions

Edit `.devcontainer/devcontainer.json`:

```json
"extensions": [
  "publisher.extension-name"
]
```

### Install Additional Tools

Edit `.devcontainer/Dockerfile`:

```dockerfile
RUN apt-get update && apt-get install -y \
    your-package-name \
    && rm -rf /var/lib/apt/lists/*
```

### Run Commands on Startup

Edit `.devcontainer/post-create.sh`:

```bash
# Add your custom setup commands
echo "Running custom setup..."
```

## Notes

### Docker Socket Mount

The dev container mounts your host's Docker socket (`/var/run/docker.sock`). This allows:

- Building Docker images from within the container
- Running docker compose for local development
- Pushing images to GAR

**Important**: Images built in the dev container use the host's Docker daemon.

### GCP Authentication

After `gcloud auth login`, your credentials are stored in the container. They persist as long as the container exists but are **not** persisted to your host machine.

To persist authentication:

- Use `gcloud auth login` in the dev container
- Your `~/.config/gcloud` is stored in the container's volume
- Or mount your host's gcloud config (not recommended for security)

### Terraform State

Terraform state files are stored in the workspace and are **not** affected by container rebuilds. However, consider using remote state for production:

```hcl
terraform {
  backend "gcs" {
    bucket = "your-terraform-state-bucket"
    prefix = "distributed-kv-cache"
  }
}
```

## Troubleshooting

### Container won't start

```bash
# Rebuild the container
F1 → "Dev Containers: Rebuild Container"

# Or clean rebuild
F1 → "Dev Containers: Rebuild Container Without Cache"
```

### Docker commands fail

```bash
# Verify Docker socket is mounted
ls -la /var/run/docker.sock

# Check if you can access Docker
docker ps
```

### gcloud not authenticated

```bash
# Login to Google Cloud
gcloud auth login

# Verify authentication
gcloud auth list

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### kubectl can't connect to cluster

```bash
# Get cluster credentials
gcloud container clusters get-credentials distributed-kv-cache --zone us-central1-a

# Verify connection
kubectl cluster-info
kubectl get nodes
```

## Resources

- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [Google Cloud SDK](https://cloud.google.com/sdk/docs)
- [Terraform Documentation](https://www.terraform.io/docs)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)

## Security Best Practices

1. **Never commit credentials** - `.gitignore` already excludes common credential files
2. **Use service accounts** - For CI/CD, use GCP service accounts instead of personal credentials
3. **Limit permissions** - Only grant necessary IAM permissions
4. **Rotate keys** - Regularly rotate service account keys
5. **Use Secret Manager** - Store sensitive data in GCP Secret Manager, not in code

## Tips

- Use `k` instead of `kubectl` for faster typing
- Run `terraform fmt` before committing to format HCL files
- Use `gcloud config configurations` to manage multiple GCP projects
- Keep the dev container running for faster startups (rebuilds are slower)
- Use `./scripts/gke_helper.sh` for common GKE operations
