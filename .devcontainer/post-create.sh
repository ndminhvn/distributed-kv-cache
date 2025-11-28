#!/bin/bash

set -e

echo "Setting up development environment..."

# Make all scripts executable
if [ -d "scripts" ]; then
    chmod +x scripts/*.sh
    echo "✓ Made all scripts executable"
fi

# Install uv dependencies for local development (optional)
echo "Setting up local Python environments..."

for service in coordinator gateway worker; do
    if [ -d "services/$service" ]; then
        echo "  ➜ Setting up $service..."
        cd "services/$service"
        
        # Generate uv.lock if it doesn't exist
        if [ ! -f "uv.lock" ]; then
            echo "    Creating uv.lock..."
            uv lock
        fi
        
        cd ../..
    fi
done

echo "✓ Development environment ready!"

# Verify deployment tools
echo ""
echo "Verifying deployment tools..."
gcloud version 2>/dev/null && echo "✓ gcloud installed" || echo "✗ gcloud not found"
terraform version 2>/dev/null && echo "✓ terraform installed" || echo "✗ terraform not found"
kubectl version --client 2>/dev/null && echo "✓ kubectl installed" || echo "✗ kubectl not found"
docker --version 2>/dev/null && echo "✓ docker CLI installed" || echo "✗ docker not found"

# Print helpful information
echo ""
echo "================================================"
echo "   Development Environment Ready!"
echo "================================================"
echo ""
echo " Local Development:"
echo "  Start services:  ./scripts/local_dev.sh start"
echo "  View logs:       ./scripts/local_dev.sh logs"
echo "  Stop services:   ./scripts/local_dev.sh stop"
echo ""
echo " GKE Deployment:"
echo "  1. Configure GCP:  gcloud auth login"
echo "  2. Quick deploy:   cd scripts && ./quickstart_gke.sh"
echo "  3. Or step by step:"
echo "     - Infrastructure: cd infra && terraform init && terraform apply"
echo "     - Build images:   cd scripts && ./build_images.sh"
echo "     - Deploy to GKE:  ./deploy_gke.sh"
echo "  4. Manage cluster: ./gke_helper.sh"
echo ""
echo " Documentation:"
echo "  - Local setup:   README.md"
echo "  - GKE deployment: infra/DEPLOYMENT.md"
echo ""
echo " Deployment Tools:"
echo "  - gcloud:   $(gcloud version 2>/dev/null | head -n1 || echo 'not found')"
echo "  - terraform: $(terraform version 2>/dev/null | head -n1 || echo 'not found')"
echo "  - kubectl:  $(kubectl version 2>/dev/null || echo 'not found')"
echo "  - docker:   $(docker --version 2>/dev/null || echo 'not found')"
echo ""
