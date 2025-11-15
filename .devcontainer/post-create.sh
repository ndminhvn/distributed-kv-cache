#!/bin/bash

set -e

echo "Setting up development environment..."

# Make local_dev.sh executable
if [ -f "scripts/local_dev.sh" ]; then
    chmod +x scripts/local_dev.sh
    echo "Made local_dev.sh executable"
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

echo "Development environment ready!"

# Print helpful information
echo ""
echo "================================================"
echo "  Development Environment Ready!"
echo "================================================"
echo ""
echo "Quick Start:"
echo "  1. Start services:  ./scripts/local_dev.sh start"
echo "  2. View logs:       ./scripts/local_dev.sh logs"
echo "  3. Stop services:   ./scripts/local_dev.sh stop"
echo ""
echo "Or use docker-compose directly:"
echo "  docker-compose up -d        # Start services"
echo "  docker-compose logs -f      # View logs"
echo "  docker-compose ps           # Check status"
echo "  docker-compose down         # Stop services"
echo ""
echo "Service URLs (after starting):"
echo "  Gateway:     http://localhost:8000"
echo "  Coordinator: http://localhost:8001"
echo "  Worker:      http://localhost:8002"
echo ""
