#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

function print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Distributed KV Cache - Local Development${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
}

function print_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start         - Start all services"
    echo "  stop          - Stop all services"
    echo "  restart       - Restart all services"
    echo "  logs          - Show logs from all services"
    echo "  logs [service] - Show logs from specific service (coordinator|worker|gateway)"
    echo "  status        - Show status of all services"
    echo "  scale-workers [N] - Scale worker service to N instances"
    echo "  clean         - Stop and remove all containers, networks, and volumes"
    echo "  rebuild       - Rebuild all images and restart"
    echo "  shell [service] - Open shell in service container"
    echo ""
}

function start_services() {
    echo -e "${GREEN}Starting services...${NC}"
    docker-compose up -d --build
    echo ""
    echo -e "${GREEN}Services started!${NC}"
    echo -e "${YELLOW}Coordinator: ${NC}http://localhost:8001"
    echo -e "${YELLOW}Gateway:     ${NC}http://localhost:8000"
    echo -e "${YELLOW}Worker:      ${NC}http://localhost:8002"
    echo ""
    echo -e "${BLUE}Run '$0 logs' to see service logs${NC}"
}

function stop_services() {
    echo -e "${RED}Stopping services...${NC}"
    docker-compose down
    echo -e "${GREEN}Services stopped!${NC}"
}

function restart_services() {
    echo -e "${YELLOW}Restarting services...${NC}"
    docker-compose restart
    echo -e "${GREEN}Services restarted!${NC}"
}

function show_logs() {
    if [ -z "$1" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$1"
    fi
}

function show_status() {
    echo -e "${BLUE}Service Status:${NC}"
    docker-compose ps
}

function scale_workers() {
    if [ -z "$1" ]; then
        echo -e "${RED}Error: Please specify number of workers${NC}"
        echo "Usage: $0 scale-workers [N]"
        exit 1
    fi
    echo -e "${GREEN}Scaling workers to $1 instances...${NC}"
    docker-compose up -d --scale worker=$1
    echo -e "${GREEN}Workers scaled!${NC}"
}

function clean_all() {
    echo -e "${RED}Cleaning up all containers, networks, and volumes...${NC}"
    docker-compose down -v --remove-orphans
    echo -e "${GREEN}Cleanup complete!${NC}"
}

function rebuild_all() {
    echo -e "${YELLOW}Rebuilding all services...${NC}"
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d
    echo -e "${GREEN}Rebuild complete!${NC}"
}

function open_shell() {
    if [ -z "$1" ]; then
        echo -e "${RED}Error: Please specify service name${NC}"
        echo "Usage: $0 shell [coordinator|worker|gateway]"
        exit 1
    fi
    docker-compose exec "$1" /bin/bash
}

# Main script
print_header

case "${1:-}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs "$2"
        ;;
    status)
        show_status
        ;;
    scale-workers)
        scale_workers "$2"
        ;;
    clean)
        clean_all
        ;;
    rebuild)
        rebuild_all
        ;;
    shell)
        open_shell "$2"
        ;;
    help|--help|-h)
        print_usage
        ;;
    "")
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
