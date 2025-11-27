#!/bin/bash
# Helper script for common GKE operations
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_menu() {
    echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  Distributed KV Cache - GKE Helper             ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "1) Show cluster status"
    echo "2) View logs"
    echo "3) Scale workers"
    echo "4) Test endpoints"
    echo "5) Port forward (for local testing)"
    echo "6) Update deployment"
    echo "7) Monitor autoscaling"
    echo "8) Get resource usage"
    echo "9) Delete deployment"
    echo "0) Exit"
    echo ""
}

cluster_status() {
    echo -e "${BLUE}Cluster Status:${NC}"
    echo ""
    echo "Pods:"
    kubectl get pods -o wide
    echo ""
    echo "Services:"
    kubectl get svc
    echo ""
    echo "HPA:"
    kubectl get hpa
    echo ""
    GATEWAY_IP=$(kubectl get svc gateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    echo -e "${GREEN}Gateway IP: ${GATEWAY_IP}${NC}"
}

view_logs() {
    echo "Select service:"
    echo "1) Coordinator"
    echo "2) Gateway"
    echo "3) Worker-0"
    echo "4) Worker-1"
    echo "5) Worker-2"
    echo "6) All workers"
    read -p "Choice: " choice
    
    case $choice in
        1) kubectl logs -f deployment/coordinator ;;
        2) kubectl logs -f deployment/gateway ;;
        3) kubectl logs -f worker-0 ;;
        4) kubectl logs -f worker-1 ;;
        5) kubectl logs -f worker-2 ;;
        6) kubectl logs -f -l app=worker --max-log-requests=10 ;;
        *) echo "Invalid choice" ;;
    esac
}

scale_workers() {
    read -p "Enter number of workers (2-10): " replicas
    if [ "$replicas" -ge 2 ] && [ "$replicas" -le 10 ]; then
        kubectl scale statefulset/worker --replicas=$replicas
        echo -e "${GREEN}Scaled to $replicas workers${NC}"
        kubectl rollout status statefulset/worker
    else
        echo "Invalid number. Must be between 2 and 10."
    fi
}

test_endpoints() {
    GATEWAY_IP=$(kubectl get svc gateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [ -z "$GATEWAY_IP" ]; then
        echo -e "${YELLOW}Gateway IP not ready yet${NC}"
        return
    fi
    
    echo -e "${BLUE}Testing endpoints on ${GATEWAY_IP}...${NC}"
    echo ""
    
    echo "1) Health check:"
    curl -s http://$GATEWAY_IP/health | jq || curl -s http://$GATEWAY_IP/health
    echo ""
    
    echo "2) Stats:"
    curl -s http://$GATEWAY_IP/stats | jq || curl -s http://$GATEWAY_IP/stats
    echo ""
    
    read -p "Test generation? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "3) Generation:"
        curl -X POST http://$GATEWAY_IP/generate \
            -H "Content-Type: application/json" \
            -d '{"prompt": "Hello world", "max_tokens": 10, "temperature": 0.7}'
        echo ""
    fi
}

port_forward() {
    echo "Select service to forward:"
    echo "1) Gateway (8080)"
    echo "2) Coordinator (8081)"
    echo "3) Worker-0 (8082)"
    read -p "Choice: " choice
    
    case $choice in
        1) 
            echo -e "${GREEN}Forwarding gateway to localhost:8080${NC}"
            kubectl port-forward svc/gateway 8080:80
            ;;
        2) 
            echo -e "${GREEN}Forwarding coordinator to localhost:8081${NC}"
            kubectl port-forward svc/coordinator 8081:8081
            ;;
        3) 
            echo -e "${GREEN}Forwarding worker-0 to localhost:8082${NC}"
            kubectl port-forward worker-0 8082:8082
            ;;
        *) echo "Invalid choice" ;;
    esac
}

update_deployment() {
    echo "Select service to update:"
    echo "1) Rebuild and update all images"
    echo "2) Redeploy from existing images"
    read -p "Choice: " choice
    
    case $choice in
        1)
            echo -e "${YELLOW}Building and pushing new images...${NC}"
            ./scripts/build_images.sh
            echo -e "${YELLOW}Restarting deployments...${NC}"
            kubectl rollout restart deployment/coordinator
            kubectl rollout restart deployment/gateway
            kubectl rollout restart statefulset/worker
            ;;
        2)
            kubectl rollout restart deployment/coordinator
            kubectl rollout restart deployment/gateway
            kubectl rollout restart statefulset/worker
            ;;
        *) echo "Invalid choice" ;;
    esac
}

monitor_autoscaling() {
    echo -e "${BLUE}Monitoring autoscaling (press Ctrl+C to stop)...${NC}"
    watch -n 2 'kubectl get hpa && echo "" && kubectl get pods -l app=gateway && echo "" && kubectl get pods -l app=worker'
}

resource_usage() {
    echo -e "${BLUE}Resource Usage:${NC}"
    echo ""
    echo "Top Nodes:"
    kubectl top nodes
    echo ""
    echo "Top Pods:"
    kubectl top pods
}

delete_deployment() {
    echo -e "${YELLOW}WARNING: This will delete all deployments${NC}"
    read -p "Are you sure? (type 'yes' to confirm): " confirm
    
    if [ "$confirm" = "yes" ]; then
        echo "Deleting Kubernetes resources..."
        kubectl delete -f k8s/
        echo -e "${GREEN}Deployment deleted${NC}"
        echo ""
        read -p "Also destroy GKE cluster with Terraform? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd infra
            terraform destroy
            cd ..
        fi
    else
        echo "Cancelled"
    fi
}

# Main loop
while true; do
    show_menu
    read -p "Select an option: " option
    echo ""
    
    case $option in
        1) cluster_status ;;
        2) view_logs ;;
        3) scale_workers ;;
        4) test_endpoints ;;
        5) port_forward ;;
        6) update_deployment ;;
        7) monitor_autoscaling ;;
        8) resource_usage ;;
        9) delete_deployment ;;
        0) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid option" ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
done
