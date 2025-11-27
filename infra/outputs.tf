# Terraform outputs reference
output "kubectl_config_command" {
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${var.region} --project ${var.project_id}"
  description = "Command to configure kubectl"
}

output "gateway_node_pool" {
  value = {
    name       = google_container_node_pool.gateway_pool.name
    node_count = google_container_node_pool.gateway_pool.node_count
    min_nodes  = var.gateway_min_nodes
    max_nodes  = var.gateway_max_nodes
  }
  description = "Gateway node pool configuration"
}

output "worker_node_pool" {
  value = {
    name          = google_container_node_pool.worker_pool.name
    node_count    = google_container_node_pool.worker_pool.node_count
    min_nodes     = var.worker_min_nodes
    max_nodes     = var.worker_max_nodes
    machine_type  = var.worker_machine_type
    gpu_enabled   = var.enable_gpu
    gpu_type      = var.enable_gpu ? var.gpu_type : "none"
  }
  description = "Worker node pool configuration"
}
