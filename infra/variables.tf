variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "distributed-kv-cache"
}

# Gateway node pool variables
variable "gateway_node_count" {
  description = "Initial number of gateway nodes"
  type        = number
  default     = 1  # Start with 1 for light workload
}

variable "gateway_min_nodes" {
  description = "Minimum number of gateway nodes for autoscaling"
  type        = number
  default     = 1
}

variable "gateway_max_nodes" {
  description = "Maximum number of gateway nodes for autoscaling"
  type        = number
  default     = 2  # Max 2 for demo purposes
}

# Worker node pool variables
variable "worker_node_count" {
  description = "Initial number of worker nodes"
  type        = number
  default     = 2  # Start with 2 for light workload
}

variable "worker_min_nodes" {
  description = "Minimum number of worker nodes for autoscaling"
  type        = number
  default     = 1  # Can scale down to 1
}

variable "worker_max_nodes" {
  description = "Maximum number of worker nodes for autoscaling"
  type        = number
  default     = 3  # Max 3 for demo purposes
}

variable "worker_machine_type" {
  description = "Machine type for worker nodes"
  type        = string
  default     = "e2-standard-4" # 4 vCPU, 16GB RAM - sufficient for CPU inference
}

# GPU configuration
variable "enable_gpu" {
  description = "Enable GPU for worker nodes"
  type        = bool
  default     = false  # Disabled by default for cost savings
}

variable "gpu_type" {
  description = "GPU type for worker nodes"
  type        = string
  default     = "nvidia-tesla-t4" # T4 is cost-effective for inference
}

variable "gpu_count" {
  description = "Number of GPUs per worker node"
  type        = number
  default     = 1
}
