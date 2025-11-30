terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC Network - Use auto mode for simplicity
resource "google_compute_network" "vpc" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = true
}

# Firewall rule - Allow external traffic to load balancer
resource "google_compute_firewall" "allow_external_lb" {
  name    = "${var.cluster_name}-allow-external-lb"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["443", "80"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["gke-node"]
}

# Firewall rule - Allow internal cluster communication
resource "google_compute_firewall" "allow_internal" {
  name    = "${var.cluster_name}-allow-internal"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
  }

  allow {
    protocol = "udp"
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/8"]  # Covers all auto-created subnets
  target_tags   = ["gke-node"]
}

# GKE Cluster (Zonal for minimal quota usage)
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.zone  # Use zone instead of region for single-zone cluster

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1
  
  # Allow Terraform to destroy the cluster (disable for production)
  deletion_protection = false

  network = google_compute_network.vpc.name

  # Use automatic IP allocation
  ip_allocation_policy {
    cluster_ipv4_cidr_block  = ""
    services_ipv4_cidr_block = ""
  }

  # Enable cluster add-ons
  addons_config {
    http_load_balancing {
      disabled = false  # Enable ingress/load balancer support
    }
    horizontal_pod_autoscaling {
      disabled = false  # Enable HPA for pod autoscaling
    }
  }

  # Release channel for automatic upgrades
  release_channel {
    channel = "REGULAR"
  }
}

# Gateway Node Pool (CPU-optimized for API routing)
resource "google_container_node_pool" "gateway_pool" {
  name       = "gateway-pool"
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = var.gateway_node_count

  node_config {
    machine_type = "e2-small" # 2 vCPU, 2GB RAM - sufficient for light API routing
    disk_size_gb = 30  # Reduced from 50GB

    labels = {
      role = "gateway"
    }

    tags = ["gke-node", "gateway"]

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  autoscaling {
    min_node_count = var.gateway_min_nodes
    max_node_count = var.gateway_max_nodes
  }
}

# Coordinator Node Pool (CPU-optimized for coordination)
resource "google_container_node_pool" "coordinator_pool" {
  name       = "coordinator-pool"
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = 1 # Coordinator doesn't need scaling

  node_config {
    machine_type = "e2-small" # 2 vCPU, 2GB RAM - sufficient for coordination
    disk_size_gb = 20  # Reduced from 30GB

    labels = {
      role = "coordinator"
    }

    tags = ["gke-node", "coordinator"]

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

# Worker Node Pool (GPU-enabled for inference)
resource "google_container_node_pool" "worker_pool" {
  name       = "worker-pool"
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = var.worker_node_count

  node_config {
    machine_type = var.worker_machine_type # e.g., "e2-standard-4" for CPU
    disk_size_gb = 50  # Reduced from 100GB - sufficient for model cache

    labels = {
      role = "worker"
    }

    tags = ["gke-node", "worker"]

    # GPU configuration (optional - enable if using GPU nodes)
    dynamic "guest_accelerator" {
      for_each = var.enable_gpu ? [1] : []
      content {
        type  = var.gpu_type # e.g., "nvidia-tesla-t4"
        count = var.gpu_count
      }
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }

  autoscaling {
    min_node_count = var.worker_min_nodes
    max_node_count = var.worker_max_nodes
  }
}

# Outputs
output "cluster_name" {
  value       = google_container_cluster.primary.name
  description = "GKE cluster name"
}

output "cluster_endpoint" {
  value       = google_container_cluster.primary.endpoint
  description = "GKE cluster endpoint"
  sensitive   = true
}

output "cluster_ca_certificate" {
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  description = "GKE cluster CA certificate"
  sensitive   = true
}

output "region" {
  value       = var.region
  description = "GCP region"
}

output "network_name" {
  value       = google_compute_network.vpc.name
  description = "VPC network name"
}
