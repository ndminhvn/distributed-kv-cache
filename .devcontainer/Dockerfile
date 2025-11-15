FROM mcr.microsoft.com/devcontainers/python:3.14-bookworm

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install additional CLI tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    less \
    jq \
    httpie \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace
