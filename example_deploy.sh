#!/bin/bash

# Example: How to use the automated deployment script
# This script shows how to set up environment variables and run the pipeline

# ============================================================
# BASIC USAGE EXAMPLE (Standard OpenAI)
# ============================================================
export DATA_FILE="/path/to/your/prompts.csv"
export OPENAI_API_KEY="your-openai-api-key-here"

python3 deploy_and_run.py


# ============================================================
# AZURE OPENAI MODE
# ============================================================
# export DATA_FILE="/path/to/your/prompts.csv"
# export OPENAI_PROVIDER="azure"
# export AZURE_OPENAI_API_KEY="your-azure-key"
# export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
# export AZURE_OPENAI_DEPLOYMENT="gpt-4"  # Optional: your deployment name
# export NUM_WORKERS="32"
# 
# python3 deploy_and_run.py


# ============================================================
# CUSTOM ENDPOINT EXAMPLE (for avalai.ir)
# ============================================================
# export DATA_FILE="/path/to/your/prompts.csv"
# export OPENAI_API_KEY="aa-zPA3KUsH8HFqC7TiF2J1dH1kMOkHqx4J2C9a5hY5CQX8eusE"
# export OPENAI_BASE_URL="https://api.avalai.ir/v1"
# export USE_PROXYCHAINS="true"
# export NUM_WORKERS="32"
# 
# python3 deploy_and_run.py


# ============================================================
# FULL CONFIGURATION EXAMPLE (Standard OpenAI)
# ============================================================
# export DATA_FILE="/home/user/dataset/large_prompts.csv"
# export OPENAI_API_KEY="sk-proj-..."
# export OPENAI_BASE_URL="https://api.avalai.ir/v1"
# export GITHUB_REPO="https://github.com/pooriyasafaei/llm-bias-fairness.git"
# export GITHUB_BRANCH="code-refactor"
# export NUM_WORKERS="64"
# export OUTPUT_DIR="custom_results"
# export USE_PROXYCHAINS="true"
# 
# python3 deploy_and_run.py


# ============================================================
# FULL AZURE CONFIGURATION EXAMPLE
# ============================================================
# export DATA_FILE="/home/user/dataset/large_prompts.csv"
# export OPENAI_PROVIDER="azure"
# export AZURE_OPENAI_API_KEY="your-azure-key"
# export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
# export AZURE_OPENAI_DEPLOYMENT="my-gpt4-deployment"
# export GITHUB_REPO="https://github.com/pooriyasafaei/llm-bias-fairness.git"
# export GITHUB_BRANCH="code-refactor"
# export NUM_WORKERS="64"
# export OUTPUT_DIR="azure_results"
# 
# python3 deploy_and_run.py


# ============================================================
# NOTES:
# ============================================================
# 1. Replace paths and API keys with your actual values
# 
# 2. Provider Modes:
#    - Standard OpenAI: Requires OPENAI_API_KEY
#    - Azure OpenAI: Requires OPENAI_PROVIDER="azure", AZURE_OPENAI_API_KEY, and AZURE_OPENAI_ENDPOINT
# 
# 3. The script will:
#    - Clone the repository from GitHub
#    - Create a virtual environment
#    - Install all dependencies
#    - Generate config.yaml automatically (provider-specific)
#    - Run the parallel pipeline
#    - Report results location
# 
# 4. Results will be in:
#    ./llm-bias-fairness-deployment/llm-bias-fairness/results/
# 
# 5. For proxychains support (standard OpenAI only), make sure proxychains4 is installed:
#    sudo apt-get install proxychains4
# 
# 6. Worker count recommendations:
#    - 8GB RAM: Use 2-4 workers
#    - 16GB RAM: Use 4-8 workers
#    - 32GB RAM: Use 8-16 workers
#    - 64GB+ RAM: Use 32+ workers

