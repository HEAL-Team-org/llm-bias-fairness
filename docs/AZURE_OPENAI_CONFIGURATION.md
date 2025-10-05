# Azure OpenAI Configuration Guide

This document provides comprehensive information about configuring and using Azure OpenAI services with the LLM Bias & Fairness project.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Configuration Methods](#configuration-methods)
- [Azure OpenAI Setup](#azure-openai-setup)
- [Deployment Names](#deployment-names)
- [Migration from Standard OpenAI](#migration-from-standard-openai)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The project supports both standard OpenAI API and Azure OpenAI Service. Azure OpenAI provides:

- Enterprise-grade security and compliance
- Regional deployment options
- Service-level agreements (SLAs)
- Integration with Azure ecosystem
- Custom deployment names for model management

## Prerequisites

Before configuring Azure OpenAI, ensure you have:

1. **Azure OpenAI Service Access**: An active Azure subscription with Azure OpenAI access
2. **Model Deployments**: Deployed models in your Azure OpenAI resource
3. **API Credentials**: Your Azure OpenAI API key and endpoint URL
4. **Python Package**: `openai` package version 1.0.0 or later

```bash
pip install openai>=1.0.0
```

## Configuration Methods

You can configure Azure OpenAI using any of these methods (in order of precedence):

### Method 1: Configuration File (Recommended)

Edit `config/config.yaml` (or create it if it doesn't exist):

```yaml
openai:
  provider: "azure"
  api_key: "15d65736dbcc47609d24fbd58af96bae"  # Your Azure API key
  
  azure:
    endpoint: "https://zanistagpteastus2.openai.azure.com/"
    api_version: "2024-12-01-preview"
    
    deployments:
      chat: "gpt-5"                          # Your chat completion deployment
      embedding: "text-embedding-3-large"     # Your embedding deployment
      image: null                             # Your DALL-E deployment (if available)
```

### Method 2: Environment Variables

Set environment variables (useful for production deployments):

```bash
# Provider selection
export OPENAI_PROVIDER="azure"

# Authentication
export OPENAI_API_KEY="15d65736dbcc47609d24fbd58af96bae"

# Azure-specific settings
export AZURE_OPENAI_ENDPOINT="https://zanistagpteastus2.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"

# Deployment names
export AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-5"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
export AZURE_OPENAI_IMAGE_DEPLOYMENT="dall-e-3"  # Optional
```

### Method 3: Programmatic Configuration

Configure Azure OpenAI directly in your code:

```python
from src.config.settings import get_config

config = get_config()
config.set("openai.provider", "azure")
config.set("openai.api_key", "your-api-key")
config.set("openai.azure.endpoint", "https://your-resource.openai.azure.com/")
config.set("openai.azure.api_version", "2024-12-01-preview")
config.set("openai.azure.deployments.chat", "gpt-5")
config.set("openai.azure.deployments.embedding", "text-embedding-3-large")
```

## Azure OpenAI Setup

### 1. Get Your Azure OpenAI Credentials

From the Azure Portal:

1. Navigate to your Azure OpenAI resource
2. Go to **Keys and Endpoint**
3. Copy:
   - **Endpoint URL** (e.g., `https://zanistagpteastus2.openai.azure.com/`)
   - **API Key** (either Key 1 or Key 2)

### 2. Check Your Deployments

From the Azure Portal:

1. Navigate to your Azure OpenAI resource
2. Go to **Model deployments** or **Deployments + Endpoints**
3. Note the **Deployment names** for:
   - Chat completion model (e.g., `gpt-5`, `gpt-4`)
   - Embedding model (e.g., `text-embedding-3-large`)
   - Image generation model (e.g., `dall-e-3`) - if available

### 3. Configure API Version

Azure OpenAI uses API versions for service updates. The recommended version is:

```yaml
api_version: "2024-12-01-preview"
```

Check [Azure OpenAI API versions](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference) for the latest version.

## Deployment Names

### Understanding Azure Deployments

Unlike standard OpenAI which uses model names directly (e.g., `gpt-4`, `text-embedding-3-large`), Azure OpenAI uses **deployment names** that you define when deploying models.

**Example:**
- **Standard OpenAI**: `model="gpt-4"`
- **Azure OpenAI**: `model="gpt-5"` (your deployment name)

### Deployment Name Mapping

Configure your deployment names in the configuration file:

```yaml
openai:
  azure:
    deployments:
      chat: "gpt-5"                        # Name of your chat model deployment
      embedding: "text-embedding-3-large"   # Name of your embedding deployment
      image: "dall-e-3"                     # Name of your image generation deployment
```

### Supported Deployment Types

| Type | Purpose | Example Deployment Name | Example Model |
|------|---------|------------------------|---------------|
| `chat` | Chat completions, Q&A | `gpt-5`, `my-gpt-4` | GPT-4, GPT-3.5-turbo |
| `embedding` | Text embeddings | `text-embedding-3-large` | text-embedding-3-large |
| `image` | Image generation | `dall-e-3` | DALL-E 3 |

## Migration from Standard OpenAI

### Before (Standard OpenAI)

```yaml
openai:
  provider: "openai"  # or omit this line (defaults to openai)
  api_key: "sk-..."
  embedding_model: "text-embedding-3-large"
  chat_model: "gpt-4o-mini"
  image_model: "dall-e-3"
```

### After (Azure OpenAI)

```yaml
openai:
  provider: "azure"
  api_key: "15d65736dbcc47609d24fbd58af96bae"
  
  azure:
    endpoint: "https://zanistagpteastus2.openai.azure.com/"
    api_version: "2024-12-01-preview"
    
    deployments:
      chat: "gpt-5"
      embedding: "text-embedding-3-large"
      image: null  # Set to your deployment name if you have DALL-E on Azure
```

### Code Changes Required

**None!** The codebase automatically detects the provider and uses the appropriate client:

```python
# This works for both OpenAI and Azure OpenAI
from src.data import OpenAIEmbedder

embedder = OpenAIEmbedder()  # Automatically uses Azure if configured
embeddings = embedder.embed_texts(["Hello world"])
```

```python
# Image generation also works for both
from src.generation.image_generator import create_image_generator

generator = create_image_generator("dalle3")  # Automatically uses Azure if configured
filepath, metadata = generator.generate_image("A beautiful sunset")
```

## Examples

### Example 1: Basic Azure OpenAI Usage

```python
from src.data import OpenAIEmbedder
from src.knowledge.graphrag import GraphRAG
from src.generation.image_generator import create_image_generator

# All of these automatically use Azure OpenAI when configured

# 1. Embeddings
embedder = OpenAIEmbedder()
texts = ["bias in AI", "fairness in machine learning"]
clean_texts, embeddings = embedder.embed_texts(texts)

# 2. Knowledge Graph RAG
graphrag = GraphRAG()
# ... use GraphRAG for Q&A (automatically uses Azure chat deployment)

# 3. Image Generation
generator = create_image_generator("dalle3")
filepath, metadata = generator.generate_image("diverse group of people")
```

### Example 2: Checking Provider Configuration

```python
from src.config.settings import (
    get_openai_provider,
    is_azure_openai,
    get_azure_openai_endpoint,
    get_azure_deployment
)

# Check which provider is configured
provider = get_openai_provider()
print(f"Using provider: {provider}")  # "azure" or "openai"

# Check if Azure is configured
if is_azure_openai():
    print(f"Azure endpoint: {get_azure_openai_endpoint()}")
    print(f"Chat deployment: {get_azure_deployment('chat')}")
    print(f"Embedding deployment: {get_azure_deployment('embedding')}")
```

### Example 3: Environment Variable Configuration

```bash
#!/bin/bash
# setup_azure.sh

# Configure Azure OpenAI via environment variables
export OPENAI_PROVIDER="azure"
export OPENAI_API_KEY="15d65736dbcc47609d24fbd58af96bae"
export AZURE_OPENAI_ENDPOINT="https://zanistagpteastus2.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-5"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-large"

# Run your application
python scripts/enhance_prompt.py
```

### Example 4: Switching Between Providers

You can easily switch between standard OpenAI and Azure OpenAI:

```python
from src.config.settings import get_config

config = get_config()

# Switch to Azure
config.set("openai.provider", "azure")

# Switch back to standard OpenAI
config.set("openai.provider", "openai")
```

## Troubleshooting

### Issue: "Azure OpenAI provider selected but no endpoint configured"

**Cause:** The provider is set to "azure" but the endpoint is not configured.

**Solution:**
```yaml
openai:
  provider: "azure"
  azure:
    endpoint: "https://zanistagpteastus2.openai.azure.com/"  # Add this
```

### Issue: "Azure OpenAI provider selected but no [type] deployment configured"

**Cause:** Missing deployment name for chat, embedding, or image generation.

**Solution:**
```yaml
openai:
  azure:
    deployments:
      chat: "gpt-5"                        # Add your chat deployment
      embedding: "text-embedding-3-large"   # Add your embedding deployment
```

### Issue: API calls fail with 404 error

**Possible causes:**
1. **Incorrect deployment name**: Verify deployment names in Azure Portal
2. **Incorrect endpoint**: Ensure endpoint URL matches your Azure resource
3. **Model not deployed**: Deploy the required model in Azure Portal

**Solution:**
1. Check deployment names: Azure Portal → Your Resource → Deployments
2. Verify endpoint: Azure Portal → Your Resource → Keys and Endpoint
3. Deploy required models if missing

### Issue: Authentication errors (401, 403)

**Possible causes:**
1. **Invalid API key**: Key may be incorrect or revoked
2. **Region mismatch**: Endpoint and key from different resources

**Solution:**
1. Regenerate API key in Azure Portal
2. Ensure endpoint and key are from the same Azure resource

### Issue: Rate limiting or quota exceeded

**Cause:** Azure OpenAI has rate limits and quotas per deployment.

**Solution:**
1. Check quota: Azure Portal → Your Resource → Quotas
2. Increase quota if needed (may require support request)
3. Implement retry logic with exponential backoff (already included in the code)

## Best Practices

1. **Use Environment Variables in Production**: Keep credentials out of configuration files
2. **Set Appropriate API Versions**: Use stable API versions for production
3. **Monitor Costs**: Azure OpenAI charges per token usage
4. **Use Deployment Names Wisely**: Use descriptive names (e.g., `gpt-4-production`, `gpt-4-dev`)
5. **Keep API Keys Secure**: Never commit API keys to version control

## Additional Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure OpenAI API Reference](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)
- [OpenAI Python SDK Documentation](https://github.com/openai/openai-python)
- [Project Configuration Guide](./USER_GUIDE_V2.md)

## Support

For issues specific to:
- **Azure OpenAI Service**: Contact Azure Support
- **This Project**: Open an issue on GitHub
- **OpenAI SDK**: See [OpenAI Python SDK Issues](https://github.com/openai/openai-python/issues)
