# Documentation Index

**LLM Bias & Fairness Project - Documentation Guide**

Welcome to the LLM Bias & Fairness Project documentation. This index will help you find the information you need quickly.

## 📖 Documentation Structure

We've organized all documentation into focused, up-to-date files:

### For All Users

#### **[README.md](../README.md)** - Start Here! 🚀
Your entry point to the project. Covers:
- Project overview and key features
- Quick start guide with unified CLI
- System architecture diagram
- Basic usage examples
- Migration guide from old scripts (v3.1)

**Read this first** to understand what the project does and get started quickly.

---

#### **[USER_GUIDE.md](USER_GUIDE.md)** - Complete Usage Reference 📚
Comprehensive guide for using all features. Includes:
- **NEW in v3.1:** Unified CLI with real-world examples
- Detailed enhancement mode comparisons
- Batch processing workflows with 5+ practical examples
- Knowledge query system with use cases
- Testing commands
- Visual bias evaluation guide
- Configuration options and best practices
- Troubleshooting and examples gallery

**Use this** when you need detailed instructions, want to use advanced features, or run into issues.

---

#### **[CLI_REFERENCE.md](CLI_REFERENCE.md)** - Command-Line Reference 📋 ⭐ NEW
Complete CLI reference documentation. Covers:
- All commands with full syntax
- Every parameter with descriptions and defaults
- Parameter ranges and valid values
- Real command examples for every option
- Quick reference tables
- Migration guide from old scripts

**Refer to this** when you need to look up specific command options, parameters, or see the exact syntax for a command.

---

### Configuration & Setup

#### **[OPENAI_BASE_URL_CONFIGURATION.md](OPENAI_BASE_URL_CONFIGURATION.md)** - Custom API Endpoints 🔌
Guide for configuring custom OpenAI-compatible API endpoints. Covers:
- Base URL configuration methods
- LocalAI setup examples
- Custom endpoint examples
- Environment variable usage
- Troubleshooting connection issues

**Use this** when working with LocalAI, custom OpenAI endpoints, or private API deployments.

---

#### **[AZURE_OPENAI_CONFIGURATION.md](AZURE_OPENAI_CONFIGURATION.md)** - Azure OpenAI Setup ☁️ ⭐ NEW
Comprehensive guide for Azure OpenAI Service integration. Covers:
- Azure OpenAI prerequisites and setup
- Configuration methods (file, environment, programmatic)
- Deployment name mapping
- Migration from standard OpenAI
- Practical examples and troubleshooting
- Best practices for production

**Use this** when deploying with Azure OpenAI Service for enterprise features, regional deployment, or Azure integration.

---

#### **[AZURE_OPENAI_INTEGRATION_SUMMARY.md](AZURE_OPENAI_INTEGRATION_SUMMARY.md)** - Implementation Details 📋 ⭐ NEW
Technical summary of Azure OpenAI integration. Covers:
- Complete change log of all files modified
- Configuration methods comparison
- Testing recommendations
- Migration path for existing users
- Known limitations and future enhancements

**Refer to this** for technical details about the Azure integration implementation or when troubleshooting Azure-specific issues.

---

### For Developers & Researchers

#### **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical Deep Dive 🏗️
Complete technical architecture and design. Covers:
- System components and their interactions
- Knowledge systems (StereoSet, CultureBank, GraphRAG)
- Enhancement pipeline details
- Visual bias evaluation implementation
- Data flow and design patterns
- Technology stack and scalability

**Refer to this** when you need to understand how the system works internally, want to extend it, or conduct research.

---

#### **[VISUAL_BIAS_EVALUATION.md](VISUAL_BIAS_EVALUATION.md)** - Demographic Analysis Guide 🔍
Specialized guide for visual bias evaluation. Covers:
- Setup and model requirements (FairFace, dlib)
- Demographic prediction categories
- Bias metrics explained (Bias-W, Bias-P, ENS, KL Divergence)
- Integration with batch processing
- Results interpretation
- Comparison workflows

**Consult this** when using visual bias evaluation features or analyzing demographic representation in generated images.

---

### Scripts & Migration

#### **[scripts/README.md](../scripts/README.md)** - Script Migration Guide 🔄
Migration guide for old individual scripts. Covers:
- Old script → new CLI command mappings
- Deprecation timeline
- Backward compatibility notes
- When to use old scripts

**Check this** if you were using old scripts and need to migrate to the unified CLI.

---

### Version History

#### **[CHANGELOG.md](../CHANGELOG.md)** - Version History 📝
Complete chronological record of all changes. Includes:
- **Version 3.1 (Current):** Unified CLI and script consolidation
- Version 3.0: Visual bias evaluation integration
- Version 2.0: Dual-pipeline enhancement system
- Version 1.0: Initial GraphRAG implementation

**Check this** to see what's new in each version or track the project's evolution.

---

## 🗺️ Quick Navigation by Task

### I want to...

**Get started quickly**
→ [README.md](../README.md) → Quick Start section

**See all available commands**
→ [CLI_REFERENCE.md](CLI_REFERENCE.md) → Overview

**Enhance a single prompt**
→ [USER_GUIDE.md](USER_GUIDE.md) → Enhancement Commands → Dual-Pipeline
→ [CLI_REFERENCE.md](CLI_REFERENCE.md) → Enhance Command

**Process multiple prompts from CSV**
→ [USER_GUIDE.md](USER_GUIDE.md) → Batch Processing (with examples)
→ [CLI_REFERENCE.md](CLI_REFERENCE.md) → Batch Command

**Look up a specific command parameter**
→ [CLI_REFERENCE.md](CLI_REFERENCE.md) → Find your command

**Research bias patterns or cultural values**
→ [USER_GUIDE.md](USER_GUIDE.md) → Knowledge Queries
→ [CLI_REFERENCE.md](CLI_REFERENCE.md) → Query Command

**Run tests to verify the system**
→ [USER_GUIDE.md](USER_GUIDE.md) → Testing Commands
→ [CLI_REFERENCE.md](CLI_REFERENCE.md) → Test Command

**Generate images with bias evaluation**
→ [USER_GUIDE.md](USER_GUIDE.md) → Visual Bias Evaluation
→ [VISUAL_BIAS_EVALUATION.md](VISUAL_BIAS_EVALUATION.md)

**Understand how the system works**
→ [ARCHITECTURE.md](ARCHITECTURE.md) → System Components

**Migrate from old scripts**
→ [scripts/README.md](../scripts/README.md) → Migration Guide
→ [CLI_REFERENCE.md](CLI_REFERENCE.md) → Migration Guide section

**Configure custom OpenAI endpoints**
→ [OPENAI_BASE_URL_CONFIGURATION.md](OPENAI_BASE_URL_CONFIGURATION.md)

**Set up Azure OpenAI Service**
→ [AZURE_OPENAI_CONFIGURATION.md](AZURE_OPENAI_CONFIGURATION.md)
→ [AZURE_OPENAI_INTEGRATION_SUMMARY.md](AZURE_OPENAI_INTEGRATION_SUMMARY.md) (technical details)

**Troubleshoot an issue**
→ [USER_GUIDE.md](USER_GUIDE.md) → Troubleshooting

**Extend the system**
→ [ARCHITECTURE.md](ARCHITECTURE.md) → Extensibility Points

**Understand bias metrics**
→ [VISUAL_BIAS_EVALUATION.md](VISUAL_BIAS_EVALUATION.md) → Evaluation Metrics

**See what's new**
→ [CHANGELOG.md](../CHANGELOG.md)

---

## 📂 File Organization

```
llm-bias-fairness/
├── README.md                          # Main entry point
├── CHANGELOG.md                       # Version history
│
├── docs/                              # All documentation
│   ├── INDEX.md                       # This file
│   ├── USER_GUIDE.md                  # Complete usage guide
│   ├── ARCHITECTURE.md                # Technical architecture
│   └── VISUAL_BIAS_EVALUATION.md      # Visual evaluation guide
│
├── src/                               # Source code
│   ├── stereoset_rag.py              # Bias detection
│   ├── diversity_rag.py              # Diversity enhancement
│   ├── graphrag.py                   # Knowledge graph
│   ├── image_generator.py            # Image generation
│   ├── visual_bias_evaluator.py      # Visual evaluation
│   └── parsers.py                    # Data parsers
│
├── enhance_prompt_dual_pipeline.py   # Main enhancement script
├── batch_processor.py                 # Batch processing pipeline
└── test_visual_bias_evaluation.py    # Evaluation test script
```

---

## 🎯 Documentation Philosophy

Our documentation follows these principles:

1. **Current & Accurate**: All docs reflect the actual current codebase
2. **Clear Organization**: Separate concerns (user guide vs architecture)
3. **Practical Examples**: Real-world usage examples throughout
4. **No Redundancy**: Each topic covered once in the right place
5. **Progressive Detail**: Start simple, go deeper as needed

---

## 📊 What Changed?

**Old Structure** (11 files, lots of overlap):
- ❌ DOCUMENTATION_INDEX.md
- ❌ PROJECT_OVERVIEW.md
- ❌ PROJECT_SUMMARY.md
- ❌ PROJECT_STATUS.md
- ❌ IMPLEMENTATION_SUMMARY.md
- ❌ PIPELINE_README.md
- ❌ RESTRUCTURING_SUMMARY.md
- ❌ SEMANTIC_SORTING_COMPLETE.md
- ❌ SEQUENTIAL_ENHANCEMENT.md
- ❌ IMAGE_PROMPT_ENHANCEMENT.md
- ❌ FINAL_STATUS.md
- ❌ REPORT.md
- ❌ DUAL_PIPELINE_COMPLETE.md

**New Structure** (4 focused files):
- ✅ README.md (overview + quick start)
- ✅ USER_GUIDE.md (complete usage guide)
- ✅ ARCHITECTURE.md (technical details)
- ✅ VISUAL_BIAS_EVALUATION.md (evaluation guide)
- ✅ CHANGELOG.md (version history)
- ✅ INDEX.md (this file)

**Benefits:**
- 📉 87% reduction in number of files
- ✨ No outdated content
- 🎯 Clear purpose for each document
- 🚀 Easier to maintain and update
- 💡 Better user experience

---

## 🔄 Keeping Docs Updated

When adding new features:
1. Update **USER_GUIDE.md** with usage instructions
2. Update **ARCHITECTURE.md** with technical details
3. Add entry to **CHANGELOG.md**
4. Update **README.md** if it affects quick start

When fixing bugs:
1. Update **USER_GUIDE.md** troubleshooting if relevant
2. Add entry to **CHANGELOG.md**

---

## 💬 Getting Help

1. **Start with**: [README.md](../README.md) for quick start
2. **Then check**: [USER_GUIDE.md](USER_GUIDE.md) for detailed instructions
3. **For deep dives**: [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
4. **Still stuck?**: Check [USER_GUIDE.md](USER_GUIDE.md) → Troubleshooting

---

**Last Updated**: October 1, 2025  
**Documentation Version**: 3.0
