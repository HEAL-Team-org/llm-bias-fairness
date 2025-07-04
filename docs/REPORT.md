# Comprehensive Project Report: GraphRAG for Bias Mitigation and Image Prompt Enhancement

**Version**: 1.0  
**Date**: July 4, 2025

---

## 1. Executive Summary

This document provides a comprehensive overview of the **LLM Bias & Fairness Project**. The project's primary achievement is the development of a sophisticated system that leverages Graph Retrieval-Augmented Generation (GraphRAG) to analyze and mitigate bias in AI. Its flagship feature is a powerful tool for enhancing image generation prompts to ensure they are diverse, inclusive, and culturally aware.

The project began as a monolithic script and evolved into a modular, production-ready Python application. It integrates large-scale knowledge graphs, cutting-edge vector embeddings for semantic search, and a multi-agent LLM system to deliver its capabilities.

**Key Accomplishments:**
*   **Modular Architecture**: The system was successfully refactored into a clean, class-based architecture, separating data parsing, graph management, embedding, and application logic.
*   **Multi-Source Knowledge Graph Integration**: The system ingests and processes over 57,000 triples from disparate sources, including a large bias/stereotype dataset and 15 distinct cultural value datasets.
*   **Advanced Semantic Retrieval**: Implemented a state-of-the-art retrieval mechanism that performs semantic similarity searches on entire graph triples, yielding highly relevant and contextual results.
*   **Dual Image Prompt Enhancement Systems**:
    1.  A **basic, single-pass tool** for rapid prompt improvement.
    2.  An **advanced, sequential system** featuring two collaborating AI agents—an Enhancement Agent and a Diversity Scoring Agent—that iteratively refine a prompt until it meets a configurable quality standard.
*   **Robustness and Efficiency**: The system features persistent embedding caching to minimize API costs and latency, and a graceful fallback to substring search if the OpenAI API is unavailable.

This report details the project's journey, architectural decisions, technical implementation, and the AI-driven methodologies that power its core features.

---

## 2. Project Genesis and Objectives

The proliferation of large-scale generative AI models has exposed a critical challenge: these models often reproduce and amplify societal biases present in their training data. An image generation model prompted with "a doctor" might disproportionately generate images of white men, while a prompt for "a nurse" might yield mostly white women.

This project was initiated to address this problem directly.

**Primary Objectives:**
1.  **Build a Knowledge-Grounded System**: To create a system that could reason about bias and culture using structured knowledge, rather than relying solely on the opaque knowledge of a base LLM.
2.  **Develop a Bias Mitigation Tool**: To create a practical tool that could actively mitigate bias in a real-world application—image generation.
3.  **Promote Inclusivity and Diversity**: To go beyond simply *removing bias* and actively *injecting diversity* by leveraging knowledge about global cultural values.
4.  **Ensure Technical Excellence**: To build a system that is not only effective but also modular, efficient, and extensible for future research and development.

---

## 3. System Architecture Deep Dive

The system's architecture is designed for a clear, linear flow of data, from raw information to intelligent, augmented output.

```
                    📊 Data Sources
                         │
           ┌─────────────┼─────────────┐
           │             │             │
    🚫 Bias Data    🌍 Cultural      📝 User Input
    (51,301 triples)  Values         (Image Prompt)
                    (6,469 triples)
           │             │             │
           └─────────────┼─────────────┘
                         │
               🔧 GraphRAG Core System
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   💾 Embedding      🔍 Similarity     🤖 LLM
   Cache System      Search Engine    Integration
        │                │                │
        └────────────────┼────────────────┘
                         │
           🎨 Enhancement Systems
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   📝 Basic          🔄 Sequential       📊 Diversity
   Enhancement       Enhancement        Scoring Agent
   (Single Pass)     (Multi-Iteration)   (7 Dimensions)
```

### Concept Explainer: Knowledge Graphs

A **Knowledge Graph** is a way of representing data as a network of entities and their relationships. In this project:
*   **Nodes**: Represent entities like "doctors," "women," or cultural concepts like "family dinner."
*   **Edges**: Represent the relationship (the predicate) between two nodes.
*   **Triple**: A single unit of knowledge is a `(subject, predicate, object)` triple, such as `(women, are stereotyped as, nurses)`.

By structuring data this way, the system can traverse relationships and retrieve contextually rich information that is far more powerful than simple keyword matching.

---

## 4. Core Engine: The GraphRAG Framework (`src/`)

The core of the project resides in the `src` directory, which contains the reusable and modular GraphRAG engine.

### 4.1. Data Ingestion: `src/parsers.py`

This module is responsible for reading raw data files and converting them into a standardized list of triples.
*   **`BaseDataParser`**: An abstract base class that defines the interface for all parsers, ensuring they have a `.parse()` method.
*   **`DataParserFactory`**: A factory class that automatically selects the correct parser based on the input file's name and extension. This makes the system extensible—to support a new data format (e.g., JSON-LD), one only needs to create a new parser class.
*   **`BiasCSVParser`**: Parses the project's primary bias dataset, extracting triples from a specific `Graph` column.
*   **`CulturalTriplesParser`**: A more general-purpose parser for simple `.txt` files where each line represents a `subject,predicate,object` triple.

### 4.2. The GraphRAG Core: `src/graphrag.py`

This is the heart of the engine, containing the classes that manage the entire RAG pipeline.

*   **`KnowledgeGraph`**: A wrapper around the `NetworkX` library that constructs the graph from triples and provides methods for accessing nodes and triples.
*   **`EmbeddingCache`**: A critical performance and cost-saving component. It serializes text embeddings to a pickle file (`embeddings.pkl`), ensuring that the expensive operation of generating an embedding for a given piece of text only happens once. On subsequent runs, the cached embedding is loaded instantly from disk.
*   **`OpenAIEmbedder`**: This class is the interface to the OpenAI API. It handles the generation of vector embeddings using the `text-embedding-3-large` model and also provides access to the chat completion API for the LLM agents.

### Concept Explainer: Vector Embeddings and Semantic Search

Traditional search relies on keywords. If you search for "physician," you won't find documents that only use the word "doctor." **Semantic search** solves this by understanding the *meaning* behind the words.

This is achieved with **Vector Embeddings**:
1.  A piece of text (e.g., the triple "women are stereotyped as nurses") is fed into an embedding model like OpenAI's `text-embedding-3-large`.
2.  The model converts this text into a high-dimensional vector (a list of numbers).
3.  Texts with similar meanings will have vectors that are "close" to each other in this high-dimensional space.

The `GraphRetriever` calculates the **cosine similarity** between the user's query vector and the vectors of all triples in the knowledge graph. By finding the vectors with the highest similarity score, it can retrieve the most semantically relevant information, even if the wording is completely different.

*   **`GraphRetriever`**: This class orchestrates the search process. Its key method, `retrieve_by_triple_similarity`, embeds the user's query, compares it against the pre-computed (and cached) embeddings of all triples, and returns the top-K most relevant triples. It also contains the crucial fallback logic to perform a simple substring search if embeddings are unavailable.
*   **`LLMAnswerer`**: A component that takes the retrieved triples and a user question, and uses the LLM to synthesize a coherent, human-readable answer.

---

## 5. Feature Deep Dive: Image Prompt Enhancement

This is the project's flagship application, demonstrating the power of the GraphRAG engine. It exists in two forms.

### 5.1. Basic Enhancement (`enhance_prompt.py`)
This script provides a straightforward, single-pass enhancement.
1.  User provides a prompt (e.g., "engineers working").
2.  The system retrieves relevant bias triples (e.g., stereotypes about engineers being male) and cultural triples.
3.  This context is fed to the LLM with a one-shot instruction to enhance the prompt for diversity.
4.  The LLM returns a single, improved prompt.

### 5.2. Sequential Enhancement (`enhance_prompt_sequential.py`)
This is a far more advanced, multi-agent system that provides quality assurance.

### Concept Explainer: Agent-Based AI Systems

Instead of using a single, monolithic LLM call, this system uses two specialized **AI agents**. An agent is an LLM equipped with a specific persona, instructions, and tools to perform a narrow task exceptionally well. By collaborating, these specialized agents can solve complex problems more effectively than a single generalist agent.

The sequential workflow is as follows:
1.  The **Enhancement Agent** performs an initial enhancement, just like in the basic script.
2.  The output is then passed to the **Diversity Scoring Agent**.
3.  This agent evaluates the enhanced prompt against a detailed rubric and returns a score (0-100) and specific feedback.
4.  If the score is below a configurable threshold (e.g., 75), the workflow loops. The original prompt, the retrieved context, the failed prompt, and the specific feedback from the scoring agent are all fed back to the **Enhancement Agent**.
5.  This iterative process continues until the diversity score meets the threshold or a maximum number of iterations is reached.

### 5.3.1. The Diversity Scoring Agent

*   **Purpose**: To provide objective, structured, and actionable feedback on the quality of an enhanced prompt.
*   **The 7 Scoring Dimensions**:
    1.  **Age Diversity (15 pts)**
    2.  **Ethnic/Racial Diversity (20 pts)**
    3.  **Gender Diversity (15 pts)**
    4.  **Cultural Diversity (20 pts)**
    5.  **Ability Inclusion (10 pts)**
    6.  **Socioeconomic Diversity (10 pts)**
    7.  **Specificity (10 pts)**
*   **System Prompt**:
    ```
    You are a Diversity and Inclusion Scoring Agent. Your task is to evaluate an image generation prompt based on a set of diversity dimensions and provide a score from 0 to 100. You must return your response in JSON format only.

    The JSON output must contain:
    - "total_score": An integer from 0 to 100.
    - "strengths": A list of strings describing the prompt's diversity strengths.
    - "weaknesses": A list of strings describing areas for improvement.
    - "scores": A dictionary with scores for each dimension.

    Scoring Dimensions:
    1.  **Age Diversity (0-15 points)**: Representation of multiple age groups (children, adults, seniors).
    2.  **Ethnic/Racial Diversity (0-20 points)**: Representation of various ethnic and racial backgrounds and skin tones.
    3.  **Gender Diversity (0-15 points)**: Representation beyond a single gender, including non-binary identities.
    4.  **Cultural Diversity (0-20 points)**: Inclusion of diverse cultural elements (clothing, settings, practices).
    5.  **Ability Inclusion (0-10 points)**: Representation of people with visible or invisible disabilities.
    6.  **Socioeconomic Diversity (0-10 points)**: Representation of different socioeconomic backgrounds.
    7.  **Specificity (0-10 points)**: How specific and actionable the diversity instructions are.

    Analyze the prompt and assign points for each dimension based on how well it addresses them. The total score is the sum of all dimension scores.
    ```

### 5.3.2. The Prompt Enhancement Agent

*   **Purpose**: To creatively rewrite a prompt to be more inclusive, using retrieved knowledge and (on subsequent iterations) feedback from the scoring agent.
*   **Initial Pass System Prompt**:
    ```
    You are an expert in AI-driven image generation, specializing in enhancing prompts for diversity, inclusivity, and bias mitigation. Your goal is to rewrite a user's prompt to be more descriptive, vivid, and representative of a global audience, while preserving the original intent.

    **Your Task:**
    1.  **Analyze the user's prompt** and the provided contextual data.
    2.  **Use the 'Bias Information'** to understand what stereotypes to actively avoid.
    3.  **Use the 'Cultural Values'** to incorporate positive, authentic, and diverse cultural elements.
    4.  **Rewrite the prompt** to be highly descriptive and explicitly inclusive across multiple dimensions (age, gender, ethnicity, ability, culture).
    5.  **Return ONLY the enhanced prompt text**, without any extra explanations, greetings, or markdown formatting.
    ```
*   **Iterative Pass System Prompt (Contextual Addition)**: When an iteration fails, the following context is added to the user message to guide the agent's next attempt.
    ```
    **Previous Attempt Analysis:**
    - **Previously Generated Prompt:** {previous_attempt['prompt']}
    - **Diversity Score:** {previous_attempt['score']}/100
    - **Areas for Improvement:** {', '.join(previous_attempt['weaknesses'])}

    **Your New Goal:**
    Address the 'Areas for Improvement' to increase the diversity score in the next version.
    ```

---

## 6. Data Sources

The system's intelligence is grounded in its data.
*   **Bias & Stereotype Data**:
    *   **Source**: `data/biases/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv`
    *   **Content**: 51,301 triples covering a vast range of stereotypes related to race, gender, profession, and other demographics.
    *   **Usage**: This knowledge is used as a "what to avoid" guide for the enhancement agent.
*   **Cultural Values Data**:
    *   **Source**: 15 `.txt` files in `data/cultural_values/`.
    *   **Content**: 6,469+ triples covering cultural practices, foods, traditions, and values from regions including Iran, India, China, Korea, the US, and several African and European nations.
    *   **Usage**: This knowledge provides the raw material for injecting positive, authentic diversity into the prompts.

---

## 7. Usage Guide

The system is designed to be run from the command line.

**Installation:**
```bash
# Clone the repository
git clone <repository-url>
cd llm-bias-fairness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (optional)
export OPENAI_API_KEY="sk-your-api-key-here"
```

**CLI Examples:**
```bash
# General-purpose graph queries (interactive)
python main.py

# Basic prompt enhancement (interactive)
python enhance_prompt.py

# Sequential prompt enhancement with custom settings
python enhance_prompt_sequential.py \
    --prompt "a team of scientists in a lab" \
    --threshold 85 \
    --max-iterations 4
```

---

## 8. Conclusion and Future Work

This project has successfully delivered a robust and innovative system for mitigating bias in AI. It demonstrates the power of combining structured knowledge graphs with the generative capabilities of LLMs. The final product is a testament to a well-executed development process, from initial concept to a fully-featured, modular application.

**Potential Future Work:**
*   **Web Interface**: Creating a user-friendly web UI to make the tool accessible to non-technical users.
*   **Expanded Knowledge Base**: Integrating more diverse knowledge graphs covering a wider range of biases and cultures.
*   **Real-time Feedback Loop**: Connecting the system directly to an image generation API to score the *output image* for diversity, not just the prompt.
*   **Plugin Architecture**: Developing a plugin model to allow easy integration into other content creation tools.
