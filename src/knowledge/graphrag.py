"""Core GraphRAG classes for knowledge graph processing and retrieval.

This module contains the main classes for building knowledge graphs
and performing retrieval-augmented generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np

from src.data import EmbeddingCache, OpenAIEmbedder, Triple
from src.data.parsers import BaseDataParser
from src.config.settings import (
    get_azure_deployment,
    get_chat_model,
    is_azure_openai,
)

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Represents a knowledge graph built from triples."""

    def __init__(self, name: str = "KnowledgeGraph"):
        """Initialize knowledge graph.
        
        Args:
            name: Name of the graph for identification

        """
        self.name = name
        self.graph = nx.Graph()
        self._triples: list[Triple] = []

    def add_triples(self, triples: list[Triple]) -> None:
        """Add triples to the knowledge graph.
        
        Args:
            triples: List of (subject, predicate, object) tuples

        """
        for subject, predicate, obj in triples:
            # Add nodes
            self.graph.add_node(subject)
            self.graph.add_node(obj)

            # Add edge with predicate as edge attribute
            self.graph.add_edge(subject, obj, predicate=predicate)

            # Store triple
            self._triples.append((subject, predicate, obj))

    def load_from_parser(self, parser: BaseDataParser) -> None:
        """Load triples from a data parser.
        
        Args:
            parser: Data parser instance

        """
        triples = parser.parse()
        self.add_triples(triples)
        logger.info(f"Loaded {len(triples)} triples into {self.name}")

    @property
    def nodes(self) -> list[str]:
        """Get all nodes in the graph."""
        return list(self.graph.nodes)

    @property
    def triples(self) -> list[Triple]:
        """Get all triples in the graph."""
        return self._triples.copy()

    def get_triples_containing_nodes(self, nodes: list[str]) -> list[Triple]:
        """Get triples that contain any of the specified nodes.
        
        Args:
            nodes: List of node names to search for
            
        Returns:
            List of triples containing the nodes

        """
        node_set = set(nodes)
        return [
            (s, p, o) for s, p, o in self._triples
            if s in node_set or o in node_set
        ]

    def __len__(self) -> int:
        """Get number of nodes in the graph."""
        return len(self.graph.nodes)

    def __str__(self) -> str:
        """String representation of the graph."""
        return f"{self.name}: {len(self.graph.nodes)} nodes, {len(self._triples)} triples"


class GraphRetriever:
    """Handles retrieval from knowledge graphs using vector similarity."""

    def __init__(self,
                 embedder: OpenAIEmbedder,
                 cache: EmbeddingCache,
                 chunk_size: int = 2000):
        """Initialize graph retriever.
        
        Args:
            embedder: OpenAI embedder instance
            cache: Embedding cache instance
            chunk_size: Size of chunks for batch embedding

        """
        self.embedder = embedder
        self.cache = cache
        self.chunk_size = chunk_size

    def ensure_embeddings(self, graph: KnowledgeGraph) -> None:
        """Ensure all graph nodes have embeddings cached.
        
        Args:
            graph: Knowledge graph to process

        """
        missing_nodes = self.cache.get_missing_texts(graph.nodes)

        if not missing_nodes:
            logger.info(f"All {len(graph.nodes)} nodes already embedded")
            return

        if not self.embedder.is_available():
            logger.warning("OpenAI embedder not available - skipping embedding")
            return

        logger.info(f"Embedding {len(missing_nodes)} new node strings …")

        # Process in chunks
        for i in range(0, len(missing_nodes), self.chunk_size):
            chunk = missing_nodes[i:i + self.chunk_size]
            try:
                clean_texts, embeddings = self.embedder.embed_texts(chunk)

                # Store embeddings in cache
                for text, embedding in zip(clean_texts, embeddings):
                    normalized_embedding = self._normalize(embedding)
                    self.cache.set(text, normalized_embedding)

            except Exception:
                logger.exception("Embedding chunk failed")
                break

        # Save cache
        self.cache.save()
        logger.info(f"Cache now holds {len(self.cache)} embeddings")

    def retrieve_by_vector_similarity(self,
                                      graph: KnowledgeGraph,
                                      query: str,
                                      top_k: int = 15) -> list[Triple]:
        """Retrieve triples using vector similarity search.
        
        Args:
            graph: Knowledge graph to search
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of retrieved triples

        """
        if not self.embedder.is_available():
            logger.warning("Vector similarity not available, falling back to substring search")
            return self.retrieve_by_substring(graph, query, top_k)

        # Ensure all nodes are embedded
        self.ensure_embeddings(graph)

        # Get query embedding
        try:
            _, query_embeddings = self.embedder.embed_texts([query])
            if not query_embeddings:
                logger.warning("Could not embed query, falling back to substring search")
                return self.retrieve_by_substring(graph, query, top_k)

            query_embedding = self._normalize(query_embeddings[0])

        except Exception:
            logger.exception("Error embedding query")
            return self.retrieve_by_substring(graph, query, top_k)

        # Calculate similarities
        similarities = []
        for node in graph.nodes:
            node_embedding = self.cache.get(node)
            if node_embedding is not None:
                similarity = np.dot(query_embedding, node_embedding)
                similarities.append((similarity, node))

        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Log top results
        logger.info(f"Top {min(top_k, len(similarities))} nodes by similarity:")
        for i, (sim, node) in enumerate(similarities[:top_k]):
            logger.info(f"  {sim:.3f} → {node}")

        # Get top nodes and retrieve triples
        top_nodes = [node for _, node in similarities[:top_k]]
        retrieved_triples = graph.get_triples_containing_nodes(top_nodes)

        logger.info(f"Retrieved {len(retrieved_triples)} triples:")
        for s, p, o in retrieved_triples[:10]:  # Log first 10
            logger.info(f"  {s} --{p}--> {o}")

        return retrieved_triples

    def retrieve_by_substring(self,
                              graph: KnowledgeGraph,
                              query: str,
                              top_k: int = 15) -> list[Triple]:
        """Retrieve triples using substring matching (fallback method).
        
        Args:
            graph: Knowledge graph to search
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of retrieved triples

        """
        query_lower = query.lower()
        matching_nodes = []

        for node in graph.nodes:
            if query_lower in node.lower():
                matching_nodes.append(node)

        # Limit to top_k nodes
        top_nodes = matching_nodes[:top_k]
        retrieved_triples = graph.get_triples_containing_nodes(top_nodes)

        logger.info(f"Substring search found {len(matching_nodes)} matching nodes")
        logger.info(f"Retrieved {len(retrieved_triples)} triples")

        return retrieved_triples

    def retrieve_by_triple_similarity(self,
                                      graph: KnowledgeGraph,
                                      query: str,
                                      top_k: int = 15) -> list[Triple]:
        """Retrieve triples using semantic similarity between query and triple embeddings.

        This method embeds each triple as a combined text and calculates similarity
        directly with the query, providing more precise retrieval than node-based search.

        Args:
            graph: Knowledge graph to search
            query: Query text
            top_k: Number of top results to return

        Returns:
            List of retrieved triples sorted by semantic similarity

        """
        if not self.embedder.is_available():
            logger.warning("Vector similarity not available, falling back to substring search")
            return self.retrieve_by_substring(graph, query, top_k)

        return self._retrieve_with_triple_embeddings(graph, query, top_k)

    def _retrieve_with_triple_embeddings(self,
                                        graph: KnowledgeGraph,
                                        query: str,
                                        top_k: int) -> list[Triple]:
        """Helper method to retrieve triples using embeddings."""
        # Get all triples from the graph
        all_triples = graph.triples
        if not all_triples:
            logger.warning("No triples found in graph")
            return []

        # Create text representations and embed them
        triple_texts = [f"{s} {p} {o}" for s, p, o in all_triples]
        self._ensure_triple_embeddings(triple_texts)

        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        if query_embedding is None:
            return self.retrieve_by_substring(graph, query, top_k)

        # Calculate and sort similarities
        return self._rank_triples_by_similarity(all_triples, triple_texts, query_embedding, top_k)

    def _ensure_triple_embeddings(self, triple_texts: list[str]) -> None:
        """Ensure all triple texts have embeddings cached."""
        missing_texts = self.cache.get_missing_texts(triple_texts)

        if missing_texts:
            logger.info(f"Embedding {len(missing_texts)} new triple texts...")
            for i in range(0, len(missing_texts), self.chunk_size):
                chunk = missing_texts[i:i + self.chunk_size]
                try:
                    clean_texts, embeddings = self.embedder.embed_texts(chunk)
                    for text, embedding in zip(clean_texts, embeddings):
                        self.cache.set(text, self._normalize(embedding))
                except Exception:
                    logger.exception("Embedding chunk failed")
                    break
            self.cache.save()

    def _get_query_embedding(self, query: str) -> np.ndarray | None:
        """Get normalized embedding for query."""
        try:
            _, query_embeddings = self.embedder.embed_texts([query])
            if query_embeddings:
                return self._normalize(query_embeddings[0])
        except Exception:
            logger.exception("Error embedding query")
        return None

    def _rank_triples_by_similarity(self,
                                   all_triples: list[Triple],
                                   triple_texts: list[str],
                                   query_embedding: np.ndarray,
                                   top_k: int) -> list[Triple]:
        """Rank triples by similarity to query."""
        similarities = []
        for triple, triple_text in zip(all_triples, triple_texts):
            triple_embedding = self.cache.get(triple_text)
            if triple_embedding is not None:
                similarity = np.dot(query_embedding, triple_embedding)
                similarities.append((similarity, triple, triple_text))

        similarities.sort(reverse=True, key=lambda x: x[0])

        # Log top results
        logger.info(f"Top {min(top_k, len(similarities))} triples by semantic similarity:")
        for sim, _, triple_text in similarities[:top_k]:
            logger.info(f"  {sim:.3f} → {triple_text}")

        top_triples = [triple for _, triple, _ in similarities[:top_k]]
        logger.info(f"Retrieved {len(top_triples)} triples using semantic similarity")
        return top_triples

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector

        """
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


class LLMAnswerer:
    """Handles LLM-based answer generation from retrieved context."""

    def __init__(self, embedder: OpenAIEmbedder, model: str = "gpt-3.5-turbo"):
        """Initialize LLM answerer.

        Args:
            embedder: OpenAI embedder (reused for chat completions)
            model: Chat model to use (or deployment name for Azure)

        """
        self.embedder = embedder
        # For Azure, the model parameter will be the deployment name
        # The embedder already knows if it's using Azure
        self.model = model

    def answer_question(self, question: str, triples: list[Triple]) -> str:
        """Generate answer from question and retrieved triples.

        Args:
            question: Question to answer
            triples: Retrieved knowledge graph triples

        Returns:
            Generated answer

        Raises:
            RuntimeError: If OpenAI client is not available

        """
        if not self.embedder.is_available():
            msg = "OpenAI client not available for answer generation"
            raise RuntimeError(msg)

        # Format triples as context
        triples_text = "\n".join([
            f"{subject} --{predicate}--> {obj}"
            for subject, predicate, obj in triples
        ])

        prompt = (
            f"Context:\n{triples_text}\n\nQuestion: {question}\nAnswer:"
        )

        try:
            # Use the model/deployment name directly
            # For Azure, this should be the deployment name
            # For standard OpenAI, this is the model name
            # Note: gpt-5 only supports default temperature (1.0), so we don't set it
            chat = self.embedder.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return chat.choices[0].message.content.strip()

        except Exception:
            logger.exception("Error generating answer")
            raise


class GraphRAG:
    """Main GraphRAG system combining knowledge graphs, retrieval, and answer generation."""

    def __init__(self,
                 cache_file: str | Path = "embeddings.pkl",
                 api_key: str | None = None):
        """Initialize GraphRAG system.

        Args:
            cache_file: Path to embedding cache file
            api_key: OpenAI API key

        """
        self.cache = EmbeddingCache(cache_file)
        self.embedder = OpenAIEmbedder(api_key)
        self.retriever = GraphRetriever(self.embedder, self.cache)

        # Determine chat model/deployment name based on provider
        if is_azure_openai():
            # For Azure, use the chat deployment name
            chat_model = get_azure_deployment("chat") or "gpt-5"
        else:
            # For standard OpenAI, use the chat model name
            chat_model = get_chat_model()

        self.answerer = LLMAnswerer(self.embedder, model=chat_model)
        self.graphs: Dict[str, KnowledgeGraph] = {}

    def add_graph(self, name: str, parser: BaseDataParser) -> KnowledgeGraph:
        """Add a knowledge graph from a data parser.
        
        Args:
            name: Name for the graph
            parser: Data parser to load from
            
        Returns:
            Created knowledge graph

        """
        graph = KnowledgeGraph(name)
        graph.load_from_parser(parser)
        self.graphs[name] = graph
        return graph

    def query(self,
              graph_name: str,
              question: str,
              top_k: int = 15,
              include_answer: bool = True,  # noqa: FBT002
              ) -> dict:
        """Query a knowledge graph.
        
        Args:
            graph_name: Name of the graph to query
            question: Question to ask
            top_k: Number of top results to retrieve
            include_answer: Whether to generate LLM answer
            
        Returns:
            Dictionary with query results
            
        Raises:
            KeyError: If graph not found
            
        """
        if graph_name not in self.graphs:
            msg = f"Graph '{graph_name}' not found"
            raise KeyError(msg)

        graph = self.graphs[graph_name]

        # Retrieve relevant triples using semantic similarity
        triples = self.retriever.retrieve_by_triple_similarity(graph, question, top_k)

        result = {
            "graph": graph_name,
            "question": question,
            "triples": triples,
            "num_triples": len(triples)
        }

        # Generate answer if requested
        if include_answer and self.embedder.is_available():
            try:
                logger.info("\n---------- LLM ANSWER ----------")
                answer = self.answerer.answer_question(question, triples)
                result["answer"] = answer
                logger.info(answer)
            except Exception:
                logger.exception("Failed to generate answer")
                result["answer"] = None

        return result

    def list_graphs(self) -> dict[str, str]:
        """List all loaded graphs.
        
        Returns:
            Dictionary mapping graph names to descriptions
            
        """
        return {name: str(graph) for name, graph in self.graphs.items()}
