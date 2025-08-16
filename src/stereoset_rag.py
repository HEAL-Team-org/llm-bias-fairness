"""Simple RAG system for stereotype detection using StereoSet dataset.

This module implements a simple Retrieval-Augmented Generation system that uses
the StereoSet dataset to identify and avoid stereotypical content in prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .graphrag import EmbeddingCache, OpenAIEmbedder
from .parsers import StereoSetParser, StereoSetRecord

logger = logging.getLogger(__name__)


class StereoSetRAG:
    """Simple RAG system for stereotype detection using StereoSet dataset."""
    
    def __init__(
        self,
        cache_file: str | Path = "stereoset_embeddings.pkl",
        api_key: str | None = None,
        top_k: int = 10
    ) -> None:
        """Initialize the StereoSet RAG system.
        
        Args:
            cache_file: Path to cache embeddings
            api_key: OpenAI API key
            top_k: Number of top results to retrieve
        """
        self.cache = EmbeddingCache(cache_file)
        self.embedder = OpenAIEmbedder(api_key)
        self.parser = StereoSetParser()
        self.top_k = top_k
        self.contexts: List[str] = []
        self.context_to_record: Dict[str, StereoSetRecord] = {}
        
    def load_dataset(self) -> bool:
        """Load StereoSet dataset using the parser.
        
        Returns:
            True if dataset loaded successfully, False otherwise
        """
        try:
            logger.info("Loading StereoSet dataset...")
            success = self.parser.load_dataset()
            
            if not success or not self.parser.records:
                logger.error("Failed to load StereoSet dataset")
                return False
            
            # Build context mappings
            self.contexts = []
            self.context_to_record = {}
            
            for record in self.parser.records:
                self.contexts.append(record.context)
                self.context_to_record[record.context] = record
            
            logger.info(f"Successfully loaded {len(self.parser.records)} StereoSet records")
            
            # Generate embeddings if not cached
            self._ensure_context_embeddings()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load StereoSet dataset: {e}")
            return False
    
    def _ensure_context_embeddings(self) -> None:
        """Ensure all contexts have embeddings in the cache."""
        if not self.embedder.is_available():
            logger.warning("OpenAI embedder not available. Skipping embedding generation.")
            return
        
        # Find contexts that need embeddings
        missing_contexts = self.cache.get_missing_texts(self.contexts)
        
        if not missing_contexts:
            logger.info("All context embeddings already cached")
            return
        
        logger.info(f"Generating embeddings for {len(missing_contexts)} contexts...")
        
        try:
            # Generate embeddings in batches
            batch_size = 100
            for i in range(0, len(missing_contexts), batch_size):
                batch = missing_contexts[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(missing_contexts) + batch_size - 1)//batch_size}")
                
                texts, embeddings = self.embedder.embed_texts(batch)
                
                # Cache the embeddings
                for text, embedding in zip(texts, embeddings):
                    self.cache.set(text, embedding)
            
            # Save cache
            self.cache.save()
            logger.info("Context embeddings generated and cached successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate context embeddings: {e}")
    
    def retrieve_related_stereotypes(
        self,
        query_prompt: str,
        bias_types: List[str] | None = None
    ) -> List[Tuple[StereoSetRecord, float]]:
        """Retrieve stereotype records related to the query prompt.
        
        Args:
            query_prompt: The prompt to analyze for stereotypes
            bias_types: Optional filter for specific bias types
            
        Returns:
            List of (record, similarity_score) tuples
        """
        if not self.contexts:
            logger.warning("No contexts loaded. Call load_dataset() first.")
            return []
        
        if not self.embedder.is_available():
            logger.warning("OpenAI embedder not available. Using fallback search.")
            return self._fallback_search(query_prompt, bias_types)
        
        try:
            # Generate embedding for query
            query_texts, query_embeddings = self.embedder.embed_texts([query_prompt])
            query_embedding = query_embeddings[0]
            
            # Calculate similarities with cached context embeddings
            similarities = []
            for context in self.contexts:
                context_embedding = self.cache.get(context)
                if context_embedding is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, context_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(context_embedding)
                    )
                    similarities.append((context, float(similarity)))
            
            # Sort by similarity and get top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_contexts = similarities[:self.top_k]
            
            # Convert to records and filter by bias type if specified
            results = []
            for context, similarity in top_contexts:
                record = self.context_to_record[context]
                if bias_types is None or record.bias_type in bias_types:
                    results.append((record, similarity))
            
            logger.info(f"Retrieved {len(results)} related stereotype records")
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return self._fallback_search(query_prompt, bias_types)
    
    def _fallback_search(
        self,
        query_prompt: str,
        bias_types: List[str] | None = None
    ) -> List[Tuple[StereoSetRecord, float]]:
        """Fallback search using simple text matching."""
        logger.info("Using fallback keyword-based search")
        
        query_lower = query_prompt.lower()
        results = []
        
        for record in self.parser.records:
            if bias_types and record.bias_type not in bias_types:
                continue
            
            # Simple keyword matching
            context_lower = record.context.lower()
            target_lower = record.target.lower()
            
            score = 0.0
            # Check if query contains target or context keywords
            if target_lower in query_lower:
                score += 0.8
            if any(word in context_lower for word in query_lower.split()):
                score += 0.5
            
            if score > 0:
                results.append((record, score))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self.top_k]
    
    def get_negative_examples(
        self,
        query_prompt: str,
        bias_types: List[str] | None = None
    ) -> List[str]:
        """Get stereotypical sentences to avoid for the given prompt.
        
        Args:
            query_prompt: The prompt to analyze
            bias_types: Optional filter for specific bias types
            
        Returns:
            List of stereotypical sentences to avoid
        """
        related_records = self.retrieve_related_stereotypes(query_prompt, bias_types)
        
        negative_examples = []
        for record, similarity in related_records:
            stereotypical_sentences = record.stereotypical_sentences
            if stereotypical_sentences:
                negative_examples.extend(stereotypical_sentences)
                logger.debug(
                    f"Found {len(stereotypical_sentences)} stereotypical sentences "
                    f"for bias type {record.bias_type} (similarity: {similarity:.3f})"
                )
        
        # Remove duplicates while preserving order
        seen = set()
        unique_examples = []
        for example in negative_examples:
            if example not in seen:
                seen.add(example)
                unique_examples.append(example)
        
        logger.info(f"Collected {len(unique_examples)} unique negative examples")
        return unique_examples
    
    def analyze_prompt_for_stereotypes(
        self,
        prompt: str,
        threshold: float = 0.6
    ) -> Dict[str, any]:
        """Analyze a prompt for potential stereotypical content.
        
        Args:
            prompt: The prompt to analyze
            threshold: Similarity threshold for flagging potential issues
            
        Returns:
            Analysis results dictionary
        """
        related_records = self.retrieve_related_stereotypes(prompt)
        
        analysis = {
            "prompt": prompt,
            "potential_issues": [],
            "bias_types_detected": set(),
            "severity_score": 0.0,
            "recommendations": []
        }
        
        for record, similarity in related_records:
            if similarity >= threshold:
                issue = {
                    "bias_type": record.bias_type,
                    "target": record.target,
                    "context": record.context,
                    "similarity": similarity,
                    "stereotypical_sentences": record.stereotypical_sentences
                }
                analysis["potential_issues"].append(issue)
                analysis["bias_types_detected"].add(record.bias_type)
                analysis["severity_score"] = max(analysis["severity_score"], similarity)
        
        # Generate recommendations
        if analysis["potential_issues"]:
            analysis["recommendations"] = [
                f"Consider avoiding stereotypes related to {record.bias_type.lower()}"
                for record in [issue for issue in analysis["potential_issues"]]
            ]
            analysis["recommendations"].append(
                "Add diverse and inclusive language to counteract potential biases"
            )
        
        return analysis
    
    def get_dataset_stats(self) -> Dict[str, any]:
        """Get statistics about the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.parser.records:
            return {"error": "No dataset loaded"}
        
        bias_type_counts = {}
        total_stereotypical = 0
        total_anti_stereotypical = 0
        
        for record in self.parser.records:
            bias_type = record.bias_type
            bias_type_counts[bias_type] = bias_type_counts.get(bias_type, 0) + 1
            total_stereotypical += len(record.stereotypical_sentences)
            total_anti_stereotypical += len(record.anti_stereotypical_sentences)
        
        return {
            "total_records": len(self.parser.records),
            "total_contexts": len(self.contexts),
            "bias_types": bias_type_counts,
            "total_stereotypical_sentences": total_stereotypical,
            "total_anti_stereotypical_sentences": total_anti_stereotypical,
            "cached_embeddings": len(self.cache)
        }
