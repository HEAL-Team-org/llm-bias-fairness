"""Embedding management and caching for knowledge graphs.

This module handles OpenAI embeddings generation and persistent caching.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from openai import OpenAI

from src.config.settings import get_embedding_model, get_openai_api_key, get_openai_base_url

logger = logging.getLogger(__name__)

# Type definitions
EmbDict = Dict[str, np.ndarray]


class EmbeddingCache:
    """Manages persistent caching of embeddings."""

    def __init__(self, cache_file: str | Path = "embeddings.pkl"):
        """Initialize embedding cache.
        
        Args:
            cache_file: Path to the cache file

        """
        self.cache_file = Path(cache_file)
        self._cache: EmbDict = {}
        self.load()

    def load(self) -> None:
        """Load embeddings from cache file."""
        if self.cache_file.exists():
            try:
                with self.cache_file.open("rb") as f:
                    self._cache = pickle.load(f)
                logger.info(f"Loaded {len(self._cache)} embeddings from cache")
            except Exception:
                logger.exception(f"Error loading cache from {self.cache_file}")
                self._cache = {}
        else:
            logger.info("No existing cache found, starting fresh")

    def save(self) -> None:
        """Save embeddings to cache file."""
        try:
            with self.cache_file.open("wb") as f:
                pickle.dump(self._cache, f)
            logger.debug(f"Saved {len(self._cache)} embeddings to cache")
        except Exception:
            logger.exception(f"Error saving cache to {self.cache_file}")

    def get(self, text: str) -> np.ndarray | None:
        """Get embedding for text from cache.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Embedding array or None if not cached

        """
        return self._cache.get(text)

    def set(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache.
        
        Args:
            text: Text key
            embedding: Embedding array

        """
        self._cache[text] = embedding

    def get_missing_texts(self, texts: list[str]) -> list[str]:
        """Get list of texts that are not in cache.
        
        Args:
            texts: List of texts to check
            
        Returns:
            List of texts not in cache

        """
        return [text for text in texts if text not in self._cache]

    def __len__(self) -> int:
        """Get number of cached embeddings."""
        return len(self._cache)

    def __contains__(self, text: str) -> bool:
        """Check if text is in cache."""
        return text in self._cache


class OpenAIEmbedder:
    """Handles OpenAI embedding generation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None
    ):
        """Initialize OpenAI embedder.
        
        Args:
            api_key: OpenAI API key (if None, will use config/environment)
            model: Embedding model to use (if None, will use config default)
            base_url: Custom base URL for OpenAI API (if None, will use config/environment)

        """
        self.model = model or get_embedding_model()
        self.api_key = api_key or get_openai_api_key()
        self.base_url = base_url or get_openai_base_url()

        self.client = None
        if self.api_key:
            try:
                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                self.client = OpenAI(**client_kwargs)
            except Exception:
                logger.warning("OpenAI client initialization failed - embeddings disabled")
                self.client = None
        else:
            logger.warning("No OpenAI API key provided - embeddings disabled")

    def is_available(self) -> bool:
        """Check if OpenAI embeddings are available."""
        return self.client is not None

    def embed_texts(self, texts: list[str]) -> tuple[list[str], list[np.ndarray]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            Tuple of (clean_texts, embeddings)

        Raises:
            RuntimeError: If OpenAI client is not available

        """
        if not self.client:
            msg = "OpenAI client not available for embeddings"
            raise RuntimeError(msg)

        # Filter out empty texts
        clean_texts = [t.strip() for t in texts if t and t.strip()]
        if not clean_texts:
            return [], []

        try:
            response = self.client.embeddings.create(
                input=clean_texts,
                model=self.model
            )

            embeddings = [np.array(embedding.embedding) for embedding in response.data]
            return clean_texts, embeddings

        except Exception:
            logger.exception("Error generating embeddings")
            raise

    def embed_single(self, text: str) -> np.ndarray | None:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding array or None if failed

        """
        try:
            clean_texts, embeddings = self.embed_texts([text])
            return embeddings[0] if embeddings else None
        except Exception:
            logger.exception(f"Error embedding text: {text[:50]}...")
            return None


class CachedEmbedder:
    """Embedder with automatic caching support."""

    def __init__(self,
                 cache_file: str | Path = "embeddings.pkl",
                 api_key: str | None = None,
                 model: str | None = None):
        """Initialize cached embedder.
        
        Args:
            cache_file: Path to embedding cache file
            api_key: OpenAI API key
            model: Embedding model to use

        """
        self.cache = EmbeddingCache(cache_file)
        self.embedder = OpenAIEmbedder(api_key, model)

    def embed_texts(self,
                    texts: list[str],
                    auto_save: bool = True) -> tuple[list[str], list[np.ndarray]]:
        """Generate embeddings for texts, using cache when available.
        
        Args:
            texts: List of texts to embed
            auto_save: Whether to automatically save cache after generating new embeddings
            
        Returns:
            Tuple of (texts, embeddings)

        """
        # Check cache for existing embeddings
        embeddings_list = []
        texts_to_embed = []
        text_indices = []

        for idx, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue

            cached_emb = self.cache.get(text)
            if cached_emb is not None:
                embeddings_list.append((idx, text, cached_emb))
            else:
                texts_to_embed.append(text)
                text_indices.append(idx)

        logger.info(f"Cache hit: {len(embeddings_list)}/{len(texts)} texts")

        # Generate embeddings for missing texts
        if texts_to_embed:
            logger.info(f"Generating {len(texts_to_embed)} new embeddings...")
            clean_texts, new_embeddings = self.embedder.embed_texts(texts_to_embed)

            # Add to cache and result list
            for text, embedding, idx in zip(clean_texts, new_embeddings, text_indices):
                self.cache.set(text, embedding)
                embeddings_list.append((idx, text, embedding))

            if auto_save:
                self.cache.save()

        # Sort by original index and return
        embeddings_list.sort(key=lambda x: x[0])
        result_texts = [item[1] for item in embeddings_list]
        result_embeddings = [item[2] for item in embeddings_list]

        return result_texts, result_embeddings

    def embed_single(self, text: str, auto_save: bool = True) -> np.ndarray | None:
        """Generate embedding for a single text, using cache when available.
        
        Args:
            text: Text to embed
            auto_save: Whether to automatically save cache
            
        Returns:
            Embedding array or None if failed

        """
        text = text.strip()
        if not text:
            return None

        # Check cache first
        cached_emb = self.cache.get(text)
        if cached_emb is not None:
            return cached_emb

        # Generate new embedding
        embedding = self.embedder.embed_single(text)
        if embedding is not None:
            self.cache.set(text, embedding)
            if auto_save:
                self.cache.save()

        return embedding

    def is_available(self) -> bool:
        """Check if embedding generation is available."""
        return self.embedder.is_available()

    def __len__(self) -> int:
        """Get number of cached embeddings."""
        return len(self.cache)
