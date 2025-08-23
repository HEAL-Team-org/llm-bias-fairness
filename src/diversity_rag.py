"""Diversity RAG System using CultureBank Dataset.

This module implements a simple RAG system for diversity enhancement using the CultureBank
dataset with keyword search and semantic similarity. It provides diversity-focused
recommendations to enhance prompts for better cultural representation.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .graphrag import EmbeddingCache, OpenAIEmbedder

logger = logging.getLogger(__name__)


@dataclass
class CultureBankRecord:
    """Data structure for CultureBank dataset records."""
    cultural_group: str
    context: str
    goal: str
    relation: str
    actor: str
    actor_behavior: str
    recipient: str
    recipient_behavior: str
    other_descriptions: str
    topic: str
    agreement: float
    eval_whole_desc: str
    eval_scenario: str
    eval_persona: str
    eval_question: str
    doc_text: str
    source: str  # reddit or tiktok
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure agreement is a float
        if isinstance(self.agreement, str):
            try:
                self.agreement = float(self.agreement)
            except (ValueError, TypeError):
                self.agreement = 0.0
        elif self.agreement is None:
            self.agreement = 0.0


class CultureBankParser:
    """Parser for CultureBank CSV files."""
    
    def __init__(self):
        """Initialize the parser."""
        self.records: List[CultureBankRecord] = []
        
    def parse_csv(self, file_path: Path, source: str) -> List[CultureBankRecord]:
        """Parse a CultureBank CSV file.
        
        Args:
            file_path: Path to CSV file
            source: Source identifier (reddit/tiktok)
            
        Returns:
            List of CultureBankRecord objects
        """
        try:
            df = pd.read_csv(file_path)
            
            # Create document text for each record
            df["doc_text"] = df.apply(self._make_doc_text, axis=1)
            
            records = []
            for _, row in df.iterrows():
                record = CultureBankRecord(
                    cultural_group=str(row.get("cultural group", "")),
                    context=str(row.get("context", "")),
                    goal=str(row.get("goal", "")),
                    relation=str(row.get("relation", "")),
                    actor=str(row.get("actor", "")),
                    actor_behavior=str(row.get("actor_behavior", "")),
                    recipient=str(row.get("recipient", "")),
                    recipient_behavior=str(row.get("recipient_behavior", "")),
                    other_descriptions=str(row.get("other_descriptions", "")),
                    topic=str(row.get("topic", "")),
                    agreement=row.get("agreement", 0.0),
                    eval_whole_desc=str(row.get("eval_whole_desc", "")),
                    eval_scenario=str(row.get("eval_scenario", "")),
                    eval_persona=str(row.get("eval_persona", "")),
                    eval_question=str(row.get("eval_question", "")),
                    doc_text=str(row.get("doc_text", "")),
                    source=source
                )
                records.append(record)
            
            logger.info(f"Parsed {len(records)} records from {file_path.name}")
            return records
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return []
    
    def _make_doc_text(self, row) -> str:
        """Create document text from row data."""
        parts = [
            f"Cultural group: {row.get('cultural group', '')}",
            f"Topic: {row.get('topic', '')}",
            f"Context: {row.get('context', '')}",
            f"Actor: {row.get('actor', '')} behaves as: {row.get('actor_behavior', '')}",
            f"Description: {row.get('eval_whole_desc', '')}"
        ]
        return " | ".join([p for p in parts if p and str(p) != "nan" and str(p) != "None"])


class DiversityRAG:
    """Simple RAG system for diversity enhancement using CultureBank dataset."""
    
    def __init__(
        self,
        cache_file: str | Path = "diversity_embeddings.pkl",
        api_key: str | None = None,
        top_k: int = 10,
        alpha: float = 0.6,  # weight for vector similarity
        beta: float = 0.3,   # weight for keyword match
        gamma: float = 0.1   # weight for agreement score
    ) -> None:
        """Initialize the Diversity RAG system.
        
        Args:
            cache_file: Path to cache embeddings
            api_key: OpenAI API key
            top_k: Number of top results to retrieve
            alpha: Weight for vector similarity
            beta: Weight for keyword matching
            gamma: Weight for agreement score
        """
        self.cache = EmbeddingCache(cache_file)
        self.embedder = OpenAIEmbedder(api_key)
        self.parser = CultureBankParser()
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.records: List[CultureBankRecord] = []
        self.doc_texts: List[str] = []
        self.agreement_scores: List[float] = []
        
    def load_datasets(
        self,
        reddit_file: str | Path = "data/cultural_values/culturebank_reddit.csv",
        tiktok_file: str | Path = "data/cultural_values/culturebank_tiktok.csv"
    ) -> bool:
        """Load CultureBank datasets from CSV files.
        
        Args:
            reddit_file: Path to Reddit CultureBank CSV
            tiktok_file: Path to TikTok CultureBank CSV
            
        Returns:
            True if datasets loaded successfully
        """
        try:
            reddit_path = Path(reddit_file)
            tiktok_path = Path(tiktok_file)
            
            self.records = []
            
            # Load Reddit data
            if reddit_path.exists():
                reddit_records = self.parser.parse_csv(reddit_path, "reddit")
                self.records.extend(reddit_records)
            else:
                logger.warning(f"Reddit file not found: {reddit_path}")
            
            # Load TikTok data
            if tiktok_path.exists():
                tiktok_records = self.parser.parse_csv(tiktok_path, "tiktok")
                self.records.extend(tiktok_records)
            else:
                logger.warning(f"TikTok file not found: {tiktok_path}")
            
            if not self.records:
                logger.error("No CultureBank records loaded")
                return False
            
            # Build document texts and agreement scores for retrieval
            self.doc_texts = [record.doc_text for record in self.records]
            self.agreement_scores = [record.agreement for record in self.records]
            
            # Normalize agreement scores (0-1 range)
            if self.agreement_scores:
                max_agreement = max(self.agreement_scores)
                min_agreement = min(self.agreement_scores)
                if max_agreement > min_agreement:
                    self.agreement_scores = [
                        (score - min_agreement) / (max_agreement - min_agreement)
                        for score in self.agreement_scores
                    ]
                else:
                    self.agreement_scores = [0.5] * len(self.agreement_scores)
            
            logger.info(f"Successfully loaded {len(self.records)} CultureBank records")
            
            # Generate embeddings if not cached
            self._ensure_embeddings_cached()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load CultureBank datasets: {e}")
            return False
    
    def _ensure_embeddings_cached(self) -> None:
        """Ensure all document texts have embeddings in the cache."""
        if not self.embedder.is_available():
            logger.warning("OpenAI embedder not available. Skipping embedding generation.")
            return
        
        # Find texts that need embeddings
        missing_texts = self.cache.get_missing_texts(self.doc_texts)
        
        if not missing_texts:
            logger.info("All diversity embeddings already cached")
            return
        
        logger.info(f"Generating embeddings for {len(missing_texts)} diversity documents...")
        
        try:
            # Generate embeddings in batches
            batch_size = 100
            for i in range(0, len(missing_texts), batch_size):
                batch = missing_texts[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(missing_texts) + batch_size - 1)//batch_size}")
                
                texts, embeddings = self.embedder.embed_texts(batch)
                
                # Cache the embeddings
                for text, embedding in zip(texts, embeddings):
                    self.cache.set(text, embedding)
            
            # Save cache
            self.cache.save()
            logger.info("Diversity embeddings generated and cached successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate diversity embeddings: {e}")
    
    def retrieve_diversity_examples(
        self,
        query_prompt: str,
        cultural_groups: List[str] | None = None,
        topics: List[str] | None = None
    ) -> List[Tuple[CultureBankRecord, float]]:
        """Retrieve diversity examples related to the query prompt.
        
        Args:
            query_prompt: The prompt to analyze for diversity enhancement
            cultural_groups: Optional filter for specific cultural groups
            topics: Optional filter for specific topics
            
        Returns:
            List of (record, score) tuples sorted by relevance
        """
        if not self.records:
            logger.warning("No CultureBank records loaded. Call load_datasets() first.")
            return []
        
        if not self.embedder.is_available():
            logger.warning("OpenAI embedder not available. Using fallback search.")
            return self._fallback_search(query_prompt, cultural_groups, topics)
        
        try:
            # Generate embedding for query
            query_texts, query_embeddings = self.embedder.embed_texts([query_prompt])
            query_embedding = query_embeddings[0]
            
            # Calculate combined scores for all records
            candidates = []
            for i, record in enumerate(self.records):
                # Apply filters
                if cultural_groups and record.cultural_group not in cultural_groups:
                    continue
                if topics and record.topic not in topics:
                    continue
                
                # Vector similarity
                doc_embedding = self.cache.get(record.doc_text)
                if doc_embedding is not None:
                    vector_sim = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                else:
                    vector_sim = 0.0
                
                # Keyword matching score
                keyword_score = self._keyword_overlap_score(query_prompt, record)
                
                # Agreement score (normalized)
                agreement_score = self.agreement_scores[i] if i < len(self.agreement_scores) else 0.0
                
                # Combined weighted score
                final_score = (
                    self.alpha * vector_sim +
                    self.beta * keyword_score +
                    self.gamma * agreement_score
                )
                
                candidates.append((record, float(final_score)))
            
            # Sort by score and return top-k
            candidates.sort(key=lambda x: x[1], reverse=True)
            results = candidates[:self.top_k]
            
            logger.info(f"Retrieved {len(results)} diversity examples")
            return results
            
        except Exception as e:
            logger.error(f"Error during diversity retrieval: {e}")
            return self._fallback_search(query_prompt, cultural_groups, topics)
    
    def _keyword_overlap_score(self, query: str, record: CultureBankRecord) -> float:
        """Compute keyword overlap score with record fields."""
        score = 0.0
        query_lower = query.lower()
        
        # Cultural group match (highest weight)
        if record.cultural_group and record.cultural_group.lower() in query_lower:
            score += 1.0
        
        # Topic match 
        if record.topic and record.topic.lower() in query_lower:
            score += 0.7
        
        # Context match
        if record.context and record.context.lower() in query_lower:
            score += 0.5
        
        # Actor behavior match
        if record.actor_behavior and record.actor_behavior.lower() in query_lower:
            score += 0.3
        
        # General keyword matches in description
        description = record.eval_whole_desc.lower()
        diversity_keywords = [
            "diverse", "multicultural", "inclusive", "cultural", "ethnicity",
            "tradition", "heritage", "community", "background", "identity"
        ]
        
        for keyword in diversity_keywords:
            if keyword in query_lower and keyword in description:
                score += 0.2
        
        return score
    
    def _fallback_search(
        self,
        query_prompt: str,
        cultural_groups: List[str] | None = None,
        topics: List[str] | None = None
    ) -> List[Tuple[CultureBankRecord, float]]:
        """Fallback search using keyword matching only."""
        logger.info("Using fallback keyword-based search for diversity")
        
        candidates = []
        for record in self.records:
            # Apply filters
            if cultural_groups and record.cultural_group not in cultural_groups:
                continue
            if topics and record.topic not in topics:
                continue
            
            # Simple keyword matching
            score = self._keyword_overlap_score(query_prompt, record)
            
            # Add small bonus for high agreement
            score += 0.1 * record.agreement
            
            if score > 0:
                candidates.append((record, score))
        
        # Sort by score and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:self.top_k]
    
    def get_diversity_recommendations(
        self,
        query_prompt: str,
        focus_areas: List[str] | None = None
    ) -> List[str]:
        """Get diversity enhancement recommendations for a prompt.
        
        Args:
            query_prompt: The prompt to enhance
            focus_areas: Optional focus areas for recommendations
            
        Returns:
            List of diversity recommendation strings
        """
        # Retrieve relevant examples
        examples = self.retrieve_diversity_examples(query_prompt)
        
        if not examples:
            return [
                "Consider including people from diverse cultural backgrounds",
                "Add representation from different age groups and ethnicities",
                "Include various socioeconomic backgrounds and abilities"
            ]
        
        recommendations = []
        cultural_groups_seen = set()
        topics_seen = set()
        
        for record, score in examples[:5]:  # Top 5 examples
            # Extract cultural group recommendations
            if record.cultural_group and record.cultural_group not in cultural_groups_seen:
                cultural_groups_seen.add(record.cultural_group)
                recommendations.append(
                    f"Include {record.cultural_group} cultural elements and perspectives"
                )
            
            # Extract topic-based recommendations
            if record.topic and record.topic not in topics_seen:
                topics_seen.add(record.topic)
                
                # Create contextual recommendations
                if record.actor_behavior:
                    recommendations.append(
                        f"Consider {record.topic.lower()} aspects: {record.actor_behavior}"
                    )
        
        # Deduplicate and limit recommendations
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:8]  # Max 8 recommendations
    
    def analyze_diversity_gaps(self, query_prompt: str) -> Dict[str, Any]:
        """Analyze diversity gaps in a prompt.
        
        Args:
            query_prompt: The prompt to analyze
            
        Returns:
            Dictionary with diversity gap analysis
        """
        examples = self.retrieve_diversity_examples(query_prompt)
        
        analysis = {
            "cultural_groups_represented": set(),
            "topics_covered": set(),
            "diversity_score": 0.0,
            "recommendations": [],
            "potential_enhancements": []
        }
        
        if not examples:
            analysis["diversity_score"] = 0.1
            analysis["recommendations"] = self.get_diversity_recommendations(query_prompt)
            return analysis
        
        # Analyze cultural representation
        for record, score in examples:
            if record.cultural_group:
                analysis["cultural_groups_represented"].add(record.cultural_group)
            if record.topic:
                analysis["topics_covered"].add(record.topic)
        
        # Calculate diversity score based on variety of representation
        cultural_diversity = len(analysis["cultural_groups_represented"])
        topic_diversity = len(analysis["topics_covered"])
        
        # Normalize scores (max 10 cultural groups, 15 topics roughly)
        cultural_score = min(cultural_diversity / 10.0, 1.0)
        topic_score = min(topic_diversity / 15.0, 1.0)
        
        analysis["diversity_score"] = (cultural_score + topic_score) / 2.0
        
        # Generate recommendations
        analysis["recommendations"] = self.get_diversity_recommendations(query_prompt)
        
        # Generate potential enhancements from high-scoring examples
        for record, score in examples[:3]:
            if score > 0.5 and record.eval_whole_desc:
                analysis["potential_enhancements"].append(
                    f"Incorporate {record.cultural_group} perspective: {record.eval_whole_desc[:150]}..."
                )
        
        # Convert sets to lists for JSON serialization
        analysis["cultural_groups_represented"] = list(analysis["cultural_groups_represented"])
        analysis["topics_covered"] = list(analysis["topics_covered"])
        
        return analysis
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded CultureBank dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.records:
            return {"error": "No dataset loaded"}
        
        cultural_groups = {}
        topics = {}
        sources = {}
        
        for record in self.records:
            # Count cultural groups
            if record.cultural_group:
                cultural_groups[record.cultural_group] = cultural_groups.get(record.cultural_group, 0) + 1
            
            # Count topics
            if record.topic:
                topics[record.topic] = topics.get(record.topic, 0) + 1
            
            # Count sources
            sources[record.source] = sources.get(record.source, 0) + 1
        
        return {
            "total_records": len(self.records),
            "total_cultural_groups": len(cultural_groups),
            "total_topics": len(topics),
            "cultural_groups": dict(sorted(cultural_groups.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_topics": dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]),
            "sources": sources,
            "cached_embeddings": len(self.cache),
            "average_agreement": sum(self.agreement_scores) / len(self.agreement_scores) if self.agreement_scores else 0.0
        }
