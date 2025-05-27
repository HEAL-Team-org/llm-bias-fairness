"""GraphRAG demo - **embedding-aware** version.
==========================================
This file replaces the previous prototype.  The new bits are:

* 🔍 **Vector search** using **OpenAI `text-embedding-3-large`**.
* 💾 **Persistent cache** (`.pkl`) for every text string we embed - so we
  only pay once.
* 🪵 **Structured logging** of the retrieval step (both nodes & triples).

Quick start
-----------
```bash
pip install networkx pandas numpy openai  # plus faiss if you wish

export OPENAI_API_KEY="sk-…"
python graph_rag.py \
    "ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv" \
    --question "Why are black people stereotyped as criminals?" \
    --top_k 20
```

If the env-var is missing, the script falls back to the *old* substring
retriever so you can still test without an API key.

"""  # noqa: D205

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from openai import OpenAI

# Set up logger
logger = logging.getLogger(__name__)

# ---------- data types ----------
Triple = Tuple[str, str, str]
EmbDict = Dict[str, np.ndarray]
EMBED_MODEL = "text-embedding-3-large"
CACHE_FILENAME = "embeddings.pkl"
TRIPLE_PARTS_COUNT = 3

client = None
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ---------- helpers ----------


def load_graph_csv(csv_path: str | Path) -> list[Triple]:
    """Parse the CSV and collect `(subj, rel, obj)` triples."""
    graph_data_frame = pd.read_csv(csv_path)
    triples: list[Triple] = []
    for row in graph_data_frame["Graph"]:
        for raw in re.findall(r"\(([^)]*)\)", row):
            parts = [p.strip() for p in raw.split(",")[:TRIPLE_PARTS_COUNT]]
            if len(parts) == TRIPLE_PARTS_COUNT:
                triples.append(tuple(parts))  # type: ignore[arg-type]
    return triples


def normalise(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


# ---------- embedding cache ----------


def load_cache(path: Path) -> EmbDict:
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301
    return {}


def save_cache(path: Path, cache: EmbDict) -> None:
    with path.open("wb") as f:
        pickle.dump(cache, f)


# ---------- embedding via OpenAI ----------


def embed_texts(texts: list[str]) -> tuple[list[str], list[np.ndarray]]:
    """Call OpenAI once for up to 2048 texts; returns (clean_texts, embeddings)."""
    error_msg = "OpenAI client not initialized"
    if client is None:
        raise RuntimeError(error_msg)

    # Filter out empty strings and strip whitespace
    clean_texts = [text.strip() for text in texts if text.strip()]
    if not clean_texts:
        return [], []

    resp = client.embeddings.create(model=EMBED_MODEL, input=clean_texts)
    # OpenAI guarantees outputs in same order
    embeddings = [
        np.array(embedding.embedding, dtype=np.float32) for embedding in resp.data
    ]
    return clean_texts, embeddings


# ---------- graph construction ----------


def build_graph(triples: list[Triple]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for s, r, o in triples:
        graph.add_node(s, kind="entity")
        graph.add_node(o, kind="entity")
        graph.add_edge(s, o, relation=r)
    return graph


# ---------- retrieval ----------


def retrieve_with_vectors(
    graph: nx.MultiDiGraph,
    embeddings: EmbDict,
    query: str,
    k: int,
) -> list[Triple]:
    """Vector-based retriever; falls back to substring if missing embedding."""
    try:
        clean_texts, query_embeddings = embed_texts([query])
        if not clean_texts:
            return retrieve_by_substring(graph, query, k)
        q_vec = normalise(query_embeddings[0])
    except Exception as exc:
        logger.warning("[embedding] failed (%s) - reverting to text match", exc)
        return retrieve_by_substring(graph, query, k)

    # build matrix of node embeddings (only nodes that *have* an embedding)
    valid_nodes = [n for n in graph.nodes if n in embeddings]

    if not valid_nodes:
        logger.warning("No nodes have valid embeddings - reverting to text match")
        return retrieve_by_substring(graph, query, k)

    # Build the embedding matrix
    node_emb_pairs = [(n, embeddings[n]) for n in valid_nodes]
    nodes, mats = zip(*node_emb_pairs)
    mat_stack = np.stack(mats)  # stack into 2D matrix
    sims = mat_stack @ q_vec  # dot-product, vectors already unit length
    top_idx = np.argsort(sims)[-k:][::-1]

    chosen_nodes = [nodes[i] for i in top_idx]
    logger.info("Top %d nodes by similarity:", len(chosen_nodes))
    for n, score in zip(chosen_nodes, sims[top_idx]):
        logger.info("  %.3f → %s", score, n)

    triples: set[Triple] = set()
    for n in chosen_nodes:
        for _, tgt, key in graph.out_edges(n, keys=True):
            triples.add((n, graph.edges[n, tgt, key]["relation"], tgt))
        for src, _, key in graph.in_edges(n, keys=True):
            triples.add((src, graph.edges[src, n, key]["relation"], n))
    # truncate
    result = list(triples)[:k]
    logger.info("Retrieved %d triples:", len(result))
    for t in result:
        logger.info("  %s --%s--> %s", *t)
    return result


def retrieve_by_substring(graph: nx.MultiDiGraph, query: str, k: int) -> list[Triple]:
    needle = query.lower()
    hits = [n for n in graph.nodes if needle in n.lower()]
    logger.info("[fallback] Substring hits: %s", hits)
    triples: set[Triple] = set()
    for n in hits:
        for _, tgt, key in graph.out_edges(n, keys=True):
            triples.add((n, graph.edges[n, tgt, key]["relation"], tgt))
        for src, _, key in graph.in_edges(n, keys=True):
            triples.add((src, graph.edges[src, n, key]["relation"], n))
    return list(triples)[:k]


# ---------- LLM answer ----------


def answer_question(
    question: str, context: list[Triple], model: str = "gpt-4o-mini"
) -> str:
    error_msg = "OpenAI client not initialized"
    if client is None:
        raise RuntimeError(error_msg)

    triples_txt = "\n".join(f"{s} --{r}--> {o}" for s, r, o in context)
    prompt = (
        "You are an assistant answering strictly from the provided graph context.\n\n"
        f"Context:\n{triples_txt}\n\nQuestion: {question}\nAnswer:"
    )
    chat = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return chat.choices[0].message.content.strip()


# ---------- main ----------


def main() -> None:
    ap = argparse.ArgumentParser(description="GraphRAG embedding demo")
    ap.add_argument("csv_path", help="Path to the ADV_GRAPH CSV")
    ap.add_argument(
        "--question", "-q", default="What stereotypes exist about black people?"
    )
    ap.add_argument("--top_k", "-k", type=int, default=15)
    ap.add_argument("--cache", default=CACHE_FILENAME, help="Embedding cache file (pkl)")
    args = ap.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    triples = load_graph_csv(args.csv_path)
    graph = build_graph(triples)

    cache_path = Path(args.cache)
    emb_dict: EmbDict = load_cache(cache_path)

    # ----------------- ensure all node embeddings cached -----------------
    missing = [n for n in graph.nodes if n not in emb_dict]
    if missing:
        logger.info("Embedding %d new node strings …", len(missing))
        # split into chunks of 2048 to stay under the API limit
        chunk_size = 2000  # a bit of headroom
        for i in range(0, len(missing), chunk_size):
            chunk = missing[i : i + chunk_size]
            try:
                # Filter chunk to match what embed_texts will actually process
                clean_chunk, vecs = embed_texts(chunk)
                if not clean_chunk:
                    continue

                # Only map the texts that were actually processed
                for txt, vec in zip(clean_chunk, vecs):
                    emb_dict[txt] = normalise(vec)
            except Exception:
                logger.exception("Embedding chunk failed")
                break
        save_cache(cache_path, emb_dict)
        logger.info("Cache now holds %d embeddings", len(emb_dict))

    # ----------------- retrieval -----------------
    retrieved = retrieve_with_vectors(graph, emb_dict, args.question, args.top_k)

    # ----------------- LLM answer (optional) -----------------
    if os.getenv("OPENAI_API_KEY"):
        try:
            logger.info("\n---------- LLM ANSWER ----------")
            ans = answer_question(args.question, retrieved)
            logger.info(ans)
        except Exception:
            logger.exception("OpenAI ChatCompletion failed")
    else:
        logger.info("Set OPENAI_API_KEY to see an LLM answer.")


if __name__ == "__main__":
    main()
