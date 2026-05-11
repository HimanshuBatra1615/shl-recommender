"""
Hybrid retrieval: ChromaDB (semantic) + BM25 (keyword), fused via RRF.
Built from catalog.json at startup.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from app.catalog import load_catalog

log = logging.getLogger(__name__)

CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "shl_assessments"

# Embedding model: lightweight, local, no API needed
EMBED_MODEL = "all-MiniLM-L6-v2"


def _build_doc_text(item: dict) -> str:
    """Create rich embedding text from catalog item."""
    type_labels = {
        "A": "Ability Aptitude cognitive reasoning",
        "B": "Biodata Situational Judgement behavior",
        "C": "Competencies",
        "D": "Development 360 feedback",
        "E": "Assessment Exercises",
        "K": "Knowledge Skills technical",
        "P": "Personality Behavior motivation",
        "S": "Simulations",
    }
    types = item.get("test_type", [])
    type_text = " ".join(type_labels.get(t, t) for t in types)
    levels = " ".join(item.get("job_levels", []))
    desc = item.get("description", "")[:500]
    name = item.get("name", "")

    return f"{name} | {type_text} | {levels} | {desc}".strip()


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """Combines ChromaDB semantic search with BM25 keyword search via RRF."""

    def __init__(self):
        self.catalog: list[dict] = []
        self.chroma_collection = None
        self.bm25: BM25Okapi | None = None
        self.doc_texts: list[str] = []
        self._initialized = False

    def initialize(self):
        """Build or load the ChromaDB index and BM25 index."""
        if self._initialized:
            return

        self.catalog = load_catalog()
        log.info(f"Building retrieval index for {len(self.catalog)} assessments...")

        # ChromaDB setup - try SentenceTransformer, fall back to default ONNX model
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))

        try:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBED_MODEL
            )
            ef(["test"])  # Verify it actually works
            log.info(f"Using SentenceTransformer embedding: {EMBED_MODEL}")
        except Exception as e:
            log.warning(f"SentenceTransformer unavailable ({e}), using default embedding")
            ef = embedding_functions.DefaultEmbeddingFunction()

        # Get or create collection
        try:
            self.chroma_collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
            )
            existing = self.chroma_collection.count()
            if existing == len(self.catalog):
                log.info(f"ChromaDB collection loaded ({existing} items, skipping rebuild)")
            else:
                log.info(f"ChromaDB count mismatch ({existing} vs {len(self.catalog)}), rebuilding...")
                client.delete_collection(COLLECTION_NAME)
                self.chroma_collection = self._build_chroma(client, ef)
        except Exception:
            self.chroma_collection = self._build_chroma(client, ef)

        # BM25 index
        self.doc_texts = [_build_doc_text(item) for item in self.catalog]
        tokenized = [_tokenize(t) for t in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)
        log.info("BM25 index built")

        self._initialized = True
        log.info("Hybrid retriever ready")

    def _build_chroma(self, client, ef) -> Any:
        """Build ChromaDB collection from scratch."""
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        # Batch upsert
        batch_size = 50
        for i in range(0, len(self.catalog), batch_size):
            batch = self.catalog[i : i + batch_size]
            ids = [str(i + j) for j in range(len(batch))]
            docs = [_build_doc_text(item) for item in batch]
            metas = [
                {
                    "name": item["name"],
                    "url": item["url"],
                    "test_type": ",".join(item.get("test_type", [])),
                    "job_levels": ",".join(item.get("job_levels", [])),
                    "duration": item.get("duration_minutes") or 0,
                    "remote": item.get("remote_testing", False),
                }
                for item in batch
            ]
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            log.info(f"  ChromaDB: upserted items {i}–{i+len(batch)}")

        log.info(f"ChromaDB collection built with {collection.count()} items")
        return collection

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Fuse semantic + BM25 results using Reciprocal Rank Fusion.
        k=60 is the standard RRF constant.
        """
        scores: dict[str, float] = {}
        url_to_item: dict[str, dict] = {}

        for rank, item in enumerate(semantic_results, 1):
            url = item["url"]
            scores[url] = scores.get(url, 0) + 1.0 / (k + rank)
            url_to_item[url] = item

        for rank, item in enumerate(bm25_results, 1):
            url = item["url"]
            scores[url] = scores.get(url, 0) + 1.0 / (k + rank)
            url_to_item[url] = item

        sorted_urls = sorted(scores, key=lambda u: scores[u], reverse=True)
        return [url_to_item[u] for u in sorted_urls]

    def search(
        self,
        query: str,
        top_k: int = 10,
        test_type_filter: list[str] | None = None,
    ) -> list[dict]:
        """
        Hybrid search: semantic + BM25, fused via RRF.
        Optionally filter by test_type codes.
        Returns up to top_k catalog items.
        """
        if not self._initialized:
            self.initialize()

        # ─── Semantic (ChromaDB) ───────────────────────────────────────────
        where_filter = None
        if test_type_filter:
            # ChromaDB doesn't support "contains" on string fields easily,
            # so we fetch more results and filter post-hoc
            pass

        chroma_results = self.chroma_collection.query(
            query_texts=[query],
            n_results=min(top_k * 3, self.chroma_collection.count()),
            include=["metadatas", "distances"],
        )

        semantic_items = []
        if chroma_results["metadatas"]:
            for meta in chroma_results["metadatas"][0]:
                semantic_items.append({
                    "name": meta["name"],
                    "url": meta["url"],
                    "test_type": meta["test_type"].split(",") if meta["test_type"] else [],
                    "job_levels": meta["job_levels"].split(",") if meta["job_levels"] else [],
                    "duration_minutes": meta.get("duration") or None,
                    "remote_testing": meta.get("remote", False),
                })

        # ─── BM25 ──────────────────────────────────────────────────────────
        query_tokens = _tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranked_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )
        bm25_items = []
        for idx in bm25_ranked_indices[: top_k * 3]:
            item = self.catalog[idx]
            bm25_items.append({
                "name": item["name"],
                "url": item["url"],
                "test_type": item.get("test_type", []),
                "job_levels": item.get("job_levels", []),
                "duration_minutes": item.get("duration_minutes"),
                "remote_testing": item.get("remote_testing", False),
            })

        # ─── Fuse ──────────────────────────────────────────────────────────
        fused = self._reciprocal_rank_fusion(semantic_items, bm25_items)

        # ─── Post-filter by test type ──────────────────────────────────────
        if test_type_filter:
            fused = [
                item for item in fused
                if any(t in item["test_type"] for t in test_type_filter)
            ]

        return fused[:top_k]

    def get_by_name(self, name: str) -> dict | None:
        """Find a specific assessment by name (for comparison requests)."""
        from app.catalog import get_item_by_name
        return get_item_by_name(name)


# Singleton
retriever = HybridRetriever()
