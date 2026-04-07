# search_engine/vector_store.py
# ChromaDB vector database: store 1M+ trait embeddings, run <50ms similarity search.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import uuid
from typing import List, Dict, Any, Optional

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from JAY.shared.config import CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION
from JAY.shared.models import CropTraitVector

_client     = None
_collection = None


def get_client():
    """Return a ChromaDB client. Tries HTTP server first, falls back to embedded."""
    global _client
    if _client is None:
        if not CHROMA_AVAILABLE:
            raise ImportError("Run: pip install chromadb")
        try:
            _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            _client.heartbeat()
            print(f"[STORE] Connected to ChromaDB server {CHROMA_HOST}:{CHROMA_PORT}")
        except Exception:
            print("[STORE] No server — using embedded ChromaDB.")
            _client = chromadb.Client()
    return _client


def get_collection():
    """Get or create the traits collection."""
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name     = CHROMA_COLLECTION,
            metadata = {"description": "Crop trait embeddings", "embedding_dim": 384}
        )
        print(f"[STORE] Collection '{CHROMA_COLLECTION}': {_collection.count():,} documents")
    return _collection


def store_traits(
    traits:     List[Dict[str, Any]],
    texts:      List[str],
    embeddings,                          # numpy ndarray (N, 384)
    batch_size: int = 1000
) -> int:
    """Store traits in ChromaDB. Returns total documents stored."""
    collection = get_collection()

    ids       = [t.get("trait_id", str(uuid.uuid4())) for t in traits]
    metadatas = [
        {
            "crop":         t.get("crop", "unknown"),
            "trait_name":   t.get("trait_name", ""),
            "value":        float(t.get("value", 0.0)),
            "unit":         t.get("unit", ""),
            "temperature":  float((t.get("conditions") or {}).get("temperature", 0.0)),
            "growth_stage": (t.get("conditions") or {}).get("growth_stage", ""),
            "stress_type":  (t.get("conditions") or {}).get("stress_type", "none"),
            "source":       t.get("source_dataset", "unknown")
        }
        for t in traits
    ]
    emb_list = embeddings.tolist()
    total    = 0

    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids        = ids[start:end],
            embeddings = emb_list[start:end],
            documents  = texts[start:end],
            metadatas  = metadatas[start:end]
        )
        total += end - start
        if total % 10000 == 0:
            print(f"[STORE] Stored {total:,} / {len(ids):,} traits…")

    print(f"[STORE] ✅ {total:,} traits stored. DB total: {collection.count():,}")
    return total


def similarity_search(
    query_embedding,
    top_k:   int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[CropTraitVector]:
    """
    Return top-K most similar traits to the query vector.
    Uses ChromaDB's HNSW index for <10ms search at 1M scale.
    """
    collection = get_collection()

    if collection.count() == 0:
        return _mock_results(top_k)

    t0 = time.time()
    where = None
    if filters and len(filters) == 1:
        k, v = next(iter(filters.items()))
        where = {k: {"$eq": v}}
    elif filters:
        where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=where,
        include=["metadatas", "distances"]
    )
    ms = (time.time() - t0) * 1000
    print(f"[STORE] Search: {ms:.1f}ms")
    if ms > 50:
        print("[STORE] ⚠️ >50ms — consider Qdrant or GPU embeddings at this scale.")

    traits = []
    for trait_id, meta, dist in zip(
        results["ids"][0], results["metadatas"][0], results["distances"][0]
    ):
        similarity = 1.0 - dist / 2.0
        traits.append(CropTraitVector(
            trait_id        = trait_id,
            trait_name      = meta.get("trait_name", "unknown"),
            value           = meta.get("value", 0.0),
            unit            = meta.get("unit", ""),
            confidence      = min(similarity + 0.1, 1.0),
            source_dataset  = meta.get("source", "unknown"),
            similarity_score = round(similarity, 4)
        ))
    return traits


def _mock_results(top_k: int) -> List[CropTraitVector]:
    """Return mock traits when the database is empty (testing mode)."""
    mock = [
        ("heat_tolerance_score", 87.5, "score",  "ICAR-IARI-2023"),
        ("pollen_viability_pct", 62.3, "pct",    "CIMMYT-2022"),
        ("grain_filling_rate",    4.2, "mg/hr",  "NRCPB-2023"),
        ("stomatal_conductance", 0.145, "mol/m2s","ICARDA-2022"),
        ("chlorophyll_content",  38.7, "SPAD",   "ICAR-2024"),
    ]
    return [
        CropTraitVector(
            trait_id=f"MOCK-{i:04d}", trait_name=name,
            value=val, unit=unit,
            confidence=round(0.95 - i * 0.05, 2),
            source_dataset=source,
            similarity_score=round(0.92 - i * 0.04, 4)
        )
        for i, (name, val, unit, source) in enumerate(mock[:top_k])
    ]


def get_stats() -> Dict[str, Any]:
    """Return collection statistics."""
    try:
        return {"total_traits": get_collection().count(),
                "collection": CHROMA_COLLECTION, "embedding_dim": 384}
    except Exception as e:
        return {"error": str(e)}


def populate_sample(n: int = 10000) -> None:
    """Populate the DB with synthetic data for testing."""
    from .embeddings import generate_sample_traits, embed_traits
    traits = generate_sample_traits(n)
    texts, embeddings = embed_traits(traits, batch_size=512)
    store_traits(traits, texts, embeddings)


if __name__ == "__main__":
    populate_sample(n=1000)
    print(get_stats())
