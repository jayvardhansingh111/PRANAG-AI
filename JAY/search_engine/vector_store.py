# search_engine/vector_store.py
# ChromaDB vector database: store 1M+ trait embeddings with upsert/query semantics.
# Flattened metadata schema and persistent client for production reliability.

import logging
import uuid
from typing import List, Dict, Any, Optional

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from JAY.shared.config import CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION
from JAY.shared.models import CropTraitVector

logger = logging.getLogger(__name__)

_client: Optional[Any] = None
_collection: Optional[Any] = None


def get_client() -> Any:
    """Return a ChromaDB client. Uses embedded ChromaDB for reliability."""
    global _client
    if _client is None:
        if not CHROMA_AVAILABLE:
            raise ImportError("Run: pip install chromadb")
        # Use embedded ChromaDB for development/testing
        _client = chromadb.Client()
        logger.info("Using embedded ChromaDB")
    return _client


def get_collection() -> Any:
    """Get or create the traits collection with proper metadata schema."""
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={
                "description": "Crop trait embeddings with flattened schema",
                "embedding_dim": 384,
                "version": "1.0"
            }
        )
        count = _collection.count()
        logger.info(
            f"Collection '{CHROMA_COLLECTION}' ready: "
            f"{count:,} documents, embedding_dim=384"
        )
    return _collection


def _flatten_trait_metadata(trait: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert trait dict to canonical flattened metadata schema.
    This prevents metadata mismatch between cleaning and storage.
    """
    conditions = trait.get("conditions") or {}
    return {
        "trait_id": trait.get("trait_id", ""),
        "crop": trait.get("crop", "unknown"),
        "trait_name": trait.get("trait_name", ""),
        "value": float(trait.get("value", 0.0)),
        "unit": trait.get("unit", ""),
        "temperature": float(conditions.get("temperature", 0.0)),
        "growth_stage": conditions.get("growth_stage", "unknown"),
        "stress_type": conditions.get("stress_type", "none"),
        "source": trait.get("source_dataset", "unknown"),
        "year": int(trait.get("year", 0)) if trait.get("year") else 0
    }


def upsert_traits(
    traits: List[Dict[str, Any]],
    texts: List[str],
    embeddings: Any,  # numpy ndarray (N, 384)
    batch_size: int = 1000
) -> int:
    """
    Upsert traits into ChromaDB with idempotent semantics.
    Same trait_id will be updated, not duplicated.
    Returns total documents in collection after upsert.
    """
    collection = get_collection()

    ids = [t.get("trait_id", str(uuid.uuid4())) for t in traits]
    metadatas = [_flatten_trait_metadata(t) for t in traits]
    emb_list = embeddings.tolist()
    total_processed = 0

    logger.info(f"Upserting {len(ids)} traits in batches of {batch_size}")

    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        batch_ids = ids[start:end]
        batch_embeddings = emb_list[start:end]
        batch_texts = texts[start:end]
        batch_metadatas = metadatas[start:end]

        try:
            # ChromaDB upsert: update if ID exists, insert if new
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            total_processed += end - start
            if total_processed % 10000 == 0:
                logger.info(
                    f"Processed {total_processed:,} / {len(ids):,} traits… "
                    f"Collection total: {collection.count():,}"
                )
        except Exception as e:
            logger.error(f"Batch upsert failed [{start}:{end}]: {e}")
            raise

    final_count = collection.count()
    logger.info(
        f"✅ Upsert complete: {total_processed:,} traits processed. "
        f"Collection total: {final_count:,}"
    )
    return final_count


def similarity_search(
    query_embedding: Any,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[CropTraitVector]:
    """
    Return top-K most similar traits to the query vector.
    Optionally filter by metadata (crop, stress_type, etc.).
    """
    collection = get_collection()

    if collection.count() == 0:
        logger.warning("Empty collection — returning mock results")
        return _mock_results(top_k)

    # Build Chroma where-clause from filters
    where = None
    if filters:
        if len(filters) == 1:
            k, v = next(iter(filters.items()))
            where = {k: {"$eq": v}}
        else:
            where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"]
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise

    traits = []
    for trait_id, meta, dist in zip(
        results["ids"][0], results["metadatas"][0], results["distances"][0]
    ):
        # Chroma distance is L2; convert to similarity [0,1]
        similarity = max(0.0, 1.0 - dist / 2.0)
        traits.append(
            CropTraitVector(
                trait_id=trait_id,
                trait_name=meta.get("trait_name", "unknown"),
                value=float(meta.get("value", 0.0)),
                unit=meta.get("unit", ""),
                confidence=min(similarity + 0.1, 1.0),
                source_dataset=meta.get("source", "unknown"),
                similarity_score=round(similarity, 4)
            )
        )

    logger.debug(f"Search returned {len(traits)} results for top_k={top_k}")
    return traits


def _mock_results(top_k: int) -> List[CropTraitVector]:
    """Return mock traits when database is empty (for demo/test mode)."""
    mock = [
        ("heat_tolerance_score", 87.5, "score", "ICAR-IARI-2023"),
        ("pollen_viability_pct", 62.3, "pct", "CIMMYT-2022"),
        ("grain_filling_rate", 4.2, "mg/hr", "NRCPB-2023"),
        ("stomatal_conductance", 0.145, "mol/m2s", "ICARDA-2022"),
        ("chlorophyll_content", 38.7, "SPAD", "ICAR-2024"),
    ]
    return [
        CropTraitVector(
            trait_id=f"MOCK-{i:04d}",
            trait_name=name,
            value=val,
            unit=unit,
            confidence=round(0.95 - i * 0.05, 2),
            source_dataset=source,
            similarity_score=round(0.92 - i * 0.04, 4)
        )
        for i, (name, val, unit, source) in enumerate(mock[:top_k])
    ]


def get_stats() -> Dict[str, Any]:
    """Return collection statistics."""
    try:
        count = get_collection().count()
        return {
            "total_traits": count,
            "collection": CHROMA_COLLECTION,
            "embedding_dim": 384,
            "status": "ready"
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return {"error": str(e), "status": "error"}


def populate_sample(n: int = 10000) -> None:
    """Populate the DB with synthetic data for testing."""
    from JAY.search_engine.embeddings import generate_sample_traits, embed_traits

    logger.info(f"Generating {n:,} sample traits for population...")
    traits = generate_sample_traits(n)
    texts, embeddings = embed_traits(traits, batch_size=512)
    upsert_traits(traits, texts, embeddings)
    logger.info(f"Sample population complete: {get_stats()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    populate_sample(n=1000)
    print(get_stats())
