# search_engine/similarity_search.py
# Public search API called by the orchestrator workflow.
# Input: query string  Output: top-K CropTraitVectors in <50ms

import time
from typing import List, Optional, Dict

from JAY.shared.models import CropTraitVector
from JAY.shared.config import SEARCH_TOP_K


def search_traits(
    query:          str,
    top_k:          int = None,
    crop_filter:    Optional[str] = None,
    min_similarity: float = 0.3
) -> List[CropTraitVector]:
    """
    Semantic similarity search over the trait vector database.

    Args:
        query:          Natural language condition string
        top_k:          Max results to return
        crop_filter:    Restrict search to a single crop type
        min_similarity: Drop results below this score

    Returns:
        List of CropTraitVector sorted by similarity descending.
    """
    top_k = top_k or SEARCH_TOP_K
    t0    = time.time()

    from .embeddings import embed_query
    query_vec = embed_query(query)

    filters = {"crop": crop_filter.lower()} if crop_filter else None

    from .vector_store import similarity_search
    raw = similarity_search(query_vec, top_k=top_k * 2, filters=filters)

    # Filter + deduplicate by trait_name (keep highest similarity per name)
    best: Dict[str, CropTraitVector] = {}
    for t in raw:
        if t.similarity_score >= min_similarity:
            if t.trait_name not in best or t.similarity_score > best[t.trait_name].similarity_score:
                best[t.trait_name] = t

    results = sorted(best.values(), key=lambda x: x.similarity_score, reverse=True)[:top_k]

    ms = (time.time() - t0) * 1000
    print(f"[SEARCH] '{query[:60]}' → {len(results)} traits in {ms:.0f}ms")
    return results


def search_by_condition(
    crop:        str,
    temperature: float,
    stress_type: str = "heat",
    location:    str = "",
    top_k:       int = 10
) -> List[CropTraitVector]:
    """
    Structured search when parameters are explicitly known.
    Builds an optimised query string from the parameters.
    """
    parts = [crop, stress_type, "stress", f"{temperature}°C", "tolerance traits"]
    if location:
        parts.append(location)
    return search_traits(" ".join(parts), top_k=top_k, crop_filter=crop)


if __name__ == "__main__":
    results = search_traits("wheat heat tolerance at 48°C Jodhpur")
    for r in results[:5]:
        print(f"  [{r.similarity_score:.3f}] {r.trait_name}: {r.value} {r.unit}")
