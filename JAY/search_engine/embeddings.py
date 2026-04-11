# search_engine/embeddings.py
# Converts crop trait records into 384-dimensional semantic vectors.
# Uses sentence-transformers (all-MiniLM-L6-v2) — fast, offline, no API key.

import logging
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from JAY.shared.config import EMBEDDING_MODEL, get_settings

logger = logging.getLogger(__name__)
_model: Optional["SentenceTransformer"] = None


def get_model() -> "SentenceTransformer":
    """Lazy-load the embedding model (singleton). Downloads once, cached after."""
    global _model
    if _model is None:
        if not ST_AVAILABLE:
            raise ImportError("Run: pip install sentence-transformers")
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _hash_based_embedding(text: str, dim: int = 384, seed: int = 42) -> np.ndarray:
    """
    Generate deterministic embedding using hash-based approach.
    Preserves similarity relationships for testing without real sentence-transformers.
    """
    h = hashlib.sha256((text + str(seed)).encode())
    hash_int = int(h.hexdigest(), 16)
    rng = np.random.RandomState(hash_int % (2**31))
    vec = rng.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec


def build_trait_text(trait: Dict[str, Any]) -> str:
    """
    Convert a trait dict into natural language string for embedding.
    Natural language gives better semantic search than raw JSON.
    """
    parts = []
    if crop := trait.get("crop"):
        parts.append(crop)
    if name := trait.get("trait_name", ""):
        parts.append(name.replace("_", " "))
    if value := trait.get("value"):
        unit = trait.get("unit", "")
        parts.append(f"{value} {unit}".strip())
    cond = trait.get("conditions", {})
    if temp := cond.get("temperature"):
        parts.append(f"at {temp}°C")
    if stage := cond.get("growth_stage"):
        parts.append(f"during {stage}")
    if stress := cond.get("stress_type"):
        parts.append(f"under {stress} stress")
    if desc := trait.get("description"):
        parts.append(desc[:100])
    return " ".join(parts)


def embed_traits(
    traits: List[Dict[str, Any]],
    batch_size: int = 256
) -> Tuple[List[str], np.ndarray]:
    """
    Embed a list of trait dicts.
    Returns (text_list, embedding_matrix shape N×384).
    Uses L2 normalization so dot-product = cosine similarity.
    """
    if not traits:
        return [], np.array([])

    texts = [build_trait_text(t) for t in traits]

    if ST_AVAILABLE:
        try:
            model = get_model()
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 1000,
                normalize_embeddings=True
            )
            return texts, embeddings
        except Exception as e:
            logger.warning(f"Embedding failed, using deterministic fallback: {e}")

    # Deterministic fallback embeddings
    settings = get_settings()
    if settings.app_env == "prod":
        raise RuntimeError(
            "Sentence-transformers unavailable in production. "
            "Install: pip install sentence-transformers"
        )

    logger.info(f"Using deterministic hash-based embeddings for {len(texts)} traits")
    embeddings = np.array([_hash_based_embedding(t) for t in texts], dtype=np.float32)
    return texts, embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single search query string. Returns 1D array of length 384."""
    if ST_AVAILABLE:
        try:
            return get_model().encode([query], normalize_embeddings=True)[0]
        except Exception as e:
            logger.warning(f"Query embedding failed, using fallback: {e}")

    settings = get_settings()
    if settings.app_env == "prod":
        raise RuntimeError("Sentence-transformers unavailable in production")

    return _hash_based_embedding(query)


def generate_sample_traits(n: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic crop trait data for testing without real Parquet files."""
    import random
    random.seed(42)  # Deterministic for reproducibility

    crops = ["wheat", "rice", "maize", "sorghum", "barley", "cotton", "soybean", "millet"]
    names = [
        "heat_tolerance_score", "drought_resistance_index", "yield_kg_ha",
        "pollen_viability_pct", "germination_rate_pct", "grain_filling_rate",
        "root_depth_cm", "leaf_area_index", "stomatal_conductance", "chlorophyll_content"
    ]
    stages = ["germination", "vegetative", "flowering", "grain_fill", "maturity"]
    stresses = ["heat", "drought", "flood", "salinity", "none"]

    samples = []
    for i in range(n):
        trait_name = random.choice(names)
        ranges = {
            "heat_tolerance_score": (40, 100), "drought_resistance_index": (0.3, 1.0),
            "yield_kg_ha": (1000, 8000), "pollen_viability_pct": (20, 100),
            "germination_rate_pct": (50, 98),
        }
        lo, hi = ranges.get(trait_name, (0, 100))
        samples.append({
            "trait_id": f"T{i:07d}",
            "crop": random.choice(crops),
            "trait_name": trait_name,
            "value": round(random.uniform(lo, hi), 2),
            "unit": (
                "score" if "score" in trait_name or "index" in trait_name
                else "pct" if "pct" in trait_name
                else "kg/ha" if "kg_ha" in trait_name
                else "value"
            ),
            "conditions": {
                "temperature": round(random.uniform(25, 55), 1),
                "growth_stage": random.choice(stages),
                "stress_type": random.choice(stresses)
            },
            "source_dataset": f"dataset_{random.randint(1, 10)}"
        })
    return samples
