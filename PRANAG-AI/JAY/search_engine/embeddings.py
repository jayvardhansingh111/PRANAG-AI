# search_engine/embeddings.py
# Converts crop trait records into 384-dimensional semantic vectors.
# Uses sentence-transformers (all-MiniLM-L6-v2) — fast, offline, no API key.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from JAY.shared.config import EMBEDDING_MODEL

_model: Optional["SentenceTransformer"] = None


def get_model() -> "SentenceTransformer":
    """Lazy-load the embedding model (singleton). Downloads once, cached after."""
    global _model
    if _model is None:
        if not ST_AVAILABLE:
            raise ImportError("Run: pip install sentence-transformers")
        print(f"[EMBED] Loading: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def build_trait_text(trait: Dict[str, Any]) -> str:
    """
    Convert a trait dict into a natural language string for embedding.
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
    traits:     List[Dict[str, Any]],
    batch_size: int = 256
) -> Tuple[List[str], np.ndarray]:
    """
    Embed a list of trait dicts. Returns (text_list, embedding_matrix shape N×384).
    Uses L2 normalization so dot-product = cosine similarity.
    """
    if not traits:
        return [], np.array([])

    texts = [build_trait_text(t) for t in traits]

    if ST_AVAILABLE:
        model = get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 1000,
            normalize_embeddings=True
        )
        return texts, embeddings

    # Mock embeddings for testing without the model installed
    dim = 384
    emb = np.random.randn(len(texts), dim).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return texts, emb


def embed_query(query: str) -> np.ndarray:
    """Embed a single search query string. Returns 1D array of length 384."""
    if ST_AVAILABLE:
        return get_model().encode([query], normalize_embeddings=True)[0]
    dim = 384
    vec = np.random.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def generate_sample_traits(n: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic crop trait data for testing without real Parquet files."""
    import random
    crops   = ["wheat", "rice", "maize", "sorghum", "barley", "cotton", "soybean", "millet"]
    names   = ["heat_tolerance_score", "drought_resistance_index", "yield_kg_ha",
               "pollen_viability_pct", "germination_rate_pct", "grain_filling_rate",
               "root_depth_cm", "leaf_area_index", "stomatal_conductance", "chlorophyll_content"]
    stages  = ["germination", "vegetative", "flowering", "grain_fill", "maturity"]
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
            "trait_id":   f"T{i:07d}",
            "crop":       random.choice(crops),
            "trait_name": trait_name,
            "value":      round(random.uniform(lo, hi), 2),
            "unit":       "score" if "score" in trait_name or "index" in trait_name
                          else "pct" if "pct" in trait_name
                          else "kg/ha" if "kg_ha" in trait_name else "value",
            "conditions": {
                "temperature":  round(random.uniform(25, 55), 1),
                "growth_stage": random.choice(stages),
                "stress_type":  random.choice(stresses)
            },
            "source_dataset": f"dataset_{random.randint(1, 10)}"
        })
    return samples
