# tests/test_embeddings.py
# Tests for search_engine/embeddings.py

import numpy as np
import pytest
from search_engine.embeddings import (
    build_trait_text, embed_traits, embed_query, generate_sample_traits
)


class TestBuildTraitText:
    def test_full_dict(self):
        trait = {"crop": "wheat", "trait_name": "heat_tolerance_score", "value": 87.5,
                 "unit": "score", "conditions": {"temperature": 45.0, "growth_stage": "flowering",
                 "stress_type": "heat"}}
        text = build_trait_text(trait)
        assert "wheat" in text
        assert "87.5" in text
        assert "45" in text
        assert "flowering" in text

    def test_minimal_dict(self):
        text = build_trait_text({"trait_name": "yield"})
        assert "yield" in text
        assert len(text) > 0

    def test_empty_dict(self):
        assert isinstance(build_trait_text({}), str)

    def test_underscores_converted(self):
        text = build_trait_text({"trait_name": "heat_tolerance_score"})
        assert "heat tolerance score" in text

    def test_temperature_has_unit(self):
        text = build_trait_text({"trait_name": "t", "conditions": {"temperature": 48.0}})
        assert "48" in text and "°C" in text

    def test_description_capped(self):
        trait = {"trait_name": "t", "value": 1.0, "unit": "", "description": "X" * 200}
        assert len(build_trait_text(trait)) < 300

    def test_different_crops_different_texts(self):
        t1 = {"crop": "wheat", "trait_name": "test", "value": 1.0, "unit": ""}
        t2 = {"crop": "rice",  "trait_name": "test", "value": 1.0, "unit": ""}
        assert build_trait_text(t1) != build_trait_text(t2)

    def test_no_conditions_no_error(self):
        text = build_trait_text({"crop": "rice", "trait_name": "yield_kg_ha", "value": 5000.0, "unit": "kg/ha"})
        assert "rice" in text


class TestEmbedTraits:
    def test_output_shape(self):
        traits = generate_sample_traits(10)
        texts, embs = embed_traits(traits, batch_size=5)
        assert embs.shape == (10, 384)

    def test_texts_length_matches(self):
        traits = generate_sample_traits(7)
        texts, embs = embed_traits(traits, batch_size=7)
        assert len(texts) == 7

    def test_embeddings_normalized(self):
        traits = generate_sample_traits(5)
        _, embs = embed_traits(traits, batch_size=5)
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01)

    def test_empty_returns_empty(self):
        texts, embs = embed_traits([])
        assert len(texts) == 0
        assert embs.shape == (0,)

    def test_single_trait(self):
        _, embs = embed_traits(generate_sample_traits(1), batch_size=1)
        assert embs.shape == (1, 384)

    def test_dtype_is_float(self):
        _, embs = embed_traits(generate_sample_traits(3), batch_size=3)
        assert embs.dtype in (np.float32, np.float64)

    def test_different_traits_different_embeddings(self):
        t1 = [{"crop": "wheat", "trait_name": "heat_tolerance", "value": 90.0, "unit": "score"}]
        t2 = [{"crop": "rice",  "trait_name": "flood_tolerance", "value": 75.0, "unit": "score"}]
        _, e1 = embed_traits(t1)
        _, e2 = embed_traits(t2)
        assert float(e1[0] @ e2[0]) < 0.999

    def test_large_batch(self):
        traits = generate_sample_traits(500)
        _, embs = embed_traits(traits, batch_size=128)
        assert embs.shape[0] == 500


class TestEmbedQuery:
    def test_returns_1d(self):
        assert embed_query("wheat heat tolerance").ndim == 1

    def test_dimension_384(self):
        assert embed_query("test query").shape[0] == 384

    def test_normalized(self):
        vec = embed_query("wheat drought stress")
        assert abs(np.linalg.norm(vec) - 1.0) < 0.01

    def test_empty_query_returns_vector(self):
        assert embed_query("").shape[0] == 384

    def test_similar_queries_high_similarity(self):
        v1 = embed_query("wheat high temperature stress")
        v2 = embed_query("wheat heat stress tolerance")
        assert float(v1 @ v2) > 0.7

    def test_can_dot_product_with_traits(self):
        qv = embed_query("wheat heat tolerance")
        traits = generate_sample_traits(5)
        _, embs = embed_traits(traits)
        scores = embs @ qv
        assert scores.shape == (5,)
        assert all(-1.01 <= s <= 1.01 for s in scores)


class TestGenerateSampleTraits:
    def test_correct_count(self):
        assert len(generate_sample_traits(50)) == 50

    def test_required_fields_present(self):
        for trait in generate_sample_traits(10):
            for f in ["trait_id", "crop", "trait_name", "value", "unit"]:
                assert f in trait

    def test_ids_unique(self):
        traits = generate_sample_traits(100)
        ids = [t["trait_id"] for t in traits]
        assert len(ids) == len(set(ids))

    def test_valid_crop_types(self):
        valid = {"wheat","rice","maize","sorghum","barley","cotton","soybean","millet"}
        for t in generate_sample_traits(50):
            assert t["crop"] in valid

    def test_values_numeric(self):
        for t in generate_sample_traits(20):
            assert isinstance(t["value"], (int, float))

    def test_conditions_present(self):
        for t in generate_sample_traits(10):
            assert isinstance(t.get("conditions"), dict)

    def test_temperature_realistic(self):
        for t in generate_sample_traits(100):
            temp = t["conditions"].get("temperature", 0)
            assert 25.0 <= temp <= 55.0

    def test_zero_count_empty(self):
        assert generate_sample_traits(0) == []
