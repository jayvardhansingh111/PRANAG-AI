# tests/conftest.py
# Shared pytest fixtures used across all test files.

import pytest
from datetime import datetime, timezone

from JAY.shared.models import (
    CropType, StressType, SoilType,
    LocationInfo, EnvironmentalConditions,
    CropTraitVector, ResearchInsight,
    ParsedPrompt, SpecJSON
)


@pytest.fixture
def jodhpur_location():
    return LocationInfo(city="Jodhpur", state="Rajasthan", country="India",
                        latitude=26.28, longitude=73.02, climate_zone="arid", koppen_class="BWh")

@pytest.fixture
def ludhiana_location():
    return LocationInfo(city="Ludhiana", state="Punjab", country="India",
                        latitude=30.90, longitude=75.85, climate_zone="humid subtropical", koppen_class="Cwa")

@pytest.fixture
def heat_stress_conditions():
    return EnvironmentalConditions(
        temperature_max=48.0, temperature_min=32.0, temperature_mean=40.0,
        rainfall_annual=200.0, humidity_mean=30.0, solar_radiation=22.5,
        co2_ppm=420.0, stress_type=StressType.HEAT
    )

@pytest.fixture
def normal_conditions():
    return EnvironmentalConditions(
        temperature_max=30.0, temperature_min=18.0, temperature_mean=24.0,
        rainfall_annual=800.0, humidity_mean=65.0, solar_radiation=18.0,
        co2_ppm=420.0, stress_type=StressType.NONE
    )

@pytest.fixture
def sample_traits():
    return [
        CropTraitVector(trait_id="T0001", trait_name="heat_tolerance_score",
                        value=87.5, unit="score", confidence=0.94,
                        source_dataset="ICAR-2023", similarity_score=0.923),
        CropTraitVector(trait_id="T0002", trait_name="pollen_viability_pct",
                        value=62.3, unit="pct", confidence=0.89,
                        source_dataset="CIMMYT-2022", similarity_score=0.881),
        CropTraitVector(trait_id="T0003", trait_name="grain_filling_rate",
                        value=4.2, unit="mg/hr", confidence=0.82,
                        source_dataset="NRCPB-2023", similarity_score=0.845),
    ]

@pytest.fixture
def sample_research():
    return [
        ResearchInsight(paper_id="SS-2023-001",
                        title="Heat Stress in Wheat: Molecular Mechanisms",
                        year=2023, journal="Plant Cell & Environment",
                        key_finding="HSP70 maintained 85% pollen viability at 45°C.",
                        relevance=0.92, doi="10.1111/pce.14567"),
        ResearchInsight(paper_id="AX-2024-002",
                        title="Grain Filling Under Terminal Heat Stress",
                        year=2024, journal=None,
                        key_finding="Temperature above 35°C reduces starch accumulation by 40%.",
                        relevance=0.87, doi=None),
    ]

@pytest.fixture
def parsed_wheat_jodhpur():
    return ParsedPrompt(crop_type=CropType.WHEAT, location_raw="Jodhpur",
                        temp_celsius=48.0, stress_hints=["heat"], variety_hint=None)

@pytest.fixture
def parsed_rice_punjab():
    return ParsedPrompt(crop_type=CropType.RICE, location_raw="Ludhiana",
                        temp_celsius=None, stress_hints=["drought"],
                        variety_hint="Pusa Basmati 1")

@pytest.fixture
def valid_spec_dict():
    return {
        "spec_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_id":  "test-pipeline-abc123",
        "original_prompt": "wheat for Jodhpur at 48°C",
        "crop_type": "wheat", "variety": None,
        "growth_stage": "flowering", "target_yield_t_ha": None,
        "location": {"city": "Jodhpur", "state": "Rajasthan", "country": "India",
                     "latitude": 26.28, "longitude": 73.02,
                     "climate_zone": "arid", "koppen_class": "BWh"},
        "conditions": {"temperature_max": 48.0, "temperature_min": 32.0,
                       "temperature_mean": 40.0, "rainfall_annual": 200.0,
                       "humidity_mean": 30.0, "solar_radiation": 22.5,
                       "co2_ppm": 420.0, "stress_type": "heat"},
        "soil_type": "sandy", "soil_ph": 7.5, "soil_nitrogen_ppm": 120.0,
        "relevant_traits": [], "research_insights": [],
        "simulation_params": {"simulation_type": "crop_stress_model_v2", "duration_days": 120},
        "confidence_score": 0.87, "validation_passed": False, "warnings": []
    }
