# orchestrator/spec_builder.py
# Combines parsed prompt + traits + research into one validated SpecJSON.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from JAY.shared.models import (
    ParsedPrompt, SpecJSON, CropTraitVector, ResearchInsight,
    LocationInfo, EnvironmentalConditions, CropType, StressType, SoilType
)


# ── City Database ─────────────────────────────────────────────────────────────

CITY_DATABASE = {
    "jodhpur":    {"state": "Rajasthan",    "lat": 26.28, "lon": 73.02, "climate": "arid",              "koppen": "BWh"},
    "jaipur":     {"state": "Rajasthan",    "lat": 26.91, "lon": 75.79, "climate": "semi-arid",          "koppen": "BSh"},
    "bikaner":    {"state": "Rajasthan",    "lat": 28.02, "lon": 73.31, "climate": "arid",              "koppen": "BWh"},
    "ludhiana":   {"state": "Punjab",       "lat": 30.90, "lon": 75.85, "climate": "humid subtropical", "koppen": "Cwa"},
    "amritsar":   {"state": "Punjab",       "lat": 31.63, "lon": 74.87, "climate": "humid subtropical", "koppen": "Cwa"},
    "nagpur":     {"state": "Maharashtra",  "lat": 21.14, "lon": 79.08, "climate": "tropical",          "koppen": "Aw"},
    "hyderabad":  {"state": "Telangana",    "lat": 17.38, "lon": 78.48, "climate": "tropical",          "koppen": "BSh"},
    "bhopal":     {"state": "M.P.",         "lat": 23.25, "lon": 77.40, "climate": "subtropical",       "koppen": "Cwa"},
    "indore":     {"state": "M.P.",         "lat": 22.72, "lon": 75.86, "climate": "subtropical",       "koppen": "Cwa"},
    "coimbatore": {"state": "Tamil Nadu",   "lat": 11.01, "lon": 76.96, "climate": "tropical",          "koppen": "As"},
    "delhi":      {"state": "Delhi",        "lat": 28.66, "lon": 77.22, "climate": "semi-arid",          "koppen": "BSh"},
    "pune":       {"state": "Maharashtra",  "lat": 18.52, "lon": 73.86, "climate": "tropical",          "koppen": "Aw"},
}

REGION_SOIL = {
    "rajasthan":   SoilType.SANDY,
    "punjab":      SoilType.LOAMY,
    "haryana":     SoilType.LOAMY,
    "maharashtra": SoilType.CLAY,
    "tamil_nadu":  SoilType.CLAY,
    "m.p.":        SoilType.LOAMY,
}

CLIMATE_TEMP_MAX = {
    "arid": 44.0, "semi-arid": 40.0, "tropical": 38.0,
    "humid subtropical": 36.0, "subtropical": 37.0, "temperate": 30.0,
}
CLIMATE_RAINFALL = {
    "arid": 200.0, "semi-arid": 500.0, "tropical": 1200.0,
    "humid subtropical": 800.0, "subtropical": 700.0,
}
CLIMATE_HUMIDITY = {
    "arid": 30.0, "semi-arid": 45.0, "tropical": 75.0,
    "humid subtropical": 65.0, "subtropical": 55.0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def enrich_location(location_raw: str) -> LocationInfo:
    """Map a city name to full LocationInfo using the built-in database."""
    data = CITY_DATABASE.get(location_raw.lower().strip(), {})
    return LocationInfo(
        city         = location_raw.title(),
        state        = data.get("state"),
        country      = "India",
        latitude     = data.get("lat"),
        longitude    = data.get("lon"),
        climate_zone = data.get("climate"),
        koppen_class = data.get("koppen")
    )


def build_conditions(parsed: ParsedPrompt, location: LocationInfo) -> EnvironmentalConditions:
    """Derive EnvironmentalConditions from parsed prompt + location defaults."""
    temp_max  = parsed.temp_celsius or CLIMATE_TEMP_MAX.get(location.climate_zone or "", 38.0)
    temp_mean = round(temp_max - 8.0, 1)
    temp_min  = round(temp_max - 16.0, 1)

    hints = set(parsed.stress_hints)
    if "heat" in hints and "drought" in hints:
        stress = StressType.COMBINED
    elif "heat" in hints or (temp_max and temp_max > 38):
        stress = StressType.HEAT
    elif "drought" in hints or "dry" in hints:
        stress = StressType.DROUGHT
    elif "flood" in hints:
        stress = StressType.FLOOD
    else:
        stress = StressType.NONE

    zone = location.climate_zone or ""
    return EnvironmentalConditions(
        temperature_max  = temp_max,
        temperature_min  = temp_min,
        temperature_mean = temp_mean,
        rainfall_annual  = CLIMATE_RAINFALL.get(zone, 600.0),
        humidity_mean    = CLIMATE_HUMIDITY.get(zone, 50.0),
        solar_radiation  = 22.5,
        co2_ppm          = 420.0,
        stress_type      = stress
    )


def calculate_confidence(
    parsed:   ParsedPrompt,
    traits:   List[CropTraitVector],
    research: List[ResearchInsight],
    location: LocationInfo
) -> float:
    """Score [0–1] based on data completeness."""
    score = 0.0
    if parsed.crop_type != CropType.UNKNOWN:        score += 0.20
    if location.latitude is not None:               score += 0.20
    if len(traits) >= 3:                            score += 0.30
    elif len(traits) >= 1:                          score += 0.15
    if len(research) >= 2:                          score += 0.20
    elif len(research) >= 1:                        score += 0.10
    if parsed.temp_celsius is not None:             score += 0.10
    return round(min(score, 1.0), 3)


def build_simulation_params(
    parsed:     ParsedPrompt,
    conditions: EnvironmentalConditions
) -> Dict[str, Any]:
    """Build parameters for the downstream simulation system."""
    return {
        "simulation_type":     "crop_stress_model_v2",
        "duration_days":       120,
        "timestep_hours":      1,
        "stress_scenario":     conditions.stress_type,
        "temperature_profile": {
            "max":          conditions.temperature_max,
            "min":          conditions.temperature_min,
            "diurnal_range": conditions.temperature_max - conditions.temperature_min
        },
        "water_balance":   True,
        "phenology_model": "DSSAT",
        "output_variables": [
            "grain_yield", "biomass", "leaf_area_index",
            "transpiration", "stress_index"
        ]
    }


# ── Main Entry Point ──────────────────────────────────────────────────────────

def build_spec(
    parsed_prompt:   ParsedPrompt,
    traits:          List[CropTraitVector],
    research:        List[ResearchInsight],
    original_prompt: str,
    pipeline_id:     str
) -> SpecJSON:
    """
    Combine all data sources into one validated SpecJSON object.
    This is the final step before output_validator.validate_spec().
    """
    location   = enrich_location(parsed_prompt.location_raw)
    conditions = build_conditions(parsed_prompt, location)

    state_key  = (location.state or "").lower().replace(" ", "_")
    soil_type  = REGION_SOIL.get(state_key, SoilType.LOAMY)

    top_traits   = sorted(traits,   key=lambda t: t.similarity_score, reverse=True)[:5]
    top_research = sorted(research, key=lambda r: r.relevance,        reverse=True)[:3]

    confidence = calculate_confidence(parsed_prompt, top_traits, top_research, location)

    warnings = []
    if conditions.temperature_max > 45:
        warnings.append(f"Extreme heat ({conditions.temperature_max}°C) exceeds most crop thresholds.")
    if parsed_prompt.crop_type == CropType.UNKNOWN:
        warnings.append("Crop type could not be identified from the prompt.")
    if not top_traits:
        warnings.append("No matching traits found in the vector database.")
    if confidence < 0.5:
        warnings.append("Low confidence score — recommend manual review before simulation.")

    return SpecJSON(
        spec_version      = "1.0.0",
        generated_at      = datetime.now(timezone.utc).isoformat(),
        pipeline_id       = pipeline_id,
        original_prompt   = original_prompt,
        crop_type         = parsed_prompt.crop_type,
        variety           = parsed_prompt.variety_hint,
        growth_stage      = None,
        target_yield_t_ha = None,
        location          = location,
        conditions        = conditions,
        soil_type         = soil_type,
        soil_ph           = 7.5 if soil_type == SoilType.SANDY else 6.8,
        soil_nitrogen_ppm = 120.0,
        relevant_traits   = top_traits,
        research_insights = top_research,
        simulation_params = build_simulation_params(parsed_prompt, conditions),
        confidence_score  = confidence,
        validation_passed = False,
        warnings          = warnings
    )
