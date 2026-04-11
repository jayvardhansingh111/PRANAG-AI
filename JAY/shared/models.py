# shared/models.py
# Pydantic data models shared across the entire pipeline.
# Every field is strictly typed — invalid data raises ValidationError immediately.

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────

class CropType(str, Enum):
    WHEAT      = "wheat"
    RICE       = "rice"
    MAIZE      = "maize"
    SORGHUM    = "sorghum"
    BARLEY     = "barley"
    COTTON     = "cotton"
    SOYBEAN    = "soybean"
    SUGARCANE  = "sugarcane"
    POTATO     = "potato"
    TOMATO     = "tomato"
    ONION      = "onion"
    GARLIC     = "garlic"
    PEANUT     = "peanut"
    CHICKPEA   = "chickpea"
    LENTIL     = "lentil"
    MUSTARD    = "mustard"
    SUNFLOWER  = "sunflower"
    SESAME     = "sesame"
    MILLET     = "millet"
    BANANA     = "banana"
    APPLE      = "apple"
    UNKNOWN    = "unknown"


class StressType(str, Enum):
    HEAT     = "heat"
    DROUGHT  = "drought"
    FLOOD    = "flood"
    SALINITY = "salinity"
    COLD     = "cold"
    COMBINED = "combined"
    NONE     = "none"


class SoilType(str, Enum):
    LOAMY  = "loamy"
    SANDY  = "sandy"
    CLAY   = "clay"
    SILTY  = "silty"
    PEATY  = "peaty"
    CHALKY = "chalky"


# ── Sub-models ────────────────────────────────────────────────────────────────

class LocationInfo(BaseModel):
    city:         str
    state:        Optional[str] = None
    country:      str = "India"
    latitude:     Optional[float] = None
    longitude:    Optional[float] = None
    climate_zone: Optional[str] = None
    koppen_class: Optional[str] = None


class EnvironmentalConditions(BaseModel):
    temperature_max:  float = Field(..., ge=-10, le=60)
    temperature_min:  float = Field(..., ge=-20, le=50)
    temperature_mean: float = Field(..., ge=-15, le=55)
    rainfall_annual:  Optional[float] = Field(None, ge=0)
    humidity_mean:    Optional[float] = Field(None, ge=0, le=100)
    solar_radiation:  Optional[float] = Field(None, ge=0)
    co2_ppm:          float = Field(default=420.0)
    stress_type:      StressType = StressType.NONE

    @field_validator("temperature_mean")
    @classmethod
    def mean_between_min_max(cls, v, info):
        values = info.data
        if "temperature_min" in values and "temperature_max" in values:
            if not (values["temperature_min"] <= v <= values["temperature_max"]):
                raise ValueError("temperature_mean must be between min and max")
        return v


class CropTraitVector(BaseModel):
    trait_id:         str
    trait_name:       str
    value:            float
    unit:             str
    confidence:       float = Field(..., ge=0.0, le=1.0)
    source_dataset:   str
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class ResearchInsight(BaseModel):
    paper_id:    str
    title:       str
    year:        int
    journal:     Optional[str]
    key_finding: str
    relevance:   float = Field(..., ge=0.0, le=1.0)
    doi:         Optional[str]


# ── Final Output Model ────────────────────────────────────────────────────────

class SpecJSON(BaseModel):
    """The validated output sent to the simulation system."""

    model_config = ConfigDict(use_enum_values=True)

    spec_version:      str = "1.0.0"
    generated_at:      str
    pipeline_id:       str
    original_prompt:   str

    crop_type:         CropType
    variety:           Optional[str] = None
    growth_stage:      Optional[str] = None
    target_yield_t_ha: Optional[float] = None

    location:          LocationInfo
    conditions:        EnvironmentalConditions

    soil_type:         Optional[SoilType] = None
    soil_ph:           Optional[float] = Field(None, ge=3.0, le=10.0)
    soil_nitrogen_ppm: Optional[float] = None

    relevant_traits:   List[CropTraitVector] = Field(default_factory=list)
    research_insights: List[ResearchInsight] = Field(default_factory=list)
    simulation_params: Dict[str, Any] = Field(default_factory=dict)

    confidence_score:  float = Field(..., ge=0.0, le=1.0)
    validation_passed: bool = False
    warnings:          List[str] = Field(default_factory=list)


# ── Intermediate Model ────────────────────────────────────────────────────────

class ParsedPrompt(BaseModel):
    """Output of the LLM prompt parser. Intermediate step before full spec."""
    crop_type:    CropType
    location_raw: str
    temp_celsius: Optional[float] = None
    stress_hints: List[str] = Field(default_factory=list)
    variety_hint: Optional[str] = None
    raw_entities: Dict[str, Any] = Field(default_factory=dict)