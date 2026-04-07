# shared/models.py
# Pydantic data models shared across the entire pipeline.
# Every field is strictly typed — invalid data raises ValidationError immediately.

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────

class CropType(str, Enum):
    WHEAT     = "wheat"
    RICE      = "rice"
    MAIZE     = "maize"
    SORGHUM   = "sorghum"
    BARLEY    = "barley"
    COTTON    = "cotton"
    SOYBEAN   = "soybean"
    SUGARCANE = "sugarcane"
    UNKNOWN   = "unknown"


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

    @validator("temperature_mean")
    def mean_between_min_max(cls, v, values):
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
    """
    The validated output sent to the simulation system.
    No free text allowed — every field is strictly typed.
    """
    spec_version:      str = "1.0.0"
    generated_at:      str
    pipeline_id:       str
    original_prompt:   str

    crop_type:         CropType
    variety:           Optional[str]
    growth_stage:      Optional[str]
    target_yield_t_ha: Optional[float]

    location:          LocationInfo
    conditions:        EnvironmentalConditions

    soil_type:         Optional[SoilType]
    soil_ph:           Optional[float] = Field(None, ge=3.0, le=10.0)
    soil_nitrogen_ppm: Optional[float]

    relevant_traits:   List[CropTraitVector] = [],
    research_insights: List[ResearchInsight] = [],
    simulation_params: Dict[str, Any] = {},

    confidence_score:  float = Field(..., ge=0.0, le=1.0)
    validation_passed: bool = False
    warnings:          List[str] = [],

    class Config:
        use_enum_values = True


# ── Intermediate Model ────────────────────────────────────────────────────────

class ParsedPrompt(BaseModel):
    """Output of the LLM prompt parser. Intermediate step before full spec."""
    crop_type:    CropType
    location_raw: str
    temp_celsius: Optional[float]
    stress_hints: List[str]
    variety_hint: Optional[str]
    raw_entities: Dict[str, Any] = {},
