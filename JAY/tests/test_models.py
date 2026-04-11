# tests/test_models.py
# Tests for shared/models.py — all Pydantic schemas.

import pytest
from pydantic import ValidationError
from shared.models import (
    CropType, StressType, SoilType,
    LocationInfo, EnvironmentalConditions,
    CropTraitVector, ResearchInsight, ParsedPrompt, SpecJSON
)


class TestCropTypeEnum:
    def test_all_valid_values(self):
        for v in ["wheat","rice","maize","sorghum","barley","cotton","soybean","sugarcane","unknown"]:
            assert CropType(v).value == v

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError): CropType("banana")

    def test_case_sensitive(self):
        with pytest.raises(ValueError): CropType("Wheat")

    def test_is_string_subclass(self):
        assert CropType.WHEAT == "wheat"


class TestStressTypeEnum:
    def test_all_valid(self):
        for v in ["heat","drought","flood","salinity","cold","combined","none"]:
            assert StressType(v).value == v

    def test_invalid_raises(self):
        with pytest.raises(ValueError): StressType("tornado")


class TestLocationInfo:
    def test_only_city_required(self):
        loc = LocationInfo(city="Jodhpur")
        assert loc.country == "India"
        assert loc.latitude is None

    def test_full_location(self, jodhpur_location):
        assert jodhpur_location.latitude == 26.28
        assert jodhpur_location.koppen_class == "BWh"

    def test_city_required(self):
        with pytest.raises(ValidationError) as exc:
            LocationInfo()
        assert "city" in str(exc.value)

    def test_custom_country(self):
        assert LocationInfo(city="Paris", country="France").country == "France"


class TestEnvironmentalConditions:
    def test_valid(self, heat_stress_conditions):
        assert heat_stress_conditions.temperature_max == 48.0

    def test_temp_max_above_60_raises(self):
        with pytest.raises(ValidationError):
            EnvironmentalConditions(temperature_max=61.0, temperature_min=20.0, temperature_mean=40.0)

    def test_temp_min_below_minus20_raises(self):
        with pytest.raises(ValidationError):
            EnvironmentalConditions(temperature_max=30.0, temperature_min=-21.0, temperature_mean=10.0)

    def test_mean_above_max_raises(self):
        with pytest.raises(ValidationError) as exc:
            EnvironmentalConditions(temperature_max=40.0, temperature_min=20.0, temperature_mean=45.0)
        assert "between min and max" in str(exc.value)

    def test_mean_below_min_raises(self):
        with pytest.raises(ValidationError):
            EnvironmentalConditions(temperature_max=40.0, temperature_min=20.0, temperature_mean=15.0)

    def test_mean_equals_min_valid(self):
        c = EnvironmentalConditions(temperature_max=40.0, temperature_min=20.0, temperature_mean=20.0)
        assert c.temperature_mean == 20.0

    def test_mean_equals_max_valid(self):
        c = EnvironmentalConditions(temperature_max=40.0, temperature_min=20.0, temperature_mean=40.0)
        assert c.temperature_mean == 40.0

    def test_humidity_above_100_raises(self):
        with pytest.raises(ValidationError):
            EnvironmentalConditions(temperature_max=30.0, temperature_min=20.0,
                                    temperature_mean=25.0, humidity_mean=101.0)

    def test_humidity_negative_raises(self):
        with pytest.raises(ValidationError):
            EnvironmentalConditions(temperature_max=30.0, temperature_min=20.0,
                                    temperature_mean=25.0, humidity_mean=-1.0)

    def test_rainfall_negative_raises(self):
        with pytest.raises(ValidationError):
            EnvironmentalConditions(temperature_max=30.0, temperature_min=20.0,
                                    temperature_mean=25.0, rainfall_annual=-50.0)

    def test_default_co2(self):
        c = EnvironmentalConditions(temperature_max=30.0, temperature_min=20.0, temperature_mean=25.0)
        assert c.co2_ppm == 420.0

    def test_default_stress_is_none(self):
        c = EnvironmentalConditions(temperature_max=30.0, temperature_min=20.0, temperature_mean=25.0)
        assert c.stress_type == StressType.NONE


class TestCropTraitVector:
    def test_valid(self, sample_traits):
        t = sample_traits[0]
        assert 0.0 <= t.confidence <= 1.0
        assert 0.0 <= t.similarity_score <= 1.0

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValidationError):
            CropTraitVector(trait_id="T1", trait_name="x", value=1.0, unit="",
                            confidence=1.5, source_dataset="x", similarity_score=0.9)

    def test_similarity_negative_raises(self):
        with pytest.raises(ValidationError):
            CropTraitVector(trait_id="T1", trait_name="x", value=1.0, unit="",
                            confidence=0.9, source_dataset="x", similarity_score=-0.1)

    def test_zero_scores_valid(self):
        t = CropTraitVector(trait_id="T1", trait_name="x", value=0.0, unit="",
                            confidence=0.0, source_dataset="x", similarity_score=0.0)
        assert t.similarity_score == 0.0


class TestResearchInsight:
    def test_valid(self, sample_research):
        assert 0.0 <= sample_research[0].relevance <= 1.0

    def test_journal_can_be_none(self):
        r = ResearchInsight(paper_id="X", title="X", year=2024,
                            journal=None, key_finding="F.", relevance=0.8, doi=None)
        assert r.journal is None

    def test_relevance_above_1_raises(self):
        with pytest.raises(ValidationError):
            ResearchInsight(paper_id="X", title="X", year=2024,
                            journal=None, key_finding="F.", relevance=1.5, doi=None)

    def test_relevance_negative_raises(self):
        with pytest.raises(ValidationError):
            ResearchInsight(paper_id="X", title="X", year=2024,
                            journal=None, key_finding="F.", relevance=-0.1, doi=None)


class TestParsedPrompt:
    def test_valid(self, parsed_wheat_jodhpur):
        assert parsed_wheat_jodhpur.crop_type == CropType.WHEAT
        assert parsed_wheat_jodhpur.temp_celsius == 48.0
        assert "heat" in parsed_wheat_jodhpur.stress_hints

    def test_temp_optional(self):
        p = ParsedPrompt(crop_type=CropType.RICE, location_raw="Punjab",
                         temp_celsius=None, stress_hints=[], variety_hint=None)
        assert p.temp_celsius is None

    def test_raw_entities_defaults_empty(self):
        p = ParsedPrompt(crop_type=CropType.WHEAT, location_raw="Delhi",
                         temp_celsius=None, stress_hints=[], variety_hint=None)
        assert p.raw_entities == {}

    def test_invalid_crop_raises(self):
        with pytest.raises(ValidationError):
            ParsedPrompt(crop_type="tomato", location_raw="Delhi",
                         temp_celsius=None, stress_hints=[], variety_hint=None)


class TestSpecJSON:
    def test_valid_full_spec(self, valid_spec_dict):
        spec = SpecJSON.model_validate(valid_spec_dict)
        assert spec.confidence_score == 0.87

    def test_missing_pipeline_id_raises(self, valid_spec_dict):
        del valid_spec_dict["pipeline_id"]
        with pytest.raises(ValidationError) as exc:
            SpecJSON.model_validate(valid_spec_dict)
        assert "pipeline_id" in str(exc.value)

    def test_confidence_above_1_raises(self, valid_spec_dict):
        valid_spec_dict["confidence_score"] = 1.5
        with pytest.raises(ValidationError): SpecJSON.model_validate(valid_spec_dict)

    def test_confidence_negative_raises(self, valid_spec_dict):
        valid_spec_dict["confidence_score"] = -0.1
        with pytest.raises(ValidationError): SpecJSON.model_validate(valid_spec_dict)

    def test_soil_ph_low_raises(self, valid_spec_dict):
        valid_spec_dict["soil_ph"] = 2.9
        with pytest.raises(ValidationError): SpecJSON.model_validate(valid_spec_dict)

    def test_soil_ph_high_raises(self, valid_spec_dict):
        valid_spec_dict["soil_ph"] = 10.1
        with pytest.raises(ValidationError): SpecJSON.model_validate(valid_spec_dict)

    def test_soil_ph_valid_range(self, valid_spec_dict):
        for ph in [3.0, 6.5, 7.0, 10.0]:
            valid_spec_dict["soil_ph"] = ph
            assert SpecJSON.model_validate(valid_spec_dict).soil_ph == ph

    def test_traits_default_empty(self, valid_spec_dict):
        valid_spec_dict.pop("relevant_traits", None)
        assert SpecJSON.model_validate(valid_spec_dict).relevant_traits == []

    def test_validation_passed_defaults_false(self, valid_spec_dict):
        valid_spec_dict.pop("validation_passed", None)
        assert SpecJSON.model_validate(valid_spec_dict).validation_passed == False

    def test_round_trip_serialization(self, valid_spec_dict):
        import json
        spec1 = SpecJSON.model_validate(valid_spec_dict)
        spec2 = SpecJSON.model_validate(json.loads(spec1.model_dump_json()))
        assert spec1.pipeline_id == spec2.pipeline_id

    def test_nested_conditions_validated(self, valid_spec_dict):
        valid_spec_dict["conditions"]["temperature_mean"] = 999.0
        with pytest.raises(ValidationError): SpecJSON.model_validate(valid_spec_dict)

    def test_invalid_crop_type_raises(self, valid_spec_dict):
        valid_spec_dict["crop_type"] = "mango"
        with pytest.raises(ValidationError): SpecJSON.model_validate(valid_spec_dict)

    def test_invalid_soil_type_raises(self, valid_spec_dict):
        valid_spec_dict["soil_type"] = "concrete"
        with pytest.raises(ValidationError): SpecJSON.model_validate(valid_spec_dict)

    def test_spec_with_traits_and_research(self, valid_spec_dict, sample_traits, sample_research):
        valid_spec_dict["relevant_traits"]   = [t.model_dump() for t in sample_traits]
        valid_spec_dict["research_insights"] = [r.model_dump() for r in sample_research]
        spec = SpecJSON.model_validate(valid_spec_dict)
        assert len(spec.relevant_traits)   == 3
        assert len(spec.research_insights) == 2
