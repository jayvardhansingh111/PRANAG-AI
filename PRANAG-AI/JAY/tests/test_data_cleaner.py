# tests/test_data_cleaner.py
# Tests for search_engine/data_cleaner.py

import pytest
from JAY.search_engine.data_cleaner import (
    validate_schema, handle_missing_values, normalize_temperature_column,
    remove_outliers, deduplicate, normalize_strings, clean_from_dict_list,
    INVALID_VALUES, REQUIRED_COLUMNS
)


def make_row(overrides=None):
    base = {"trait_id": "T001", "crop": "wheat", "trait_name": "heat_tolerance_score",
            "value": 85.0, "unit": "score", "temperature": 45.0,
            "growth_stage": "flowering", "stress_type": "heat", "source": "ICAR-2023"}
    if overrides:
        base.update(overrides)
    return base


def make_df(rows):
    import pandas as pd
    return pd.DataFrame(rows)


class TestValidateSchema:
    def test_valid_passes(self):
        df, warnings = validate_schema(make_df([make_row()]))
        assert len(df) == 1

    def test_missing_trait_id_raises(self):
        row = make_row(); del row["trait_id"]
        with pytest.raises(ValueError) as exc:
            validate_schema(make_df([row]))
        assert "trait_id" in str(exc.value)

    def test_missing_crop_raises(self):
        row = make_row(); del row["crop"]
        with pytest.raises(ValueError): validate_schema(make_df([row]))

    def test_missing_value_raises(self):
        row = make_row(); del row["value"]
        with pytest.raises(ValueError): validate_schema(make_df([row]))

    def test_optional_column_added_if_missing(self):
        row = make_row(); del row["temperature"]
        df, warnings = validate_schema(make_df([row]))
        assert "temperature" in df.columns

    def test_warning_for_missing_optional(self):
        row = make_row(); del row["temperature"]
        _, warnings = validate_schema(make_df([row]))
        assert any("temperature" in w for w in warnings)

    def test_multiple_missing_required_reported(self):
        with pytest.raises(ValueError) as exc:
            validate_schema(make_df([{"trait_id": "T001"}]))
        assert "Missing required columns" in str(exc.value)


class TestHandleMissingValues:
    def test_drops_row_with_null_trait_id(self):
        import pandas as pd
        rows = [make_row(), make_row({"trait_id": None})]
        df, stats = handle_missing_values(make_df(rows))
        assert len(df) == 1
        assert stats.get("dropped_missing_trait_id", 0) == 1

    def test_drops_row_with_null_crop(self):
        rows = [make_row(), make_row({"crop": None})]
        df, _ = handle_missing_values(make_df(rows))
        assert len(df) == 1

    def test_fills_temperature_with_median(self):
        rows = [make_row({"temperature": 40.0}),
                make_row({"trait_id": "T002", "temperature": 50.0}),
                make_row({"trait_id": "T003", "temperature": None})]
        df, stats = handle_missing_values(make_df(rows))
        val = df[df["trait_id"] == "T003"]["temperature"].iloc[0]
        assert val == 45.0
        assert stats.get("filled_temperature_with_median", 0) == 1

    def test_fills_stress_type_with_unknown(self):
        rows = [make_row({"stress_type": None})]
        df, _ = handle_missing_values(make_df(rows))
        assert df["stress_type"].iloc[0] == "unknown"

    def test_fills_growth_stage_with_unknown(self):
        rows = [make_row({"growth_stage": None})]
        df, _ = handle_missing_values(make_df(rows))
        assert df["growth_stage"].iloc[0] == "unknown"

    def test_no_changes_when_clean(self):
        rows = [make_row(), make_row({"trait_id": "T002"})]
        df, stats = handle_missing_values(make_df(rows))
        assert len(df) == 2
        assert stats.get("total_dropped", 0) == 0


class TestNormalizeTemperatureColumn:
    def test_celsius_unchanged(self):
        df = make_df([make_row({"temperature": 45.0})])
        result = normalize_temperature_column(df)
        assert abs(result["temperature"].iloc[0] - 45.0) < 0.1

    def test_kelvin_converted(self):
        df = make_df([make_row({"temperature": 318.15})])
        result = normalize_temperature_column(df)
        assert abs(result["temperature"].iloc[0] - 45.0) < 0.5

    def test_explicit_unit_column(self):
        import pandas as pd
        row = make_row({"temperature": 113.0})
        row["temperature_unit"] = "f"
        result = normalize_temperature_column(pd.DataFrame([row]))
        assert abs(result["temperature"].iloc[0] - 45.0) < 0.5

    def test_no_temperature_column_no_error(self):
        import pandas as pd
        df = pd.DataFrame([{"trait_id": "T001", "crop": "wheat"}])
        normalize_temperature_column(df)  # must not raise


class TestRemoveOutliers:
    def test_removes_placeholder_999(self):
        rows = [make_row(), make_row({"trait_id": "T002", "value": 999})]
        df, n = remove_outliers(make_df(rows))
        assert n >= 1

    def test_removes_negative_placeholder(self):
        rows = [make_row(), make_row({"trait_id": "T002", "value": -999})]
        _, n = remove_outliers(make_df(rows))
        assert n >= 1

    def test_temperature_above_65_removed(self):
        rows = [make_row(), make_row({"trait_id": "T002", "temperature": 70.0})]
        _, n = remove_outliers(make_df(rows))
        assert n >= 1

    def test_temperature_below_minus10_removed(self):
        rows = [make_row(), make_row({"trait_id": "T002", "temperature": -15.0})]
        _, n = remove_outliers(make_df(rows))
        assert n >= 1

    def test_valid_rows_retained(self):
        rows = [make_row({"trait_id": f"T{i:03d}"}) for i in range(5)]
        df, n = remove_outliers(make_df(rows))
        assert n == 0 and len(df) == 5

    def test_returns_int_count(self):
        _, n = remove_outliers(make_df([make_row()]))
        assert isinstance(n, int)


class TestDeduplicate:
    def test_removes_exact_duplicate(self):
        rows = [make_row(), make_row()]
        df, n = deduplicate(make_df(rows))
        assert len(df) == 1 and n == 1

    def test_keeps_unique_records(self):
        rows = [make_row({"trait_id": "T001", "value": 80.0}),
                make_row({"trait_id": "T002", "value": 90.0})]
        df, n = deduplicate(make_df(rows))
        assert len(df) == 2 and n == 0

    def test_three_dupes_become_one(self):
        rows = [make_row({"trait_id": f"T{i:03d}"}) for i in range(3)]
        df, n = deduplicate(make_df(rows))
        assert len(df) == 1


class TestNormalizeStrings:
    def test_crop_lowercased(self):
        df = normalize_strings(make_df([make_row({"crop": "Wheat"})]))
        assert df["crop"].iloc[0] == "wheat"

    def test_crop_stripped(self):
        df = normalize_strings(make_df([make_row({"crop": "  wheat  "})]))
        assert df["crop"].iloc[0] == "wheat"

    def test_trait_name_underscored(self):
        df = normalize_strings(make_df([make_row({"trait_name": "heat tolerance score"})]))
        assert df["trait_name"].iloc[0] == "heat_tolerance_score"

    @pytest.mark.parametrize("raw,expected", [
        ("high_temp",         "heat"),
        ("high temperature",  "heat"),
        ("thermal",           "heat"),
        ("water_stress",      "drought"),
        ("water deficit",     "drought"),
        ("dry",               "drought"),
        ("waterlog",          "flood"),
        ("flooding",          "flood"),
        ("salt",              "salinity"),
        ("saline",            "salinity"),
        ("control",           "none"),
        ("normal",            "none"),
    ])
    def test_stress_synonym_normalized(self, raw, expected):
        df = normalize_strings(make_df([make_row({"stress_type": raw})]))
        assert df["stress_type"].iloc[0] == expected


class TestCleanFromDictList:
    def test_valid_data_passes(self):
        rows = [make_row({"trait_id": f"T{i}"}) for i in range(5)]
        assert len(clean_from_dict_list(rows)) == 5

    def test_removes_invalid_value_999(self):
        rows = [make_row(), make_row({"trait_id": "T002", "value": 999})]
        assert len(clean_from_dict_list(rows)) == 1

    def test_removes_none_value(self):
        rows = [make_row(), make_row({"trait_id": "T002", "value": None})]
        assert len(clean_from_dict_list(rows)) == 1

    def test_output_is_list_of_dicts(self):
        result = clean_from_dict_list([make_row()])
        assert isinstance(result, list) and isinstance(result[0], dict)

    def test_empty_list_returns_empty(self):
        assert clean_from_dict_list([]) == []

    def test_crops_normalized_lowercase(self):
        rows = [make_row({"crop": "WHEAT"}), make_row({"trait_id": "T002", "crop": "Rice"})]
        for row in clean_from_dict_list(rows):
            assert row["crop"] == row["crop"].lower()

    def test_duplicates_removed(self):
        rows = [make_row(), make_row()]
        assert len(clean_from_dict_list(rows)) == 1

    def test_required_fields_retained(self):
        result = clean_from_dict_list([make_row()])
        for f in REQUIRED_COLUMNS:
            assert f in result[0]
