import logging
from typing import List, Dict, Any, Tuple, Optional

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "trait_id": str,
    "crop": str,
    "trait_name": str,
    "value": float,
    "unit": str,
}
OPTIONAL_COLUMNS = {
    "temperature": float,
    "growth_stage": str,
    "stress_type": str,
    "source": str,
    "year": int,
    "location": str,
    "variety": str,
}

VALUE_RANGES = {
    "temperature": (-10, 65),
    "heat_tolerance_score": (0, 100),
    "yield_kg_ha": (0, 20000),
    "pollen_viability_pct": (0, 100),
    "germination_rate_pct": (0, 100),
    "leaf_area_index": (0, 15),
}
INVALID_VALUES = {-999, -9999, 999, 9999, 99999, -1, float("inf"), float("-inf")}

STRESS_SYNONYMS = {
    "high_temp": "heat",
    "high temperature": "heat",
    "thermal": "heat",
    "water_stress": "drought",
    "water deficit": "drought",
    "dry": "drought",
    "waterlog": "flood",
    "flooding": "flood",
    "salt": "salinity",
    "saline": "salinity",
    "control": "none",
    "normal": "none",
}


def validate_schema(df: "pd.DataFrame") -> Tuple["pd.DataFrame", List[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    warnings: List[str] = []
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            warnings.append(f"Optional column '{col}' missing — filling with None")
            df[col] = None

    return df, warnings


def handle_missing_values(df: "pd.DataFrame") -> Tuple["pd.DataFrame", Dict[str, int]]:
    stats: Dict[str, int] = {}

    for col in REQUIRED_COLUMNS:
        before = len(df)
        df = df.dropna(subset=[col])
        dropped = before - len(df)
        if dropped:
            stats[f"dropped_missing_{col}"] = dropped

    if "temperature" in df.columns and df["temperature"].isna().any():
        count_missing = int(df["temperature"].isna().sum())
        median_val = df["temperature"].median()
        df["temperature"] = df["temperature"].fillna(median_val)
        stats[f"filled_temperature_with_median"] = count_missing

    if "year" in df.columns and df["year"].isna().any():
        count_missing = int(df["year"].isna().sum())
        median_val = df["year"].median()
        df["year"] = df["year"].fillna(median_val)
        stats[f"filled_year_with_median"] = count_missing

    for col in ["growth_stage", "stress_type", "source", "location", "variety"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna("unknown")

    return df, stats


def normalize_temperature_column(df: "pd.DataFrame") -> "pd.DataFrame":
    if "temperature" not in df.columns:
        return df

    if "temperature_unit" in df.columns:
        mask_f = df["temperature_unit"].astype(str).str.lower().isin(["f", "°f", "fahrenheit"])
        mask_k = df["temperature_unit"].astype(str).str.lower().isin(["k", "kelvin"])

        df.loc[mask_f, "temperature"] = (df.loc[mask_f, "temperature"] - 32) * 5 / 9
        df.loc[mask_k, "temperature"] = df.loc[mask_k, "temperature"] - 273.15
        df = df.drop(columns=["temperature_unit"], errors="ignore")
    else:
        mask_k = df["temperature"] > 100
        df.loc[mask_k, "temperature"] = df.loc[mask_k, "temperature"] - 273.15

        mask_f = (df["temperature"] > 60) & (df["temperature"] <= 100)
        df.loc[mask_f, "temperature"] = (df.loc[mask_f, "temperature"] - 32) * 5 / 9

    return df


def remove_outliers(df: "pd.DataFrame") -> Tuple["pd.DataFrame", int]:
    initial = len(df)

    for col, (lo, hi) in VALUE_RANGES.items():
        if col in df.columns:
            df = df[(df[col] >= lo) & (df[col] <= hi)]

    for col in ["value", "temperature"]:
        if col in df.columns:
            df = df[~df[col].isin(INVALID_VALUES)]

    if "value" in df.columns and "trait_name" in df.columns:
        def iqr_filter(group: "pd.DataFrame") -> "pd.DataFrame":
            q1 = group["value"].quantile(0.25)
            q3 = group["value"].quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            return group[(group["value"] >= low) & (group["value"] <= high)]

        df = df.groupby("trait_name", group_keys=False, observed=True).apply(iqr_filter)

    return df, initial - len(df)


def deduplicate(df: "pd.DataFrame") -> Tuple["pd.DataFrame", int]:
    initial = len(df)
    cols = ["crop", "trait_name", "value"]
    if "temperature" in df.columns:
        cols.append("temperature")
    if "stress_type" in df.columns:
        cols.append("stress_type")
    if "growth_stage" in df.columns:
        cols.append("growth_stage")

    df = df.drop_duplicates(subset=cols, keep="first")
    return df, initial - len(df)


def normalize_strings(df: "pd.DataFrame") -> "pd.DataFrame":
    if "crop" in df.columns:
        df["crop"] = df["crop"].astype(str).str.lower().str.strip()

    if "stress_type" in df.columns:
        df["stress_type"] = (
            df["stress_type"].astype(str)
            .str.lower()
            .str.strip()
            .replace(STRESS_SYNONYMS)
        )

    if "trait_name" in df.columns:
        df["trait_name"] = (
            df["trait_name"].astype(str)
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.strip()
        )

    return df


def clean_parquet_file(input_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    if not PANDAS_AVAILABLE:
        raise ImportError("Run: pip install pandas pyarrow")

    df = pd.read_parquet(input_path, engine="pyarrow")
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    df, warnings = validate_schema(df)
    for w in warnings:
        logger.warning(f"  ⚠️  {w}")

    df, missing_stats = handle_missing_values(df)
    df = normalize_temperature_column(df)
    df, n_outliers = remove_outliers(df)
    df, n_dupes = deduplicate(df)
    df = normalize_strings(df)

    total_removed = missing_stats.get("total_dropped", 0) + n_outliers + n_dupes
    logger.info(
        f"✅ {len(df):,} clean rows (removed {total_removed:,} total: "
        f"{missing_stats.get('total_dropped', 0)} missing, "
        f"{n_outliers} outliers, {n_dupes} duplicates)"
    )

    if output_path:
        df.to_parquet(output_path, engine="pyarrow", compression="snappy")
        logger.info(f"Saved cleaned data to {output_path}")

    return df.to_dict(orient="records")


def clean_from_dict_list(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not PANDAS_AVAILABLE:
        cleaned = []
        for t in raw:
            if t.get("value") is None or t.get("value") in INVALID_VALUES:
                continue
            cleaned.append({
                "trait_id": t.get("trait_id"),
                "crop": str(t.get("crop", "unknown")).lower().strip(),
                "trait_name": str(t.get("trait_name", "unknown")).lower().replace(" ", "_").strip(),
                "value": t.get("value"),
                "unit": t.get("unit"),
                "temperature": t.get("temperature"),
                "growth_stage": t.get("growth_stage", "unknown"),
                "stress_type": STRESS_SYNONYMS.get(str(t.get("stress_type", "unknown")).lower(), str(t.get("stress_type", "unknown")).lower()),
                "source": t.get("source", "unknown"),
                "year": t.get("year"),
                "location": t.get("location", "unknown"),
                "variety": t.get("variety", "unknown"),
            })
        return cleaned

    df = pd.DataFrame(raw)
    df, _ = validate_schema(df)
    df, _ = handle_missing_values(df)
    df, _ = remove_outliers(df)
    df, _ = deduplicate(df)
    df = normalize_strings(df)
    return df.to_dict(orient="records")
