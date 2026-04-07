# search_engine/data_cleaner.py
# Cleans raw Parquet trait data before embedding.
# Pipeline: Load → Validate Schema → Handle Missing → Fix Units
#           → Remove Outliers → Deduplicate → Normalize Strings → Output

from typing import List, Dict, Any, Tuple, Optional
import time

start=time.time()
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

REQUIRED_COLUMNS = {"trait_id": str, "crop": str, "trait_name": str, "value": float, "unit": str}
OPTIONAL_COLUMNS = {"temperature": float, "growth_stage": str, "stress_type": str,
                    "source": str, "year": int, "location": str, "variety": str}

VALUE_RANGES = {
    "temperature": (-10, 65), "heat_tolerance_score": (0, 100),
    "yield_kg_ha": (0, 20000), "pollen_viability_pct": (0, 100),
    "germination_rate_pct": (0, 100), "leaf_area_index": (0, 15),
}
INVALID_VALUES = {-999, -9999, 999, 9999, 99999, -1, float("inf"), float("-inf")}

STRESS_SYNONYMS = {
    "high_temp": "heat", "high temperature": "heat", "thermal": "heat",
    "water_stress": "drought", "water deficit": "drought", "dry": "drought",
    "waterlog": "flood", "flooding": "flood", "inundation": "flood",
    "salt": "salinity", "saline": "salinity", "nacl": "salinity",
    "control": "none", "normal": "none"
}


def validate_schema(df: "pd.DataFrame") -> Tuple["pd.DataFrame", List[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    warnings = []
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            warnings.append(f"Optional column '{col}' missing — filling with None")
            df[col] = None
    return df, warnings


def handle_missing_values(df: "pd.DataFrame") -> Tuple["pd.DataFrame", Dict[str, int]]:
    stats = {}
    initial = len(df)
    for col in REQUIRED_COLUMNS:
        before = len(df)
        df = df.dropna(subset=[col])
        dropped = before - len(df)
        if dropped:
            stats[f"dropped_missing_{col}"] = dropped

    for col in ["temperature", "year"]:
        if col in df.columns and df[col].isna().any():
            median = df[col].median()
            stats[f"filled_{col}_with_median"] = int(df[col].isna().sum())
            df[col] = df[col].fillna(median)

    for col in ["growth_stage", "stress_type", "source", "location", "variety"]:
        if col in df.columns and df[col].isna().any():
            stats[f"filled_{col}_with_unknown"] = int(df[col].isna().sum())
            df[col] = df[col].fillna("unknown")

    stats["total_dropped"] = initial - len(df)
    return df, stats


def normalize_temperature_column(df: "pd.DataFrame") -> "pd.DataFrame":
    if "temperature" not in df.columns:
        return df
    if "temperature_unit" in df.columns:
        def convert(row):
            u = str(row.get("temperature_unit", "c")).lower()
            v = row["temperature"]
            if u in ("f", "°f", "fahrenheit"):
                return round((v - 32) * 5 / 9, 2)
            if u in ("k", "kelvin"):
                return round(v - 273.15, 2)
            return v
        df["temperature"] = df.apply(convert, axis=1)
        df = df.drop(columns=["temperature_unit"], errors="ignore")
    else:
        mask_k = df["temperature"] > 100
        df.loc[mask_k, "temperature"] = df.loc[mask_k, "temperature"] - 273.15
        mask_f = df["temperature"] > 60
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
        def iqr_filter(g):
            Q1, Q3 = g["value"].quantile(0.01), g["value"].quantile(0.99)
            IQR = Q3 - Q1
            return g[(g["value"] >= Q1 - 3 * IQR) & (g["value"] <= Q3 + 3 * IQR)]
        df = df.groupby("trait_name", group_keys=False).apply(iqr_filter)
    return df, initial - len(df)


def deduplicate(df: "pd.DataFrame") -> Tuple["pd.DataFrame", int]:
    initial = len(df)
    cols = ["crop", "trait_name", "value"]
    if "temperature" in df.columns:
        cols.append("temperature")
    df = df.drop_duplicates(subset=cols, keep="first")
    return df, initial - len(df)


def normalize_strings(df: "pd.DataFrame") -> "pd.DataFrame":
    if "crop" in df.columns:
        df["crop"] = df["crop"].str.lower().str.strip()
    if "stress_type" in df.columns:
        df["stress_type"] = df["stress_type"].str.lower().str.strip().replace(STRESS_SYNONYMS)
    if "trait_name" in df.columns:
        df["trait_name"] = df["trait_name"].str.lower().str.replace(" ", "_").str.strip()
    return df


def clean_parquet_file(input_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Full cleaning pipeline for a Parquet file."""
    if not PANDAS_AVAILABLE:
        raise ImportError("Run: pip install pandas pyarrow")
    df = pd.read_parquet(input_path, engine="pyarrow")
    print(f"[CLEAN] Loaded {len(df):,} rows from {input_path}")

    df, warnings = validate_schema(df)
    for w in warnings:
        print(f"  ⚠️  {w}")

    df, missing_stats = handle_missing_values(df)
    df = normalize_temperature_column(df)
    df, n_outliers = remove_outliers(df)
    df, n_dupes    = deduplicate(df)
    df             = normalize_strings(df)

    print(f"[CLEAN] ✅ {len(df):,} clean rows (removed {missing_stats.get('total_dropped', 0)} missing, "
          f"{n_outliers} outliers, {n_dupes} duplicates)")

    if output_path:
        df.to_parquet(output_path, engine="pyarrow", compression="snappy")

    return df.to_dict(orient="records")


def clean_from_dict_list(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean trait data from a plain Python list (no Parquet needed)."""
    if not PANDAS_AVAILABLE:
        cleaned = []
        for t in raw:
            if t.get("value") is None or t.get("value") in INVALID_VALUES:
                continue
            t["crop"]       = str(t.get("crop", "unknown")).lower().strip()
            t["trait_name"] = str(t.get("trait_name", "unknown")).lower().replace(" ", "_")
            cleaned.append(t)
        return cleaned

    df = pd.DataFrame(raw)
    df, _  = validate_schema(df)
    df, _  = handle_missing_values(df)
    df, _  = remove_outliers(df)
    df, _  = deduplicate(df)
    df     = normalize_strings(df)
    return df.to_dict(orient="records")

end=time.time()
print("data cleaned")
print('Total Time:',end-start)

