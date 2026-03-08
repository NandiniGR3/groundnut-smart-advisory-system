#!/usr/bin/env python3
"""
eda_analysis.py — Fixed and robust version of your earlier EDA / preprocessing script.

- NASA POWER fetch (with retries/backoff)
- Optional Google Earth Engine (GEE) NDVI/EVI extraction with graceful fallback
- Per-district raw saves, combined raw saves
- Cleaned CSVs & parquet (if pyarrow available)
- Numeric cleaning to avoid 'None' string -> TypeError during groupby
- Monthly aggregation and merging with optional soil/disease/yield files
- Uses pd.period_range(...).to_timestamp() to avoid freq('M'/'ME') ambiguity
"""
from __future__ import annotations
import os
import time
import logging
import requests
import pandas as pd
import numpy as np
import datetime
from typing import Dict, Tuple, List

# --------------------
# Paths & Folders
# --------------------
ROOT = os.path.dirname(os.path.abspath(__file__))  # backend/preprocessing
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))  # adjust if needed
RAW_DIR = os.path.join(PROJECT_ROOT, "raw_data")
CLEAN_DIR = os.path.join(PROJECT_ROOT, "clean_data")
RAW_WEATHER_DIR = os.path.join(RAW_DIR, "weather_by_district")
RAW_NDVI_DIR = os.path.join(RAW_DIR, "ndvi_by_district")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(RAW_WEATHER_DIR, exist_ok=True)
os.makedirs(RAW_NDVI_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

# --------------------
# Optional local file paths (change if your filenames differ)
# --------------------
SOIL_FILE = os.path.join(CLEAN_DIR, "soil_clean.csv")
DISEASE_FILE = os.path.join(CLEAN_DIR, "disease_severity.csv")   # if not present script will continue
YIELD_FILE = os.path.join(CLEAN_DIR, "yield_clean.csv")          # if not present script will continue

# --------------------
# Settings
# --------------------
END_DATE = datetime.date.today()
START_DATE = END_DATE.replace(year=END_DATE.year - 9, day=1)  # ~10 years

NASA_BASE_URL = "https://power.larc.nasa.gov/api/temporal/{tstep}/point"
NASA_PARAMS = ["T2M", "PRECTOTCORR", "RH2M", "WS2M", "T2M_MAX", "T2M_MIN"]
NASA_TIMESTEP = "daily"
NASA_MAX_RETRIES = 3
NASA_BACKOFF = 1.8

# If you have authenticated Earth Engine on this machine, set to True
USE_GEE = False

DISTRICTS: Dict[str, Tuple[float, float]] = {
    "Bagalkot": (16.18, 75.70),
    "Ballari": (15.14, 76.92),
    "Belagavi": (15.85, 74.50),
    "Bidar": (17.91, 77.52),
    "Chamarajanagar": (11.93, 76.94),
    "Chikkaballapur": (13.43, 77.73),
    "Chikkamagaluru": (13.32, 75.77),
    "Chitradurga": (14.23, 76.40),
    "Dakshina Kannada": (12.87, 75.00),
    "Davanagere": (14.47, 75.92),
    "Dharwad": (15.46, 75.01),
    "Gadag": (15.43, 75.63),
    "Hassan": (13.00, 76.10),
    "Haveri": (14.79, 75.40),
    "Kalaburagi": (17.33, 76.83),
    "Kodagu": (12.34, 75.80),
    "Kolar": (13.14, 78.13),
    "Koppal": (15.35, 76.15),
    "Mandya": (12.52, 76.90),
    "Mysuru": (12.30, 76.64),
    "Raichur": (16.20, 77.35),
    "Ramanagara": (12.72, 77.28),
    "Shimoga": (13.92, 75.57),
    "Tumakuru": (13.34, 77.10),
    "Udupi": (13.34, 74.74),
    "Uttara Kannada": (14.80, 74.12),
    "Vijayanagara": (15.30, 76.39),
    "Yadgir": (16.76, 77.13),
    "Bengaluru Urban": (12.97, 77.59),
    "Bengaluru Rural": (13.10, 77.50)
}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# --------------------
# Utility helpers
# --------------------
def sanitize_filename(name: str) -> str:
    allowed = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in allowed else "_" for c in name).replace(" ", "_")


# --------------------
# NASA POWER fetch with retries/backoff
# --------------------
def fetch_nasa(lat: float, lon: float, start: datetime.date, end: datetime.date,
               parameters: List[str], timestep: str = "daily",
               max_retries: int = NASA_MAX_RETRIES, backoff: float = NASA_BACKOFF) -> pd.DataFrame:
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    url = NASA_BASE_URL.format(tstep=timestep)
    payload = {
        "parameters": ",".join(parameters),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_str,
        "end": end_str,
        "format": "JSON"
    }
    attempt = 0
    while attempt < max_retries:
        try:
            r = requests.get(url, params=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            params_block = data.get("properties", {}).get("parameter", {})
            if not params_block:
                logging.warning("NASA response missing parameter block for (%s,%s)", lat, lon)
                return pd.DataFrame()
            first = next(iter(params_block.values()))
            dates = sorted(first.keys())
            rows = []
            fmt = "%Y%m%d" if timestep == "daily" else "%Y%m"
            for d in dates:
                try:
                    dt = pd.to_datetime(d, format=fmt)
                except Exception:
                    dt = pd.to_datetime(d, errors="coerce")
                row = {"date": dt}
                for p_name, mapping in params_block.items():
                    row[p_name.lower()] = mapping.get(d) if isinstance(mapping, dict) else None
                rows.append(row)
            df = pd.DataFrame(rows)
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            attempt += 1
            wait = backoff ** attempt
            logging.warning("NASA fetch fail attempt %d/%d for (%s,%s): %s — retry in %.1fs", attempt, max_retries, lat, lon, e, wait)
            time.sleep(wait)
    logging.error("NASA fetch failed after %d attempts for (%s,%s)", max_retries, lat, lon)
    return pd.DataFrame()


# --------------------
# GEE NDVI helper (safe)
# --------------------
def init_gee_safe() -> bool:
    if not USE_GEE:
        return False
    try:
        import ee  # type: ignore
        ee.Initialize()
        logging.info("GEE initialized.")
        return True
    except Exception as e:
        logging.warning("GEE init failed: %s — NDVI placeholders will be used.", e)
        return False


def get_ndvi_gee(lat: float, lon: float, start: datetime.date, end: datetime.date, ee_ready: bool) -> pd.DataFrame:
    # Build months using period_range -> to_timestamp (month end) to avoid freq warning/compat issues
    months = pd.period_range(start=start, end=end, freq="M").to_timestamp(how="end")
    if not ee_ready:
        return pd.DataFrame({"date": months, "ndvi": [None] * len(months), "evi": [None] * len(months)})

    try:
        import ee  # type: ignore
        point = ee.Geometry.Point([lon, lat])
        collection = ee.ImageCollection("MODIS/006/MOD13A1").filterDate(start.isoformat(), end.isoformat()).filterBounds(point)

        def img_to_feature(img):
            date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            mean = img.reduceRegion(ee.Reducer.mean(), point, 500, bestEffort=True)
            ndvi = mean.get("NDVI")
            evi = mean.get("EVI")
            return ee.Feature(None, {"date": date, "ndvi": ndvi, "evi": evi})

        features = collection.map(img_to_feature).filter(ee.Filter.notNull(["ndvi"]))
        fc = features.getInfo()
        rows = []
        for f in fc.get("features", []):
            props = f.get("properties", {})
            if not props:
                continue
            d = props.get("date")
            ndvi = props.get("ndvi")
            evi = props.get("evi")
            try:
                d_parsed = pd.to_datetime(d)
            except Exception:
                continue
            rows.append({"date": d_parsed, "ndvi": ndvi, "evi": evi})
        if not rows:
            return pd.DataFrame({"date": months, "ndvi": [None] * len(months), "evi": [None] * len(months)})
        df = pd.DataFrame(rows)
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp(how="end")
        df.columns = [c.lower() for c in df.columns]
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        logging.error("GEE NDVI failed for (%s,%s): %s — returning placeholders.", lat, lon, e)
        return pd.DataFrame({"date": months, "ndvi": [None] * len(months), "evi": [None] * len(months)})


# --------------------
# load optional local file
# --------------------
def load_optional(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # normalize columns
        df.columns = [c.lower() for c in df.columns]
        # normalize date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp(how="end")
        elif "date" in df.columns or "date" in df.keys():
            df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
            df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp(how="end")
        return df
    except Exception as e:
        logging.error("Failed to load optional file %s: %s", path, e)
        return pd.DataFrame()


# --------------------
# cleaning functions
# --------------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.drop_duplicates(inplace=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.columns = [c.lower() for c in df.columns]
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    return df


# --------------------
# main run pipeline
# --------------------
def run_pipeline():
    logging.info("Starting EDA pipeline — districts=%d start=%s end=%s", len(DISTRICTS), START_DATE, END_DATE)

    ee_ready = init_gee_safe()

    weather_parts = []
    ndvi_parts = []

    for dname, (lat, lon) in DISTRICTS.items():
        logging.info("Collecting for %s (lat=%.3f lon=%.3f)", dname, lat, lon)

        # NASA
        w = fetch_nasa(lat, lon, START_DATE, END_DATE, NASA_PARAMS, timestep=NASA_TIMESTEP)
        if not w.empty:
            w["district"] = dname
            w.columns = [c.lower() for c in w.columns]
            # ensure date is datetime
            if "date" in w.columns:
                w["date"] = pd.to_datetime(w["date"], errors="coerce")
            fn = os.path.join(RAW_WEATHER_DIR, f"{sanitize_filename(dname)}_weather_raw.csv")
            w.to_csv(fn, index=False)
            weather_parts.append(w)
            logging.info("Saved per-district weather for %s (rows=%d)", dname, len(w))
        else:
            logging.warning("No weather for %s", dname)

        # NDVI
        n = get_ndvi_gee(lat, lon, START_DATE, END_DATE, ee_ready)
        if not n.empty:
            n["district"] = dname
            n.columns = [c.lower() for c in n.columns]
            if "date" in n.columns:
                n["date"] = pd.to_datetime(n["date"], errors="coerce")
            fn2 = os.path.join(RAW_NDVI_DIR, f"{sanitize_filename(dname)}_ndvi_raw.csv")
            n.to_csv(fn2, index=False)
            ndvi_parts.append(n)
            logging.info("Saved per-district ndvi for %s (rows=%d)", dname, len(n))
        else:
            logging.warning("No NDVI for %s", dname)

        time.sleep(0.6)  # politeness

    # combine raw
    weather_df = pd.concat(weather_parts, ignore_index=True, sort=False) if weather_parts else pd.DataFrame()
    ndvi_df = pd.concat(ndvi_parts, ignore_index=True, sort=False) if ndvi_parts else pd.DataFrame()

    # normalize & save combined raw
    if not weather_df.empty:
        weather_df.columns = [c.lower() for c in weather_df.columns]
        if "date" in weather_df.columns:
            weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce")
        weather_combined = os.path.join(RAW_DIR, "weather_nasa_raw.csv")
        weather_df.to_csv(weather_combined, index=False)
        try:
            weather_df.to_parquet(os.path.join(RAW_DIR, "weather_nasa_raw.parquet"), index=False)
        except Exception:
            logging.info("Parquet write skipped for weather (pyarrow not installed).")
        logging.info("Saved combined weather raw: %s (rows=%d)", weather_combined, len(weather_df))
    else:
        logging.warning("No weather data collected.")

    if not ndvi_df.empty:
        ndvi_df.columns = [c.lower() for c in ndvi_df.columns]
        if "date" in ndvi_df.columns:
            ndvi_df["date"] = pd.to_datetime(ndvi_df["date"], errors="coerce")
        ndvi_combined = os.path.join(RAW_DIR, "ndvi_raw.csv")
        ndvi_df.to_csv(ndvi_combined, index=False)
        try:
            ndvi_df.to_parquet(os.path.join(RAW_DIR, "ndvi_raw.parquet"), index=False)
        except Exception:
            logging.info("Parquet write skipped for ndvi (pyarrow not installed).")
        logging.info("Saved combined ndvi raw: %s (rows=%d)", ndvi_combined, len(ndvi_df))
    else:
        logging.warning("No ndvi data collected.")

    # cleaning
    weather_clean = clean_df(weather_df)
    ndvi_clean = clean_df(ndvi_df)

    # ---------------------------
    # FIX: Clean NDVI & weather numeric columns BEFORE aggregation
    # ---------------------------
    # Convert 'None' (string) and empty strings to NaN and then numeric
    if not ndvi_clean.empty:
        ndvi_col = "ndvi" if "ndvi" in ndvi_clean.columns else None
        if ndvi_col:
            ndvi_clean[ndvi_col] = ndvi_clean[ndvi_col].replace({"None": np.nan, "": np.nan, "nan": np.nan})
            ndvi_clean[ndvi_col] = pd.to_numeric(ndvi_clean[ndvi_col], errors="coerce")
        if "evi" in ndvi_clean.columns:
            ndvi_clean["evi"] = ndvi_clean["evi"].replace({"None": np.nan, "": np.nan, "nan": np.nan})
            ndvi_clean["evi"] = pd.to_numeric(ndvi_clean["evi"], errors="coerce")

    # weather numeric cleaning: coerce non-date/district columns to numeric where reasonable
    if not weather_clean.empty:
        for col in weather_clean.columns:
            if col not in ("date", "district"):
                weather_clean[col] = pd.to_numeric(weather_clean[col], errors="coerce")

    # save cleaned outputs
    if not weather_clean.empty:
        weather_clean.to_csv(os.path.join(CLEAN_DIR, "weather_clean.csv"), index=False)
        try:
            weather_clean.to_parquet(os.path.join(CLEAN_DIR, "weather_clean.parquet"), index=False)
        except Exception:
            logging.info("Parquet for weather_clean skipped (pyarrow not installed).")
        logging.info("Saved weather_clean (rows=%d)", len(weather_clean))

    if not ndvi_clean.empty:
        ndvi_clean.to_csv(os.path.join(CLEAN_DIR, "ndvi_clean.csv"), index=False)
        try:
            ndvi_clean.to_parquet(os.path.join(CLEAN_DIR, "ndvi_clean.parquet"), index=False)
        except Exception:
            logging.info("Parquet for ndvi_clean skipped (pyarrow not installed).")
        logging.info("Saved ndvi_clean (rows=%d)", len(ndvi_clean))

    # ---------------------------
    # Monthly aggregation (groupby)
    # ---------------------------
    if not weather_clean.empty and "date" in weather_clean.columns:
        # month-end timestamp via period -> timestamp (avoid freq deprecation)
        weather_clean["year_month"] = weather_clean["date"].dt.to_period("M").dt.to_timestamp(how="end")
        agg_map = {}
        for col in weather_clean.columns:
            if col in ("date", "district", "year_month"):
                continue
            # precipitation/rainfall-like fields summed; others averaged
            if any(k in col for k in ("prec", "rain", "prectot")):
                agg_map[col] = "sum"
            else:
                agg_map[col] = "mean"
        weather_monthly = weather_clean.groupby(["district", "year_month"], as_index=False).agg(agg_map)
        logging.info("Built weather_monthly (rows=%d)", len(weather_monthly))
    else:
        weather_monthly = pd.DataFrame()
        logging.warning("weather_monthly empty (no weather_clean).")

    if not ndvi_clean.empty and "date" in ndvi_clean.columns:
        ndvi_clean["year_month"] = ndvi_clean["date"].dt.to_period("M").dt.to_timestamp(how="end")
        available = {}
        if "ndvi" in ndvi_clean.columns:
            available["ndvi"] = "mean"
        if "evi" in ndvi_clean.columns:
            available["evi"] = "mean"
        ndvi_monthly = ndvi_clean.groupby(["district", "year_month"], as_index=False).agg(available) if available else pd.DataFrame()
        logging.info("Built ndvi_monthly (rows=%d)", len(ndvi_monthly))
    else:
        ndvi_monthly = pd.DataFrame()
        logging.warning("ndvi_monthly empty (no ndvi_clean).")

    # ---------------------------
    # Load optional local files
    # ---------------------------
    soil_df = load_optional(SOIL_FILE)
    disease_df = load_optional(DISEASE_FILE)
    yield_df = load_optional(YIELD_FILE)

    # ---------------------------
    # Merge datasets
    # ---------------------------
    merged = pd.DataFrame()
    if not weather_monthly.empty:
        merged = weather_monthly.copy()
    elif not ndvi_monthly.empty:
        merged = ndvi_monthly.copy()
    else:
        logging.warning("No base dataset to merge from (both weather_monthly and ndvi_monthly empty).")

    if not merged.empty and not ndvi_monthly.empty:
        merged = merged.merge(ndvi_monthly, on=["district", "year_month"], how="left")
    if not merged.empty and not soil_df.empty:
        merged = merged.merge(soil_df, on=["district", "year_month"], how="left")
    if not merged.empty and not disease_df.empty:
        merged = merged.merge(disease_df, on=["district", "year_month"], how="left")
    if not merged.empty and not yield_df.empty:
        merged = merged.merge(yield_df, on=["district", "year_month"], how="left")

    if not merged.empty:
        merged.to_csv(os.path.join(CLEAN_DIR, "environment_monthly_merged.csv"), index=False)
        try:
            merged.to_parquet(os.path.join(CLEAN_DIR, "environment_monthly_merged.parquet"), index=False)
        except Exception:
            logging.info("Parquet skipped for merged (pyarrow not installed).")
        logging.info("Saved merged dataset (rows=%d)", len(merged))
    else:
        logging.warning("Merged dataset empty — nothing saved.")

    logging.info("EDA pipeline finished.")


if __name__ == "__main__":
    run_pipeline()
