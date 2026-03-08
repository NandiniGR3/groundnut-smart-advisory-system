#!/usr/bin/env python3
"""
env_data_pipeline.py — Production-ready pipeline

- NASA POWER timeseries (daily or monthly) with retries/backoff
- Optional Google Earth Engine (GEE) monthly NDVI/EVI extraction (safe fallback)
- Per-district raw CSVs + combined raw CSV + parquet (when pyarrow available)
- Cleaned CSVs + parquet
- Monthly aggregation and merging (weather, NDVI, optional soil/disease/yield)
- Robust error handling and logging

Before using:
- pip install requests pandas pyarrow
- For GEE NDVI: pip install earthengine-api and run `earthengine authenticate` once interactively,
  then set USE_GEE = True below.
"""
from __future__ import annotations
import os
import time
import logging
import requests
import pandas as pd
import datetime
from typing import Dict, Tuple, List

# ---------------------------
# USER CONFIGURATION
# ---------------------------
RAW_DIR = "raw_data"
RAW_WEATHER_DIR = os.path.join(RAW_DIR, "weather_by_district")
RAW_NDVI_DIR = os.path.join(RAW_DIR, "ndvi_by_district")
CLEAN_DIR = "clean_data"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(RAW_WEATHER_DIR, exist_ok=True)
os.makedirs(RAW_NDVI_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

# Time range (default last 10 years)
END_DATE = datetime.date.today()
START_DATE = END_DATE.replace(year=END_DATE.year - 9, day=1)

# NASA POWER configuration
NASA_BASE_URL = "https://power.larc.nasa.gov/api/temporal/{tstep}/point"
NASA_COMMUNITY = "AG"
NASA_TIMESTEP = "daily"  # "daily" or "monthly"
NASA_PARAMS = [
    "T2M",         # 2m air temperature (°C)
    "PRECTOTCORR", # precipitation (mm/day)
    "RH2M",        # relative humidity (%)
    "WS2M",        # wind speed (m/s)
    "T2M_MAX",
    "T2M_MIN"
]
NASA_MAX_RETRIES = 3
NASA_BACKOFF = 1.8  # exponential base

# Districts (Karnataka)
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

# Optional local clean files to merge
SOIL_FILE = os.path.join(CLEAN_DIR, "soil_clean.csv")
DISEASE_FILE = os.path.join(CLEAN_DIR, "disease_severity.csv")
YIELD_FILE = os.path.join(CLEAN_DIR, "yield_clean.csv")

# Earth Engine usage flag (True only if ee auth completed on this machine)
USE_GEE = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ---------------------------
# Utilities
# ---------------------------
def sanitize_filename(name: str) -> str:
    allowed = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in allowed else "_" for c in name).replace(" ", "_")


# ---------------------------
# NASA POWER fetch (retries/backoff)
# ---------------------------
def fetch_nasa_power_for_point(lat: float, lon: float,
                               start: datetime.date, end: datetime.date,
                               parameters: List[str], timestep: str = "daily",
                               max_retries: int = NASA_MAX_RETRIES, backoff: float = NASA_BACKOFF) -> pd.DataFrame:
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    params_str = ",".join(parameters)
    url = NASA_BASE_URL.format(tstep=timestep)
    payload = {
        "parameters": params_str,
        "community": NASA_COMMUNITY,
        "longitude": lon,
        "latitude": lat,
        "start": start_str,
        "end": end_str,
        "format": "JSON"
    }

    attempt = 0
    while attempt < max_retries:
        try:
            resp = requests.get(url, params=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            param_block = data.get("properties", {}).get("parameter", {})
            if not param_block:
                logging.warning("NASA response missing parameter block for (%s,%s)", lat, lon)
                return pd.DataFrame()
            first = next(iter(param_block.values()))
            dates = sorted(first.keys())
            rows = []
            fmt = "%Y%m%d" if timestep == "daily" else "%Y%m"
            for d in dates:
                try:
                    date = pd.to_datetime(d, format=fmt)
                except Exception:
                    date = pd.to_datetime(d, errors="coerce")
                row = {"date": date}
                for p_name, mapping in param_block.items():
                    val = mapping.get(d) if isinstance(mapping, dict) else None
                    row[p_name.lower()] = val
                rows.append(row)
            df = pd.DataFrame(rows)
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            attempt += 1
            wait = backoff ** attempt
            logging.warning("NASA fetch failed (attempt %d/%d) for (%s,%s): %s — retrying in %.1fs", attempt, max_retries, lat, lon, e, wait)
            time.sleep(wait)
    logging.error("NASA fetch failed after %d attempts for (%s,%s).", max_retries, lat, lon)
    return pd.DataFrame()


# ---------------------------
# GEE NDVI helper (safe)
# ---------------------------
def init_gee_safe() -> bool:
    if not USE_GEE:
        return False
    try:
        import ee  # type: ignore
        ee.Initialize()
        logging.info("Google Earth Engine initialized.")
        return True
    except Exception as e:
        logging.warning("GEE init failed: %s — will use NDVI placeholders.", e)
        return False


def get_ndvi_series_gee(lat: float, lon: float, start: datetime.date, end: datetime.date, ee_initialized: bool) -> pd.DataFrame:
    months = pd.date_range(start=start, end=end, freq="M")
    if not ee_initialized:
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
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
        df.columns = [c.lower() for c in df.columns]
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        logging.error("GEE NDVI extraction failed for (%s,%s): %s — placeholders returned.", lat, lon, e)
        return pd.DataFrame({"date": months, "ndvi": [None] * len(months), "evi": [None] * len(months)})


# ---------------------------
# Cleaning helpers
# ---------------------------
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


def load_optional(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        elif "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            df["date"] = pd.NaT
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        return df
    except Exception as e:
        logging.error("Failed loading optional file %s: %s", path, e)
        return pd.DataFrame()


# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline():
    logging.info("Pipeline start — districts=%d start=%s end=%s", len(DISTRICTS), START_DATE, END_DATE)

    ee_ready = init_gee_safe()

    weather_parts = []
    ndvi_parts = []

    for district, (lat, lon) in DISTRICTS.items():
        logging.info("Processing: %s", district)

        # NASA
        df_weather = fetch_nasa_power_for_point(lat, lon, START_DATE, END_DATE, NASA_PARAMS, timestep=NASA_TIMESTEP)
        if not df_weather.empty:
            df_weather["district"] = district
            fn = os.path.join(RAW_WEATHER_DIR, f"{sanitize_filename(district)}_weather_raw.csv")
            df_weather.to_csv(fn, index=False)
            weather_parts.append(df_weather)
            logging.info("Saved per-district weather CSV: %s (rows=%d)", fn, len(df_weather))
        else:
            logging.warning("Empty weather for %s", district)

        # NDVI
        df_ndvi = get_ndvi_series_gee(lat, lon, START_DATE, END_DATE, ee_ready)
        if not df_ndvi.empty:
            df_ndvi["district"] = district
            df_ndvi.columns = [c.lower() for c in df_ndvi.columns]
            fn2 = os.path.join(RAW_NDVI_DIR, f"{sanitize_filename(district)}_ndvi_raw.csv")
            df_ndvi.to_csv(fn2, index=False)
            ndvi_parts.append(df_ndvi)
            logging.info("Saved per-district NDVI CSV: %s (rows=%d)", fn2, len(df_ndvi))
        else:
            logging.warning("Empty NDVI for %s", district)

        time.sleep(0.8)  # politeness

    # combine raw
    if weather_parts:
        weather_df = pd.concat(weather_parts, ignore_index=True, sort=False)
        weather_df.columns = [c.lower() for c in weather_df.columns]
        if "date" in weather_df.columns:
            weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce")
        weather_df.to_csv(os.path.join(RAW_DIR, "weather_nasa_raw.csv"), index=False)
        try:
            weather_df.to_parquet(os.path.join(RAW_DIR, "weather_nasa_raw.parquet"), index=False)
        except Exception:
            logging.info("pyarrow not available for weather parquet.")
        logging.info("Saved combined weather raw (rows=%d).", len(weather_df))
    else:
        weather_df = pd.DataFrame()
        logging.warning("No weather collected.")

    if ndvi_parts:
        ndvi_df = pd.concat(ndvi_parts, ignore_index=True, sort=False)
        ndvi_df.columns = [c.lower() for c in ndvi_df.columns]
        if "date" in ndvi_df.columns:
            ndvi_df["date"] = pd.to_datetime(ndvi_df["date"], errors="coerce")
        ndvi_df.to_csv(os.path.join(RAW_DIR, "ndvi_raw.csv"), index=False)
        try:
            ndvi_df.to_parquet(os.path.join(RAW_DIR, "ndvi_raw.parquet"), index=False)
        except Exception:
            logging.info("pyarrow not available for ndvi parquet.")
        logging.info("Saved combined NDVI raw (rows=%d).", len(ndvi_df))
    else:
        ndvi_df = pd.DataFrame()
        logging.warning("No ndvi collected.")

    # clean
    weather_clean = clean_df(weather_df)
    ndvi_clean = clean_df(ndvi_df)

    # ---------------------------
    # FIX: clean NDVI & weather numeric columns BEFORE aggregation
    # ---------------------------
    import numpy as np
    # NDVI clean
    if not ndvi_clean.empty:
        ndvi_col = "ndvi" if "ndvi" in ndvi_clean.columns else None
        if ndvi_col:
            ndvi_clean[ndvi_col] = ndvi_clean[ndvi_col].replace({"None": np.nan, "": np.nan, "nan": np.nan})
            ndvi_clean[ndvi_col] = pd.to_numeric(ndvi_clean[ndvi_col], errors="coerce")
        if "evi" in ndvi_clean.columns:
            ndvi_clean["evi"] = ndvi_clean["evi"].replace({"None": np.nan, "": np.nan, "nan": np.nan})
            ndvi_clean["evi"] = pd.to_numeric(ndvi_clean["evi"], errors="coerce")

    # weather numeric cleaning: attempt numeric convert for non-date/district columns
    if not weather_clean.empty:
        for col in weather_clean.columns:
            if col not in ("date", "district"):
                weather_clean[col] = pd.to_numeric(weather_clean[col], errors="coerce")

    # save cleaned
    if not weather_clean.empty:
        weather_clean.to_csv(os.path.join(CLEAN_DIR, "weather_clean.csv"), index=False)
        try:
            weather_clean.to_parquet(os.path.join(CLEAN_DIR, "weather_clean.parquet"), index=False)
        except Exception:
            logging.info("pyarrow not installed; skipping weather parquet.")
    if not ndvi_clean.empty:
        ndvi_clean.to_csv(os.path.join(CLEAN_DIR, "ndvi_clean.csv"), index=False)
        try:
            ndvi_clean.to_parquet(os.path.join(CLEAN_DIR, "ndvi_clean.parquet"), index=False)
        except Exception:
            logging.info("pyarrow not installed; skipping ndvi parquet.")

    # monthly aggregation
    if not weather_clean.empty and "date" in weather_clean.columns:
        weather_clean["year_month"] = weather_clean["date"].dt.to_period("M").dt.to_timestamp()
        agg_map = {}
        for col in weather_clean.columns:
            if col in ("date", "district", "year_month"):
                continue
            if any(k in col for k in ("prec", "rain", "prectot")):
                agg_map[col] = "sum"
            else:
                agg_map[col] = "mean"
        weather_monthly = weather_clean.groupby(["district", "year_month"], as_index=False).agg(agg_map)
    else:
        weather_monthly = pd.DataFrame()

    if not ndvi_clean.empty and "date" in ndvi_clean.columns:
        ndvi_clean["year_month"] = ndvi_clean["date"].dt.to_period("M").dt.to_timestamp()
        available = {}
        if "ndvi" in ndvi_clean.columns:
            available["ndvi"] = "mean"
        if "evi" in ndvi_clean.columns:
            available["evi"] = "mean"
        if available:
            ndvi_monthly = ndvi_clean.groupby(["district", "year_month"], as_index=False).agg(available)
        else:
            ndvi_monthly = pd.DataFrame()
    else:
        ndvi_monthly = pd.DataFrame()

    # load optional local files
    soil_df = load_optional(SOIL_FILE)
    disease_df = load_optional(DISEASE_FILE)
    yield_df = load_optional(YIELD_FILE)

    # merge chain
    merged = pd.DataFrame()
    if not weather_monthly.empty:
        merged = weather_monthly.copy()
    elif not ndvi_monthly.empty:
        merged = ndvi_monthly.copy()

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
            logging.info("pyarrow not installed; skipping merged parquet.")
        logging.info("Saved merged dataset (rows=%d)", len(merged))
    else:
        logging.warning("Merged dataset empty.")

    logging.info("Pipeline finished.")


if __name__ == "__main__":
    run_pipeline()
