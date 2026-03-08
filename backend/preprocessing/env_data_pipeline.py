#!/usr/bin/env python3
"""
env_data_pipeline_fixed.py

Fixed and improved pipeline:
- NASA POWER daily/monthly time-series for many districts (with retry/backoff)
- Optional Google Earth Engine (GEE) monthly NDVI/EVI extraction (safe fallback)
- Saves per-district raw CSVs, combined raw CSV + Parquet, cleaned CSV + Parquet
- Aggregates to monthly and merges with optional local soil/disease/yield files

How to use:
1) Install dependencies:
   pip install requests pandas pyarrow

   If you want GEE NDVI:
   pip install earthengine-api
   earthengine authenticate   # run once interactively in terminal/browser
   (Then set USE_GEE = True below)

2) Run:
   python env_data_pipeline_fixed.py

3) Output files:
   raw_data/
     weather_by_district/<district>.csv
     weather_nasa_raw.csv
     weather_nasa_raw.parquet
     ndvi_by_district/<district>_ndvi.csv
     ndvi_raw.csv
   clean_data/
     weather_clean.csv, weather_clean.parquet
     ndvi_clean.csv, ndvi_clean.parquet
     environment_monthly_merged.csv, .parquet

Notes:
- If you cannot do interactive GEE auth on the current machine, leave USE_GEE=False.
- The script is intentionally conservative and uses placeholders for NDVI when GEE is unavailable.
"""
import os
import time
import logging
import requests
import pandas as pd
import datetime
from typing import List, Dict, Tuple

# -----------------------
# CONFIGURATION
# -----------------------
RAW_DIR = "raw_data"
RAW_WEATHER_DIR = os.path.join(RAW_DIR, "weather_by_district")
RAW_NDVI_DIR = os.path.join(RAW_DIR, "ndvi_by_district")
CLEAN_DIR = "clean_data"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(RAW_WEATHER_DIR, exist_ok=True)
os.makedirs(RAW_NDVI_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

# Time range: default last 10 years up to today (adjustable)
END_DATE = datetime.date.today()
START_DATE = END_DATE.replace(year=END_DATE.year - 9, day=1)

# NASA POWER config
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

# Districts (Karnataka) with approximate centroid lat/lon
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

# Optional local clean files to merge (set paths if you have them)
SOIL_FILE = os.path.join(CLEAN_DIR, "soil_clean.csv")
DISEASE_FILE = os.path.join(CLEAN_DIR, "disease_severity.csv")
YIELD_FILE = os.path.join(CLEAN_DIR, "yield_clean.csv")

# GEE settings (optional; requires earthengine-api and interactive auth)
USE_GEE = False  # set True if you have run `earthengine authenticate` interactively
if USE_GEE:
    try:
        import ee
        # Authenticate() opens a browser and requires user interaction (run once)
        # If you're automating in a headless server, prefer service account or leave USE_GEE=False
        ee.Authenticate()   # interactive step (may be skipped if already authenticated)
        ee.Initialize()
    except Exception as e:
        print("GEE initialization/auth failed:", e)
        print("Continuing with USE_GEE=False and NDVI placeholders.")
        USE_GEE = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# -----------------------
# NASA POWER helper (with retries)
# -----------------------
def fetch_nasa_power_for_point(lat: float, lon: float,
                               start: datetime.date, end: datetime.date,
                               parameters: List[str], timestep: str = "daily",
                               max_retries: int = 3, backoff: float = 1.5) -> pd.DataFrame:
    """
    Fetch NASA POWER timeseries for a single lat/lon point and return a DataFrame.
    Retries on failures with exponential backoff. Returns empty DataFrame on persistent failure.
    """
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
            params = data.get("properties", {}).get("parameter", {})
            if not params:
                logging.warning("NASA returned no 'parameter' block for point (%s,%s)", lat, lon)
                return pd.DataFrame()
            # choose dates from first parameter
            first_param = next(iter(params.values()))
            dates = sorted(first_param.keys())
            rows = []
            for d in dates:
                # daily format YYYYMMDD, monthly format YYYYMM
                fmt = "%Y%m%d" if timestep == "daily" else "%Y%m"
                try:
                    parsed_date = pd.to_datetime(d, format=fmt)
                except Exception:
                    # fallback to iso parsing
                    parsed_date = pd.to_datetime(d)
                row = {"date": parsed_date}
                for p_name, p_map in params.items():
                    # some params might be dict mapping date->value, some may be nested; handle gracefully
                    row[p_name.lower()] = p_map.get(d, None) if isinstance(p_map, dict) else None
                rows.append(row)
            df = pd.DataFrame(rows)
            # normalize column names to lowercase
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            attempt += 1
            wait = backoff ** attempt
            logging.warning("NASA request failed (attempt %d/%d): %s — retrying in %.1fs", attempt, max_retries, e, wait)
            time.sleep(wait)
    logging.error("NASA request failed after %d attempts for point (%s,%s). Returning empty DataFrame.", max_retries, lat, lon)
    return pd.DataFrame()


# -----------------------
# GEE NDVI helper (safe)
# -----------------------
def get_ndvi_series_gee(lat: float, lon: float, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    If USE_GEE is True and EE initialized, query MODIS/NDVI monthly images and return monthly series.
    Otherwise returns a placeholder monthly series with None values so pipeline continues.
    """
    months = pd.date_range(start=start, end=end, freq="ME")  # month ends
    if not USE_GEE:
        # placeholder DataFrame with 'date','ndvi','evi' columns (lowercase)
        return pd.DataFrame({"date": months, "ndvi": [None] * len(months), "evi": [None] * len(months)})

    try:
        import ee  # type: ignore
        point = ee.Geometry.Point([lon, lat])
        # Use MOD13A1 monthly product (MODIS monthly 500m)
        collection = ee.ImageCollection("MODIS/006/MOD13A1").filterDate(start.isoformat(), end.isoformat()).filterBounds(point)

        def extract_mean(img):
            mean = img.reduceRegion(ee.Reducer.mean(), point, 500)
            date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            return ee.Feature(None, {"date": date, "ndvi": mean.get("NDVI"), "evi": mean.get("EVI")})

        features = collection.map(extract_mean).filter(ee.Filter.notNull(["ndvi"]))
        # safe getInfo: convert to client-side list of features
        fc = features.getInfo()
        rows = []
        for f in fc.get("features", []):
            props = f.get("properties", {})
            if not props:
                continue
            # parse date, ndvi, evi
            try:
                d = pd.to_datetime(props.get("date"))
            except Exception:
                continue
            ndvi_val = props.get("ndvi")
            evi_val = props.get("evi")
            rows.append({"date": d, "ndvi": ndvi_val, "evi": evi_val})
        if not rows:
            # fallback to placeholder monthly index
            return pd.DataFrame({"date": months, "ndvi": [None] * len(months), "evi": [None] * len(months)})
        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)
        # ensure monthly alignment: convert date to month end timestamp
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp(freq="ME")
        return df
    except Exception as e:
        logging.error("GEE NDVI extraction failed for point (%s,%s): %s. Returning placeholder NDVI series.", lat, lon, e)
        return pd.DataFrame({"date": months, "ndvi": [None] * len(months), "evi": [None] * len(months)})


# -----------------------
# Cleaning helper
# -----------------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.drop_duplicates(inplace=True)
    # unify date -> datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # lowercase column names
    df.columns = [c.lower() for c in df.columns]
    # strip string columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    return df


# -----------------------
# Load optional local files safely
# -----------------------
def load_optional(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # normalize date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        elif "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            # no date -> leave empty
            df["date"] = pd.NaT
        df.columns = [c.lower() for c in df.columns]
        # add year_month for monthly merging
        if "date" in df.columns:
            df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp(freq="ME")
        return df
    except Exception as e:
        logging.error("Failed to load optional file %s: %s", path, e)
        return pd.DataFrame()


# -----------------------
# MAIN PIPELINE
# -----------------------
def run_pipeline():
    logging.info("Starting pipeline for %d districts from %s to %s", len(DISTRICTS), START_DATE, END_DATE)

    weather_all = []
    ndvi_all = []

    # Iterate districts, fetch NASA and NDVI; save per-district raw files as well
    for name, (lat, lon) in DISTRICTS.items():
        logging.info("Processing district: %s (lat=%s lon=%s)", name, lat, lon)

        # NASA POWER
        df_nasa = fetch_nasa_power_for_point(lat, lon, START_DATE, END_DATE, NASA_PARAMS, timestep=NASA_TIMESTEP)
        if df_nasa.empty:
            logging.warning("No NASA data for district %s (empty DataFrame)", name)
        else:
            # attach district col and save per-district raw
            df_nasa["district"] = name
            df_nasa.to_csv(os.path.join(RAW_WEATHER_DIR, f"{sanitize_filename(name)}_weather_raw.csv"), index=False)
            logging.info("Saved per-district NASA CSV for %s", name)
            weather_all.append(df_nasa)

        # NDVI via GEE (or placeholder)
        df_ndvi = get_ndvi_series_gee(lat, lon, START_DATE, END_DATE)
        if df_ndvi.empty:
            logging.warning("NDVI placeholder/empty for %s", name)
        else:
            df_ndvi["district"] = name
            # normalize column names to lowercase for consistent downstream processing
            df_ndvi.columns = [c.lower() for c in df_ndvi.columns]
            df_ndvi.to_csv(os.path.join(RAW_NDVI_DIR, f"{sanitize_filename(name)}_ndvi_raw.csv"), index=False)
            ndvi_all.append(df_ndvi)
            logging.info("Saved per-district NDVI CSV for %s", name)

        # polite pause for API friendliness
        time.sleep(1.0)

    # Combine and save raw combined files
    if weather_all:
        weather_df = pd.concat(weather_all, ignore_index=True, sort=False)
        # ensure 'date' exists and is datetime
        if "date" in weather_df.columns:
            weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce")
        weather_df.columns = [c.lower() for c in weather_df.columns]
        weather_df.to_csv(os.path.join(RAW_DIR, "weather_nasa_raw.csv"), index=False)
        try:
            weather_df.to_parquet(os.path.join(RAW_DIR, "weather_nasa_raw.parquet"), index=False)
        except Exception:
            logging.info("pyarrow/parquet write skipped (pyarrow not installed?)")
        logging.info("Saved combined NASA weather raw files: %s", os.path.join(RAW_DIR, "weather_nasa_raw.csv"))
    else:
        weather_df = pd.DataFrame()
        logging.warning("No weather data collected for any district.")

    if ndvi_all:
        ndvi_df = pd.concat(ndvi_all, ignore_index=True, sort=False)
        ndvi_df.columns = [c.lower() for c in ndvi_df.columns]
        ndvi_df.to_csv(os.path.join(RAW_DIR, "ndvi_raw.csv"), index=False)
        try:
            ndvi_df.to_parquet(os.path.join(RAW_DIR, "ndvi_raw.parquet"), index=False)
        except Exception:
            logging.info("pyarrow/parquet write skipped for NDVI")
        logging.info("Saved combined NDVI raw files: %s", os.path.join(RAW_DIR, "ndvi_raw.csv"))
    else:
        ndvi_df = pd.DataFrame()
        logging.warning("No NDVI data collected for any district (all placeholders).")

    # Clean dataframes
    weather_clean = clean_df(weather_df)
    ndvi_clean = clean_df(ndvi_df)

    # Save cleaned outputs
    if not weather_clean.empty:
        weather_clean.to_csv(os.path.join(CLEAN_DIR, "weather_clean.csv"), index=False)
        try:
            weather_clean.to_parquet(os.path.join(CLEAN_DIR, "weather_clean.parquet"), index=False)
        except Exception:
            logging.info("pyarrow not available for weather_clean.parquet")
    else:
        logging.warning("weather_clean is empty; nothing saved in clean_data for weather.")

    if not ndvi_clean.empty:
        ndvi_clean.to_csv(os.path.join(CLEAN_DIR, "ndvi_clean.csv"), index=False)
        try:
            ndvi_clean.to_parquet(os.path.join(CLEAN_DIR, "ndvi_clean.parquet"), index=False)
        except Exception:
            logging.info("pyarrow not available for ndvi_clean.parquet")
    else:
        logging.warning("ndvi_clean is empty; nothing saved in clean_data for ndvi.")

    # Aggregate to monthly for merging (year_month)
    if not weather_clean.empty and "date" in weather_clean.columns:
        weather_clean["year_month"] = weather_clean["date"].dt.to_period("M").dt.to_timestamp(freq="ME")
        # Build aggregation mapping: numeric columns -> mean except precipitation -> sum
        agg_funcs = {}
        for col in weather_clean.columns:
            if col in ("date", "district", "year_month"):
                continue
            if "prectot" in col or "prec" in col or "rain" in col:
                agg_funcs[col] = "sum"
            else:
                agg_funcs[col] = "mean"
        weather_monthly = weather_clean.groupby(["district", "year_month"], as_index=False).agg(agg_funcs)
    else:
        weather_monthly = pd.DataFrame()

    if not ndvi_clean.empty and "date" in ndvi_clean.columns:
        ndvi_clean["year_month"] = ndvi_clean["date"].dt.to_period("M").dt.to_timestamp(freq="ME")
        # standardize NDVI/EVI column names to 'ndvi' and 'evi' if possible
        ndvi_cols = [c for c in ndvi_clean.columns]
        ndvi_map = {}
        if "ndvi" in ndvi_cols:
            ndvi_map["ndvi"] = "ndvi"
        elif "NDVI" in ndvi_cols:
            ndvi_map["NDVI"] = "ndvi"
        if "evi" in ndvi_cols:
            ndvi_map["evi"] = "evi"
        elif "EVI" in ndvi_cols:
            ndvi_map["EVI"] = "evi"
        # perform a safe agg: mean of numeric ndvi/evi if present
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

    # Load optional local datasets (soil, disease, yield)
    soil_df = load_optional(SOIL_FILE)
    disease_df = load_optional(DISEASE_FILE)
    yield_df = load_optional(YIELD_FILE)

    # Prepare merge base (weather_monthly preferred)
    merged = weather_monthly.copy() if not weather_monthly.empty else pd.DataFrame()
    if not merged.empty and not ndvi_monthly.empty:
        # ensure both have same column name casing
        ndvi_monthly.columns = [c.lower() for c in ndvi_monthly.columns]
        merged = merged.merge(ndvi_monthly, on=["district", "year_month"], how="left")
    if not merged.empty and not soil_df.empty:
        merged = merged.merge(soil_df, left_on=["district", "year_month"], right_on=["district", "year_month"], how="left")
    if not merged.empty and not disease_df.empty:
        merged = merged.merge(disease_df, left_on=["district", "year_month"], right_on=["district", "year_month"], how="left")
    if not merged.empty and not yield_df.empty:
        merged = merged.merge(yield_df, left_on=["district", "year_month"], right_on=["district", "year_month"], how="left")

    if not merged.empty:
        merged.to_csv(os.path.join(CLEAN_DIR, "environment_monthly_merged.csv"), index=False)
        try:
            merged.to_parquet(os.path.join(CLEAN_DIR, "environment_monthly_merged.parquet"), index=False)
        except Exception:
            logging.info("pyarrow not available for merged parquet")
        logging.info("Saved merged dataset: %s", os.path.join(CLEAN_DIR, "environment_monthly_merged.csv"))
    else:
        logging.warning("Merged dataset is empty — nothing saved to clean_data.")

    logging.info("Pipeline run completed.")


# -----------------------
# Utility helper
# -----------------------
def sanitize_filename(name: str) -> str:
    """Simple file-safe name."""
    return "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in name).replace(" ", "_")


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    run_pipeline()
