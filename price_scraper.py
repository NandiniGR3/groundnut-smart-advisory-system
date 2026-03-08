# price_scraper.py
# --------------------------------------------------
# Groundnut Mandi Price Scraper - Karnataka
# Source: apmcprice.in
# --------------------------------------------------

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
URL = "https://www.apmcprice.in/groundnut-price-in-karnataka"
SAVE_DIR = "data/mandi_prices"
SAVE_FILE = "groundnut_karnataka.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-IN,en;q=0.9"
}

# --------------------------------------------------
# SCRAPER FUNCTION
# --------------------------------------------------
def scrape_groundnut_prices():
    print("Fetching mandi price data...")

    response = requests.get(URL, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table")
    if table is None:
        raise ValueError("Price table not found.")

    rows = table.find_all("tr")

    data = []
    for row in rows[1:]:
        cols = [c.text.strip() for c in row.find_all("td")]

        if len(cols) >= 5:
            data.append({
                "State": "Karnataka",
                "District": cols[0],
                "Market": cols[1],
                "Min_Price_Rs_Quintal": cols[2],
                "Max_Price_Rs_Quintal": cols[3],
                "Modal_Price_Rs_Quintal": cols[4],
                "Scrape_Date": datetime.now().strftime("%Y-%m-%d")
            })

    return pd.DataFrame(data)

# --------------------------------------------------
# CLEANING
# --------------------------------------------------
def clean_price_data(df):
    price_cols = [
        "Min_Price_Rs_Quintal",
        "Max_Price_Rs_Quintal",
        "Modal_Price_Rs_Quintal"
    ]

    for col in price_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "")
            .str.replace("₹", "")
            .astype(float)
        )

    df["District"] = df["District"].str.title()
    df["Market"] = df["Market"].str.title()

    return df.dropna()

# --------------------------------------------------
# SAVE
# --------------------------------------------------
def save_data(df):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, SAVE_FILE)
    df.to_csv(path, index=False)
    print(f"Data saved to {path}")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    df_raw = scrape_groundnut_prices()
    df_clean = clean_price_data(df_raw)
    save_data(df_clean)

    print("\nSample Data:")
    print(df_clean.head())
