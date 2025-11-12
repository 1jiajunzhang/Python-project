
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st

from geopy.geocoders import Nominatim
from ratelimit import limits, sleep_and_retry

from config import (
    ROLLING_SALES_CSV,
    GEOCODE_CACHE,
    GEOCODE_CITY_HINT,
    GEOCODE_BATCH_LIMIT,
    GEOCODE_CALLS_PER_PERIOD,
    GEOCODE_PERIOD_SECONDS,
)


# 基础读取与清洗
@st.cache_data(show_spinner=False)
def load_raw_sales() -> pd.DataFrame:
    df = pd.read_csv(ROLLING_SALES_CSV)
    df.columns = [c.strip().upper() for c in df.columns]

    # 有效成交
    df["SALE PRICE"] = pd.to_numeric(df["SALE PRICE"], errors="coerce")
    df = df[df["SALE PRICE"] >= 100000].dropna(subset=["ADDRESS", "ZIP CODE", "SALE PRICE"])

    # ZIP 统一为 5 位
    df["ZIP CODE"] = (
        df["ZIP CODE"].astype(str)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(5)
    )


    for col in ["GROSS SQUARE FEET", "YEAR BUILT", "TOTAL UNITS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    if "BUILDING CLASS CATEGORY" not in df.columns:
        df["BUILDING CLASS CATEGORY"] = ""
    if "SALE DATE" in df.columns:
        df["SALE DATE"] = pd.to_datetime(df["SALE DATE"], errors="coerce")
    else:
        df["SALE DATE"] = pd.NaT

    return df


def _est_year_built(v): return 1960 if pd.isna(v) or v <= 0 else int(v)
def _est_sqft(v): return 900 if pd.isna(v) or v <= 0 else int(v)

def _est_beds(row):
    cat = str(row.get("BUILDING CLASS CATEGORY", "")).upper()
    units = row.get("TOTAL UNITS")
    if "ONE FAMILY" in cat: return 3
    if "TWO FAMILY" in cat: return 4
    if "THREE FAMILY" in cat: return 5
    if "COND" in cat or "COOP" in cat: return 2
    if pd.notna(units):
        if units <= 1: return 2
        if units == 2: return 3
        if units <= 4: return 4
        return 5
    return 2

def _est_baths(beds): return 2.0 if beds >= 4 else (1.5 if beds >= 2 else 1.0)


# 构造前端 listings
@st.cache_data(show_spinner=False)
def build_listings_base() -> pd.DataFrame:
    """不含坐标的 listings 基表（后续再补经纬度）"""
    df = load_raw_sales().copy()

    ranks = pd.Series(range(1, len(df) + 1), index=df.index)
    dom = 10 + (ranks % 81)

    listings = pd.DataFrame({
        "postcode": df["ZIP CODE"],
        "address": df["ADDRESS"].astype(str).str.title(),
        "year_built": df["YEAR BUILT"].apply(_est_year_built),
        "price": df["SALE PRICE"].round(0).astype(int),
        "square_feet": df["GROSS SQUARE FEET"].apply(_est_sqft),
        "neighborhood": df.get("NEIGHBORHOOD", pd.NA),
        "building_class_category": df.get("BUILDING CLASS CATEGORY", pd.NA),
        "sale_date": df.get("SALE DATE", pd.NaT),
        "days_on_market": dom,
    })
    listings["beds"] = df.apply(_est_beds, axis=1)
    listings["baths"] = listings["beds"].apply(_est_baths)
    listings["full_address"] = listings["address"] + ", " + listings["postcode"]
    return listings


# 地理编码缓存
def _load_cache() -> Dict[str, tuple]:
    if GEOCODE_CACHE.exists():
        c = pd.read_parquet(GEOCODE_CACHE)
        return {k: (float(lat), float(lon)) for k, lat, lon in zip(c["full_address"], c["lat"], c["lon"])}
    return {}

def _save_cache(cache_dict: Dict[str, tuple]):
    if not cache_dict:
        return
    out = pd.DataFrame([{"full_address": k, "lat": v[0], "lon": v[1]} for k, v in cache_dict.items()])
    out.to_parquet(GEOCODE_CACHE, index=False)


# 地理编码器
@sleep_and_retry
@limits(calls=GEOCODE_CALLS_PER_PERIOD, period=GEOCODE_PERIOD_SECONDS)
def _geocode_one(geolocator: Nominatim, query: str):
    try:
        return geolocator.geocode(query, timeout=10)
    except Exception:
        return None


def _fmt_query(row, city_hint: str) -> str:
    # 组合地址 + 城市提示 + 邮编
    return f"{row['address']}, {city_hint} {row['postcode']}"


def geocode_addresses_incremental(listings: pd.DataFrame,
                                  limit: Optional[int] = None,
                                  city_hint: str = GEOCODE_CITY_HINT) -> pd.DataFrame:
    """
    对没有 lat/lon 的记录做增量地理编码，并写入/更新缓存。
    每次调用只编码 limit 条，便于尊重免费服务的频率限制。
    """
    cache_map = _load_cache()

    # 只挑选缓存里没有的
    todo_mask = ~listings["full_address"].isin(cache_map.keys())
    todo = listings.loc[todo_mask].copy()
    if limit:
        todo = todo.head(int(limit))

    if todo.empty:
        return listings.assign(lat=listings.get("lat"), lon=listings.get("lon"))

    geolocator = Nominatim(user_agent="housing_advisor_app_geocoder")

    # 进度条
    progress = st.progress(0)
    status = st.empty()

    done = 0
    new_pairs = {}
    for idx, row in todo.iterrows():
        query = _fmt_query(row, city_hint)
        loc = _geocode_one(geolocator, query)
        if loc is not None:
            new_pairs[row["full_address"]] = (loc.latitude, loc.longitude)
        else:
            new_pairs[row["full_address"]] = (np.nan, np.nan)

        done += 1
        progress.progress(done / len(todo))
        status.write(f"Geocoding {done}/{len(todo)} : {query}")

    # 缓存
    cache_map.update(new_pairs)
    _save_cache(cache_map)

    # 回填到 listings
    lats, lons = [], []
    for addr in listings["full_address"]:
        if pd.notna(listings.get("lat")) is False:
            pass
        lat, lon = cache_map.get(addr, (np.nan, np.nan))
        lats.append(lat)
        lons.append(lon)

    enriched = listings.copy()
    if "lat" not in enriched.columns:
        enriched["lat"] = np.nan
        enriched["lon"] = np.nan
    enriched["lat"] = enriched["lat"].fillna(pd.Series(lats, index=enriched.index))
    enriched["lon"] = enriched["lon"].fillna(pd.Series(lons, index=enriched.index))
    return enriched


# 主入口
@st.cache_data(show_spinner=False)
def attach_geometry(use_address_geocoding: bool = False,
                    batch_limit: int = GEOCODE_BATCH_LIMIT) -> gpd.GeoDataFrame:

    lst = build_listings_base()

    if use_address_geocoding:
        lst = geocode_addresses_incremental(lst, limit=batch_limit)
    else:

        cache_map = _load_cache()
        if cache_map:
            lats = []
            lons = []
            for addr in lst["full_address"]:
                lat, lon = cache_map.get(addr, (np.nan, np.nan))
                lats.append(lat)
                lons.append(lon)
            lst["lat"] = lats
            lst["lon"] = lons


    lst = lst.dropna(subset=["lat", "lon"])

    gdf = gpd.GeoDataFrame(
        lst,
        geometry=gpd.points_from_xy(lst["lon"], lst["lat"]),
        crs="EPSG:4326",
    )
    return gdf


def get_filter_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    return {"postcodes": sorted(df["postcode"].dropna().unique().tolist())}
