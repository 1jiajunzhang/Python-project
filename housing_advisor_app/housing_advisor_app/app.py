#cd C:\Users\szzddx\Desktop\housing_advisor_app
#cd .\housing_advisor_app
#dir
#streamlit run app.py
import streamlit as st
import pandas as pd

from config import APP_TITLE, ensure_data_exists
from data_io import attach_geometry, get_filter_options
from models_trend import compute_trend_score
from models_ratio import price_to_rent_ratio, ratio_score, affordability_score
from decision_engine import DecisionEngine, DecisionInputs
from viz import render_map, find_listing_from_click


#App 初始化
ensure_data_exists()
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Interactive NYC housing explorer with map, filters, and Buy / Watch / Avoid advisor (3-factor)."
)

engine = DecisionEngine()


#侧栏：筛选
st.sidebar.header("Search & Filter")

# 先加载一次（不触发编码），仅用于拿到筛选选项；若首次无缓存，这里可能为空
try:
    gdf_preview = attach_geometry(use_address_geocoding=False)
    options = get_filter_options(gdf_preview) if not gdf_preview.empty else {"postcodes": []}
except Exception:
    options = {"postcodes": []}

selected_zip = st.sidebar.selectbox(
    "Postcode (ZIP)",
    options=["All"] + options["postcodes"],
    index=0,
)


max_price_default = 5_000_000
if options["postcodes"] and "price" in getattr(gdf_preview, "columns", []):
    try:
        p90 = int(gdf_preview["price"].quantile(0.9))
        max_price_default = max(1_000_000, p90)
    except Exception:
        pass

min_price, max_price = st.sidebar.slider(
    "Price range ($)",
    min_value=0,
    max_value=int(max_price_default * 1.5),
    value=(0, int(max_price_default)),
    step=5000,
)

min_sqft, max_sqft = st.sidebar.slider(
    "Size range (sqft)",
    min_value=0,
    max_value=6000,
    value=(0, 2500),
    step=50,
)

bed_filter = st.sidebar.selectbox("Min beds", options=[0, 1, 2, 3, 4, 5], index=0)
bath_filter = st.sidebar.selectbox("Min baths", options=[0.0, 1.0, 1.5, 2.0, 3.0], index=0)

address_query = st.sidebar.text_input(
    "Search by address keyword",
    value="",
    help="Type part of an address or postcode to filter.",
)


# 侧栏：地址地理编码
st.sidebar.markdown("---")
st.sidebar.subheader("Address Geocoding (Nominatim + local cache)")

use_geocode = st.sidebar.checkbox(
    "Use address-level geocoding (slower, cached)",
    value=True,
    help="Turn on to geocode addresses to real lat/lon using local parquet cache."
)

batch = st.sidebar.number_input(
    "Batch size per run",
    min_value=50, max_value=2000, value=600, step=50,
    help="How many new addresses to geocode this time (respects 1 req/sec)."
)

do_geocode = st.sidebar.button("Geocode now")

# 若点击按钮：先做一轮增量编码（写入缓存），再读取
if do_geocode:
    _ = attach_geometry(use_address_geocoding=True, batch_limit=int(batch))
    st.sidebar.success("Geocoding finished for this batch. Map refreshed.")

# 根据开关状态加载地图数据（带或不带缓存坐标）
gdf = attach_geometry(use_address_geocoding=use_geocode, batch_limit=int(batch))


#  应用筛选
df = gdf.copy()

if selected_zip != "All":
    df = df[df["postcode"] == selected_zip]

df = df[
    (df["price"] >= min_price)
    & (df["price"] <= max_price)
    & (df["square_feet"] >= min_sqft)
    & (df["square_feet"] <= max_sqft)
    & (df["beds"] >= bed_filter)
    & (df["baths"] >= bath_filter)
]

if address_query.strip():
    q = address_query.strip()
    mask = (
        df["full_address"].str.contains(q, case=False, na=False)
        | df["address"].str.contains(q, case=False, na=False)
        | df["postcode"].str.contains(q, case=False, na=False)
        | df.get("neighborhood", pd.Series("", index=df.index)).astype(str).str.contains(q, case=False, na=False)
    )
    df = df[mask]

st.write(f"**{len(df)}** matching listings (from total {len(gdf)} geocoded listings).")


# ========== 地图 + 列表 ==========
col_map, col_table = st.columns([2, 1.7])

with col_map:
    highlight_address = df.iloc[0]["full_address"] if not df.empty else None
    map_data = render_map(
        df,
        selected_postcode=None if selected_zip == "All" else selected_zip,
        highlight_address=highlight_address,
    )
    clicked_listing = find_listing_from_click(map_data, df) if not df.empty else None

with col_table:
    st.subheader("Listings")
    if df.empty:
        st.info("No listings under current filters.")
    else:
        show_cols = [
            "full_address",
            "postcode",
            "price",
            "beds",
            "baths",
            "square_feet",
            "days_on_market",
        ]
        display = df[show_cols].copy()
        display["price"] = display["price"].map(lambda x: f"${x:,.0f}")
        st.dataframe(display, use_container_width=True, height=430)

    if clicked_listing is not None:
        st.markdown("**Selected from map:**")
        st.code(
            f"{clicked_listing['full_address']} | "
            f"${clicked_listing['price']:,.0f} | "
            f"{clicked_listing['beds']} bd / {clicked_listing['baths']} ba, "
            f"{int(clicked_listing['square_feet'])} sqft, "
            f"{int(clicked_listing['days_on_market'])} days on market",
            language="text",
        )



st.markdown("---")
st.subheader("Buy-Timing Advisor (3-factor)")

target_listing = None
if clicked_listing is not None:
    target_listing = clicked_listing
elif not df.empty:
    target_listing = df.iloc[0]

st.caption(
    "Pick a point on the map or keep current selection. "
    "Enter income & rent below, then run the analysis."
)

c_income, c_rent, c_run = st.columns([1, 1, 1])
with c_income:
    user_income = st.number_input(
        "Your annual income ($)",
        min_value=0,
        value=100000,
        step=5000,
    )
with c_rent:
    est_monthly_rent = st.number_input(
        "Comparable monthly rent ($)",
        min_value=0,
        value=2500,
        step=100,
    )
with c_run:
    run_button = st.button("Run Buy/Watch/Avoid Analysis", use_container_width=True)

if target_listing is not None and run_button:
    # 1) 趋势分数：同 ZIP 序列的短长均线差（见 models_trend.compute_trend_score）
    same_zip = gdf[gdf["postcode"] == target_listing["postcode"]]
    ts = same_zip["price"].sort_values()
    t_score = compute_trend_score(ts)

    # 2) 租售比 & 评分（见 models_ratio）
    ptr = price_to_rent_ratio(
        price=target_listing["price"],
        monthly_rent=est_monthly_rent if est_monthly_rent > 0 else None,
    )
    r_score = ratio_score(ptr)

    # 3) 可负担性评分
    a_score = affordability_score(
        price=target_listing["price"],
        income=user_income if user_income > 0 else None,
    )

    decision_inputs = DecisionInputs(
        trend_score=t_score,
        ratio_score=r_score,
        affordability_score=a_score,
    )
    result = engine.decide(decision_inputs)

    # 三个指标展示
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Trend score", f"{t_score:.2f}")
    with m2:
        st.metric("P/R ratio", f"{ptr:.1f}" if (ptr is not None and ptr > 0) else "N/A")
    with m3:
        st.metric("Affordability score", f"{a_score:.2f}")

    st.markdown(f"### Recommendation: **{result.label}**  (composite score {result.score:.3f})")
    st.write(result.explanation)

    st.info(
        f"Target listing: {target_listing['full_address']}  |  "
        f"${target_listing['price']:,.0f}, "
        f"{target_listing['beds']} bd / {target_listing['baths']} ba, "
        f"{int(target_listing['square_feet'])} sqft."
    )
elif run_button and target_listing is None:
    st.warning("No target listing available under current filters. Please adjust filters or click a point on the map.")
