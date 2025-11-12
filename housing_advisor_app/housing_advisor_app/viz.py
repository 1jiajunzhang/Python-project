import folium
import numpy as np
from streamlit_folium import st_folium
import streamlit as st
from typing import Optional, Dict, Any


def render_map(gdf, selected_postcode: Optional[str] = None,
               highlight_address: Optional[str] = None) -> Dict[str, Any]:
    if gdf.empty:
        st.warning("No listings to display on the map.")
        return {}

    center_lat = float(gdf["lat"].mean())
    center_lon = float(gdf["lon"].mean())

    if highlight_address:
        sub = gdf[gdf["full_address"] == highlight_address]
        if not sub.empty:
            center_lat = float(sub.iloc[0]["lat"])
            center_lon = float(sub.iloc[0]["lon"])

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    for _, row in gdf.iterrows():
        color = "#2563eb"
        if selected_postcode and row["postcode"] == selected_postcode:
            color = "#f97316"
        if highlight_address and row["full_address"] == highlight_address:
            color = "#22c55e"

        popup_html = f"""
        <b>{row['full_address']}</b><br>
        Price: ${row['price']:,.0f}<br>
        {row['beds']} bd / {row['baths']} ba, {row['square_feet']} sqft<br>
        DOM: {row['days_on_market']} days
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=popup_html,
        ).add_to(m)

    return st_folium(m, width="100%", height=520)


def find_listing_from_click(map_data, gdf):
    if not map_data:
        return None
    click = map_data.get("last_object_clicked")
    if not click:
        return None
    lat = click.get("lat")
    lng = click.get("lng")
    if lat is None or lng is None:
        return None
    coords = gdf[["lat", "lon"]].to_numpy()
    d2 = (coords[:, 0] - lat) ** 2 + (coords[:, 1] - lng) ** 2
    idx = int(np.argmin(d2))
    return gdf.iloc[idx]
