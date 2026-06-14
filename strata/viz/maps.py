"""
Interactive maps (Folium).

Builders return a folium.Map that the Streamlit app embeds and the demo saves
to standalone HTML. Maps are the platform's primary output surface because
every sector resolves to specific coordinates - known deposits, prospective
targets, priority sites, and clean-firm-power candidates.
"""
from __future__ import annotations

import folium

# commodity -> marker colour (folium's named icon colours)
_COMMODITY_COLOR = {
    "Li": "green", "REE": "purple", "Co": "blue", "Ni": "darkblue",
    "Cu": "orange", "Gr": "black", "Mn": "cadetblue", "Nb": "darkred",
    "U": "red",
}


def _center(df):
    return [float(df["lat"].mean()), float(df["lon"].mean())]


def minerals_map(catalog, targets=None, priority=None, zoom_start=4):
    """Map of known deposits (by commodity) plus prospective targets.

    catalog : DataFrame (name, lat, lon, commodity, status)
    targets : optional DataFrame (lat, lon, prospectivity) - new prospects
    priority: optional set/list of commodity codes to emphasise
    """
    m = folium.Map(location=_center(catalog), zoom_start=zoom_start,
                   tiles="CartoDB positron")
    known = folium.FeatureGroup(name="Known deposits")
    for _, s in catalog.iterrows():
        color = _COMMODITY_COLOR.get(s["commodity"], "gray")
        emph = " [PRIORITY]" if priority and s["commodity"] in priority else ""
        popup = f"{s['name']} ({s['state']}) - {s['commodity']}, {s['status']}{emph}"
        folium.CircleMarker(
            [s["lat"], s["lon"]], radius=6, color=color, fill=True,
            fill_opacity=0.85, popup=popup,
            tooltip=f"{s['name']} - {s['commodity']}",
        ).add_to(known)
    known.add_to(m)

    if targets is not None and len(targets):
        tg = folium.FeatureGroup(name="Prospective targets")
        for _, t in targets.iterrows():
            folium.Marker(
                [t["lat"], t["lon"]],
                icon=folium.Icon(color="red", icon="star"),
                popup=f"Prospect - score {t['prospectivity']:.3f}",
                tooltip=f"Prospect {t['prospectivity']:.2f}",
            ).add_to(tg)
        tg.add_to(m)

    folium.LayerControl().add_to(m)
    return m


def value_points_map(df, value_col, *, lat_col="lat", lon_col="lon",
                     caption=None, sample=600, zoom_start=11, seed=0):
    """Folium map of points coloured by a continuous value (e.g. real NDTI).

    Robust 5-95 percentile colour stretch; subsamples to ``sample`` points for
    responsiveness. Used to render real per-pixel index values on a basemap.
    """
    import branca.colormap as cm

    pts = df if len(df) <= sample else df.sample(sample, random_state=seed)
    m = folium.Map(location=[float(df[lat_col].mean()), float(df[lon_col].mean())],
                   zoom_start=zoom_start, tiles="CartoDB positron")
    vmin = float(pts[value_col].quantile(0.05))
    vmax = float(pts[value_col].quantile(0.95))
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    scale = cm.LinearColormap(["#2c7bb6", "#ffffbf", "#d7191c"],
                              vmin=vmin, vmax=vmax, caption=caption or value_col)
    for _, r in pts.iterrows():
        folium.CircleMarker(
            [r[lat_col], r[lon_col]], radius=3, weight=0, fill=True,
            fill_opacity=0.85, color=scale(float(r[value_col])),
        ).add_to(m)
    scale.add_to(m)
    return m


def siting_map(top_sites, mineral_sites=None, zoom_start=4):
    """Map of recommended clean-firm-power sites and (optional) mineral sites."""
    m = folium.Map(location=_center(top_sites), zoom_start=zoom_start,
                   tiles="CartoDB positron")
    fg = folium.FeatureGroup(name="Recommended sites")
    for i, t in top_sites.reset_index(drop=True).iterrows():
        folium.Marker(
            [t["lat"], t["lon"]],
            icon=folium.Icon(color="green", icon="bolt", prefix="fa"),
            popup=f"Site #{i+1} - suitability {t['score']:.3f}",
            tooltip=f"Site #{i+1} ({t['score']:.2f})",
        ).add_to(fg)
    fg.add_to(m)

    if mineral_sites is not None and len(mineral_sites):
        mg = folium.FeatureGroup(name="Critical-mineral sites")
        for _, s in mineral_sites.iterrows():
            folium.CircleMarker(
                [s["lat"], s["lon"]], radius=5,
                color=_COMMODITY_COLOR.get(s["commodity"], "gray"),
                fill=True, fill_opacity=0.8,
                tooltip=f"{s['name']} - {s['commodity']}",
            ).add_to(mg)
        mg.add_to(m)

    folium.LayerControl().add_to(m)
    return m
