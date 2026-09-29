"""Streamlit dashboard for the Vehicle Health Monitoring System.

    streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pdm.config import (ALL_MODELS, ARTIFACTS_DIR, CRITICAL_THRESHOLD, MODELS_DIR, PLOTS_DIR,
                        PRIMARY_MODEL, RAW_DATA_PATH, SENSOR_LABELS, SENSORS, WARNING_THRESHOLD)
from pdm.inference import VehicleHealthMonitor

st.set_page_config(page_title="Vehicle Health Monitor", page_icon="🚗", layout="wide")

STATUS_COLOR = {"NORMAL": "#16a34a", "WARNING": "#f59e0b", "CRITICAL": "#dc2626"}


@st.cache_resource
def load_monitor():
    return VehicleHealthMonitor()


@st.cache_data
def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    return df, meta


if not (MODELS_DIR / "meta.json").exists():
    st.error("No trained models found. Run `python train.py` first.")
    st.stop()

monitor = load_monitor()
df, meta = load_data()
test_ids = meta["test_vehicles"]
modes = df.groupby("vehicle_id").failure_mode.first()

st.title("🚗 AI/ML Predictive Maintenance & Vehicle Health Monitoring")
tab_vehicle, tab_fleet, tab_whatif, tab_models = st.tabs(
    ["Vehicle monitor", "Fleet overview", "What-if simulator", "Model performance"])


def gauge(value: float, title: str, color: str, invert=False):
    steps = ([{"range": [0, 40], "color": "#fee2e2"}, {"range": [40, 70], "color": "#fef3c7"},
              {"range": [70, 100], "color": "#dcfce7"}] if not invert else
             [{"range": [0, 100 * WARNING_THRESHOLD], "color": "#dcfce7"},
              {"range": [100 * WARNING_THRESHOLD, 100 * CRITICAL_THRESHOLD], "color": "#fef3c7"},
              {"range": [100 * CRITICAL_THRESHOLD, 100], "color": "#fee2e2"}])
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, number={"suffix": "%"},
                                 title={"text": title},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                                        "steps": steps}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def render_assessment(r: dict):
    color = STATUS_COLOR[r["status"]]
    c1, c2, c3 = st.columns([1, 1, 1.2])
    c1.plotly_chart(gauge(r["health_score"], "Vehicle Health", color), width="stretch")
    c2.plotly_chart(gauge(100 * r["failure_probability"], "Failure Probability (48 h)", color,
                          invert=True), width="stretch")
    with c3:
        st.markdown(f"### Status: <span style='color:{color}'>{r['status']}</span>",
                    unsafe_allow_html=True)
        if r["factors"]:
            st.markdown("**Major contributing factors:**")
            for i, f in enumerate(r["factors"], 1):
                st.markdown(f"{i}. {f['description'][0].upper() + f['description'][1:]} "
                            f"<span style='color:gray'>(SHAP {f['shap']:+.2f}, z={f['z_score']:+.1f})</span>",
                            unsafe_allow_html=True)
        else:
            st.markdown("No significant risk factors detected.")
        if r["alert"]:
            (st.error if r["status"] == "CRITICAL" else st.warning)(f"🔧 {r['alert']}")


# ---------------------------------------------------------------- Vehicle monitor
with tab_vehicle:
    s1, s2, s3 = st.columns([1, 1, 2])
    vid = s1.selectbox("Vehicle (held-out test fleet)", test_ids,
                       format_func=lambda v: f"{v}  [{modes[v]}]")
    model = s2.selectbox("Prediction model", [m for m in ALL_MODELS if m in meta["models"]],
                         index=meta["models"].index(PRIMARY_MODEL))
    vdf = df[df.vehicle_id == vid]
    t_min, t_max = int(vdf.timestamp.min()) + 1, int(vdf.timestamp.max())
    hour = s3.slider("Current operating hour (replay history)", t_min, t_max, t_max)
    hist = vdf[vdf.timestamp <= hour]

    r = monitor.assess(hist, model=model)
    render_assessment(r)

    timeline = monitor.probability_timeline(vdf[vdf.timestamp <= hour])
    left, right = st.columns([2, 1])
    with left:
        fig = go.Figure()
        fig.add_hrect(y0=100 * CRITICAL_THRESHOLD, y1=100, fillcolor="#fee2e2", line_width=0)
        fig.add_hrect(y0=100 * WARNING_THRESHOLD, y1=100 * CRITICAL_THRESHOLD, fillcolor="#fef3c7",
                      line_width=0)
        fig.add_trace(go.Scatter(x=timeline.index, y=100 * timeline.values, name="Failure prob.",
                                 line=dict(color="#1d4ed8")))
        fig.update_layout(title=f"Failure probability over time ({PRIMARY_MODEL})", height=300,
                          yaxis_title="%", xaxis_title="operating hour",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        contrib = pd.Series(r["sensor_contributions"]).sort_values()
        fig = go.Figure(go.Bar(x=contrib.values, y=[SENSOR_LABELS[s] for s in contrib.index],
                               orientation="h",
                               marker_color=["#dc2626" if v > 0 else "#2563eb" for v in contrib]))
        fig.update_layout(title="XAI: SHAP contribution per sensor", height=300,
                          xaxis_title="log-odds (red raises risk)", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    window = st.slider("Sensor history to display (hours)", 24, 400, 150, step=6)
    recent = hist.tail(window)
    fig = make_subplots(rows=3, cols=3, subplot_titles=[SENSOR_LABELS[s] for s in SENSORS],
                        vertical_spacing=0.09)
    for i, s in enumerate(SENSORS):
        row, col = i // 3 + 1, i % 3 + 1
        fig.add_trace(go.Scatter(x=recent.timestamp, y=recent[s], mode="lines", showlegend=False,
                                 line=dict(width=1.2)), row=row, col=col)
        if s in meta["baseline"]:
            b = meta["baseline"][s]
            fig.add_hrect(y0=b["p01"], y1=b["p99"], fillcolor="green", opacity=0.08, line_width=0,
                          row=row, col=col)
    fig.update_layout(height=650, title="Live sensor readings (green band = normal 1-99 % range)",
                      margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Ground truth (for evaluation only)"):
        ttf = vdf[vdf.timestamp == hour].time_to_failure.iloc[0]
        st.write(f"Failure mode: **{modes[vid]}** — "
                 + (f"actual failure in **{ttf:.0f} h**" if pd.notna(ttf) else "no failure in record"))

# ---------------------------------------------------------------- Fleet overview
with tab_fleet:
    st.markdown("Snapshot of the held-out test fleet. Choose how many hours before the end of each "
                "vehicle's record to take the snapshot.")
    back = st.slider("Hours before end of record", 0, 250, 60, step=10)

    @st.cache_data(show_spinner="Scoring fleet…")
    def fleet_snapshot(back: int) -> pd.DataFrame:
        rows = []
        for v in test_ids:
            h = df[df.vehicle_id == v]
            r = monitor.assess(h[h.timestamp <= h.timestamp.max() - back])
            rows.append({"vehicle": v, "health_%": round(r["health_score"]),
                         "failure_prob_%": round(100 * r["failure_probability"]),
                         "status": r["status"],
                         "top_factor": r["factors"][0]["description"] if r["factors"] else "",
                         "true_mode": modes[v]})
        return pd.DataFrame(rows).sort_values("failure_prob_%", ascending=False)

    fleet = fleet_snapshot(back)
    c = st.columns(4)
    c[0].metric("Vehicles", len(fleet))
    for col, s in zip(c[1:], ["NORMAL", "WARNING", "CRITICAL"]):
        col.metric(s, int((fleet.status == s).sum()))
    st.dataframe(
        fleet.style.map(lambda s: f"color:{STATUS_COLOR.get(s, 'inherit')};font-weight:600",
                        subset=["status"]),
        width="stretch", hide_index=True, height=600)

# ---------------------------------------------------------------- What-if simulator
with tab_whatif:
    st.markdown("Enter current sensor readings. The simulator builds a 48-hour history that drifts "
                "linearly from a normal baseline to these values, then runs the full pipeline.")
    defaults = {s: round(b["mean"], 2) for s, b in meta["baseline"].items()}
    defaults["operating_hours"] = 5000.0
    ranges = {"engine_temp": (60, 140), "rpm": (600, 6000), "oil_pressure": (0, 80),
              "vibration": (0, 10), "battery_voltage": (9, 16), "coolant_temp": (60, 140),
              "fuel_consumption": (0.5, 20), "vehicle_speed": (0, 140),
              "operating_hours": (0, 20000)}
    cols = st.columns(3)
    vals = {s: cols[i % 3].number_input(SENSOR_LABELS[s].capitalize(), *map(float, ranges[s]),
                                        defaults[s], key=f"wi_{s}")
            for i, s in enumerate(SENSORS)}
    wi_model = st.selectbox("Model", [m for m in ALL_MODELS if m in meta["models"]],
                            index=meta["models"].index(PRIMARY_MODEL), key="wi_model")
    n = 48
    frac = pd.Series(range(n)) / (n - 1)
    synth = pd.DataFrame({s: defaults[s] + (vals[s] - defaults[s]) * frac for s in SENSORS})
    synth["operating_hours"] = vals["operating_hours"] - (n - 1) + frac.index
    synth["vehicle_id"], synth["timestamp"] = "WHAT-IF", range(n)
    render_assessment(monitor.assess(synth, model=wi_model))

# ---------------------------------------------------------------- Model performance
with tab_models:
    metrics_path = ARTIFACTS_DIR / "metrics.csv"
    if metrics_path.exists():
        m = pd.read_csv(metrics_path, index_col=0)
        st.markdown("Test-set performance on held-out vehicles (task: failure within 48 h).")
        st.dataframe(m.style.format("{:.4f}").highlight_max(
            subset=["roc_auc", "pr_auc", "precision", "recall", "f1"], color="#bbf7d0"),
            width="stretch")
    for name, cap in [("roc_pr_curves.png", "ROC and precision-recall curves"),
                      ("confusion_matrix.png", "Confusion matrix (primary model)"),
                      ("sensor_importance.png", "Global sensor importance (SHAP)"),
                      ("shap_summary.png", "SHAP summary (engineered features)")]:
        if (PLOTS_DIR / name).exists():
            st.image(str(PLOTS_DIR / name), caption=cap)
