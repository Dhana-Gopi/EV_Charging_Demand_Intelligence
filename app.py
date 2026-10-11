from pathlib import Path
import joblib

import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------
st.set_page_config(
    page_title="EV Charging Demand Dashboard",
    page_icon="⚡",
    layout="wide"
)


# --------------------------------------------------
# COMPACT FULL-WIDTH EXECUTIVE DASHBOARD THEME
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Use the full available screen */
    .stApp {
        background: #FFFEFA;
        color: #292821;
        font-family: 'Inter', sans-serif;
    }

    
    /* Keep content clear of Streamlit's top toolbar */
    .stApp [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 5rem !important;
        padding-bottom: 2rem !important;
    }

    [data-testid="stHeader"] {
        height: 1.8rem;
        background: #FFFEFA;
    }

    /* Compact typography */
    h1 {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        margin: 0.1rem 0 0.2rem 0 !important;
    }

    h2 {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        margin: 0.3rem 0 0.25rem 0 !important;
    }

    h3 {
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        margin: 0.2rem 0 0.15rem 0 !important;
    }

    p, label, .stMarkdown {
        font-size: 0.86rem;
        line-height: 1.25;
    }

    /* Reduce vertical whitespace between elements */
    [data-testid="stVerticalBlock"] {
        gap: 0.4rem;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 0.65rem;
    }

    /* Compact KPI cards */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #EAE2CE;
        border-top: 3px solid #E5C34F;
        border-radius: 9px;
        padding: 8px 12px;
        min-height: 75px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.76rem !important;
        color: #716D60 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        color: #292821 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.7rem !important;
    }

    /* Compact centered horizontal navigation */
    div[role="radiogroup"] {
        justify-content: center;
        gap: 5px;
        flex-wrap: wrap;
        padding: 2px 0 5px 0;
    }

    div[role="radiogroup"] label {
        background: #FFFFFF;
        border: 1px solid #E8E0C8;
        border-radius: 7px;
        padding: 5px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    div[role="radiogroup"] label[data-checked="true"] {
        background: #F9E7A0 !important;
        border-color: #D5AD26 !important;
    }

    /* Compact buttons and inputs */
    .stButton > button,
    .stFormSubmitButton > button {
        min-height: 32px;
        padding: 4px 10px;
        border-radius: 7px;
        background: #F2CF59 !important;
        color: #292821 !important;
        border: 1px solid #D8B43B !important;
        font-size: 0.8rem;
        font-weight: 700;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        min-height: 32px;
        border-color: #DCCB91 !important;
        border-radius: 7px;
    }

    /* Compact tables and alerts */
    [data-testid="stDataFrame"] {
        border: 1px solid #EAE2CE;
        border-radius: 8px;
    }

    [data-testid="stAlert"] {
        padding: 0.45rem 0.7rem;
        border-radius: 8px;
        background: #FFF8DF !important;
        border: 1px solid #E9D58A !important;
        color: #292821 !important;
    }

    hr {
        margin: 0.35rem 0 !important;
        border-color: #EAE2CE !important;
    }

    /* Hide the sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none;
    }

    /* Keep charts from adding excess surrounding space */
    [data-testid="stPlotlyChart"] {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Smaller screens: let columns wrap instead of becoming unreadable */
    @media (max-width: 900px) {
        .block-container {
            padding: 0.5rem !important;
        }
    }
    
    /* Executive header */
    .executive-header {
        width: 100%;
        text-align: center;
        padding: 0.5rem 0;
        margin: 0 auto;
    }
    
    .executive-eyebrow {
        color: #9B7815;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2.5px;
        margin-bottom: 4px;
    }
    
    .executive-title {
        color: #302D25;
        font-size: clamp(22px, 2.2vw, 32px);
        font-weight: 800;
        line-height: 1.2;
    }
    
    .executive-subtitle {
        color: #777264;
        font-size: 12px;
        margin-top: 5px;
    }
    
    /* Navigation alignment */
    div[role="radiogroup"] {
        display: flex;
        justify-content: center !important;
        align-items: center;
        gap: 6px !important;
        width: 100%;
        flex-wrap: wrap;
        padding: 4px 0 8px 0 !important;
    }
    
    div[role="radiogroup"] label {
        margin: 0 !important;
        padding: 5px 10px !important;
        border: 1px solid #EAE2CE;
        border-radius: 8px !important;
        background: #FFFFFF;
    }
    
    div[role="radiogroup"] label[data-checked="true"] {
        background: #F8EDC5 !important;
        border-color: #C9A227 !important;
    }
    
    /* Keep the dashboard content below the top edge */
    .stApp [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 2rem !important;
    }

    /* Center the navigation tabs */
    .stApp [data-testid="stTabs"] [data-testid="stTabList"] {
        justify-content: center !important;
        gap: 1.5rem;
        width: 100%;
    }
    
    /* Keep tab labels neatly aligned */
    .stApp [data-testid="stTabs"] button[role="tab"] {
        white-space: nowrap;
    }
    
    /* Center the horizontal radio navigation */
    .stApp div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        flex-wrap: wrap !important;
        gap: 0.5rem 1.25rem !important;
    }
    
    /* Keep each navigation option on one line */
    .stApp div[data-testid="stRadio"] label {
        white-space: nowrap !important;
    }

    
    /* Make navigation labels readable when not selected */
    .stApp [data-testid="stRadio"] label p {
        color: #302D25 !important;
        opacity: 1 !important;
    }
    
    /* Keep the radio navigation centered */
    .stApp [data-testid="stRadio"] > div[role="radiogroup"] {
        justify-content: center !important;
        align-items: center !important;
        flex-wrap: nowrap !important;
        gap: 1rem !important;
    }
    
    /* Improve contrast for the selected option */
    .stApp [data-testid="stRadio"] label:has(input:checked) p {
        color: #9B7815 !important;
        font-weight: 600 !important;
    }

    


    /* Final navigation override: centered, readable labels in all states */
    .stApp [data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        flex-wrap: wrap !important;
        gap: 0.45rem !important;
        width: 100% !important;
    }
    .stApp [data-testid="stRadio"] label {
        background: #FFFFFF !important;
        border: 1px solid #E8E0C8 !important;
        border-radius: 8px !important;
        padding: 5px 10px !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }
    .stApp [data-testid="stRadio"] label p,
    .stApp [data-testid="stRadio"] label span {
        color: #302D25 !important;
        opacity: 1 !important;
    }
    .stApp [data-testid="stRadio"] label:has(input:checked) {
        background: #F8EDC5 !important;
        border-color: #C9A227 !important;
    }
    .stApp [data-testid="stRadio"] label:has(input:checked) p,
    .stApp [data-testid="stRadio"] label:has(input:checked) span {
        color: #765B0E !important;
        font-weight: 700 !important;
    }
    /* One consistent top offset; avoid duplicate/competing padding rules */
    .stApp [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 100% !important;
        padding-top: 2.25rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        padding-bottom: 1.5rem !important;
    }
    .insight-card {
        background: #FFFFFF;
        border: 1px solid #EAE2CE;
        border-left: 3px solid #C9A227;
        border-radius: 9px;
        padding: 10px 12px;
        margin-bottom: 8px;
        color: #302D25;
    }
    .insight-label { color: #9B7815; font-weight: 800; font-size: 0.72rem; letter-spacing: 0.04em; }
    .insight-body { color: #4D493F; font-size: 0.82rem; line-height: 1.35; margin-top: 3px; }


    /* Give the objective its own space before the KPI grid. */
    .objective-panel {
        box-sizing: border-box;
        width: 100%;
        text-align: center;
        background: #FFFFFF;
        border: 1px solid #EAE2CE;
        border-radius: 10px;
        padding: 13px 18px;
        margin: 8px 0 16px 0;
        position: relative;
        clear: both;
    }
    .objective-heading {
        color: #302D25;
        font-size: 0.92rem;
        font-weight: 800;
        line-height: 1.4;
        margin: 0 0 5px 0;
    }
    .objective-copy {
        color: #777264;
        font-size: 0.86rem;
        line-height: 1.5;
        margin: 0;
    }
    [data-testid="stMetric"] {
        box-sizing: border-box !important;
        width: 100%;
        min-width: 0;
        overflow: hidden;
        margin-bottom: 8px;
    }
    [data-testid="stMetricValue"] {
        overflow-wrap: anywhere;
        word-break: normal;
        line-height: 1.25 !important;
    }
    /* Hide the per-element fullscreen toolbar and Plotly's own modebar. */
    [data-testid="stElementToolbar"] { display: none !important; }
    [data-testid="stPlotlyChart"] .modebar { display: none !important; }
    @media (max-width: 760px) {
        .objective-panel { padding: 11px 12px; margin-bottom: 14px; }
        [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    }
</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# PROJECT FILES
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "ChargingRecords.csv"

REGRESSION_PREPROCESSOR_FILE = BASE_DIR / "demand_preprocessor.pkl"
REGRESSION_MODEL_FILE = BASE_DIR / "lightgbm_demand_model.pkl"

CLASSIFICATION_DIR = BASE_DIR / "saved_models"
CLASSIFIER_FILE = CLASSIFICATION_DIR / "ev_demand_classifier.pkl"
CLASSIFICATION_PREPROCESSOR_FILE = CLASSIFICATION_DIR / "ev_preprocessor.pkl"
THRESHOLD_FILE = CLASSIFICATION_DIR / "demand_threshold.pkl"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_pickle(file_path):
    return joblib.load(file_path)


# --------------------------------------------------
# LOAD DATA AND MODELS
# --------------------------------------------------
try:
    data = load_data()

    regression_model = load_pickle(REGRESSION_MODEL_FILE)
    regression_preprocessor = load_pickle(REGRESSION_PREPROCESSOR_FILE)

    classifier = load_pickle(CLASSIFIER_FILE)
    classification_preprocessor = load_pickle(
        CLASSIFICATION_PREPROCESSOR_FILE
    )
    demand_threshold = load_pickle(THRESHOLD_FILE)

    

except Exception as error:
    st.error(f"Unable to load a dataset or model: {error}")
    st.stop()


# --------------------------------------------------
# TOP-CENTRED NAVIGATION
# --------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 8px 0 0 0;">
        <div style="font-size:12px; font-weight:700;
                    letter-spacing:2px; color:#9B812A;">
            EV CHARGING ANALYTICS
        </div>
        <h1 style="margin:5px 0 0 0; font-size:32px;">
            EV Charging Demand Intelligence
        </h1>
        <p style="color:#777264; margin-top:5px;">
            Usage patterns · Predictive insights · Demand outlook
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# ALIGNED EXECUTIVE HEADER + NAVIGATION
# --------------------------------------------------


page = st.radio(
        "Dashboard navigation",
        [
            "Executive Overview",
            "Regression Prediction",
            "Demand Classification",
            "Time-Series Forecasting"
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation"
    )

st.divider()

# --------------------------------------------------
# EXECUTIVE OVERVIEW — COMPACT ONE-PAGE DASHBOARD
# --------------------------------------------------
if page == "Executive Overview":
    overview = data.copy()
    for col in ["Demand", "Duration"]:
        if col in overview.columns:
            overview[col] = pd.to_numeric(overview[col], errors="coerce")
    if "StartDatetime" in overview.columns:
        overview["StartDatetime"] = pd.to_datetime(overview["StartDatetime"], errors="coerce")


    st.markdown(
        "<div class='objective-panel'>"
        "<div class='objective-heading'>Objective of this analysis</div>"
        "<div class='objective-copy'>Understand when and where charging demand is highest, compare usage patterns, "
        "and support evidence-based capacity planning.</div></div>",
        unsafe_allow_html=True
    )

    total_records = len(overview)
    locations = overview["Location"].nunique() if "Location" in overview.columns else 0
    avg_demand = overview["Demand"].mean() if "Demand" in overview.columns else np.nan
    avg_duration = overview["Duration"].mean() if "Duration" in overview.columns else np.nan
    total_hours = overview["Duration"].sum() / 60 if "Duration" in overview.columns else np.nan
    valid_time_demand = overview.dropna(subset=["StartDatetime", "Demand"]) if {"StartDatetime", "Demand"}.issubset(overview.columns) else pd.DataFrame()
    hourly = valid_time_demand.groupby(valid_time_demand["StartDatetime"].dt.hour)["Demand"].mean().reindex(range(24)) if not valid_time_demand.empty else pd.Series(dtype=float)
    peak_hour_text = f"{int(hourly.idxmax()):02d}:00" if not hourly.dropna().empty else "N/A"

    k1, k2, k3 = st.columns(3, gap="medium")
    k1.metric("Total Records", f"{total_records:,}")
    k2.metric("Average Demand", f"{avg_demand:,.2f}" if pd.notna(avg_demand) else "N/A")
    k3.metric("Avg. Duration", f"{avg_duration:,.1f} min" if pd.notna(avg_duration) else "N/A")

    k4, k5, k6 = st.columns(3, gap="medium")
    k4.metric("Total Charging Hours", f"{total_hours:,.0f}" if pd.notna(total_hours) else "N/A")
    k5.metric("Locations", f"{locations:,}")
    k6.metric("Peak-Demand Hour", peak_hour_text)

    st.divider()
    chart_col, insight_col = st.columns([3.15, 1.15], gap="medium")

    def compact_layout(fig, height=220):
        fig.update_layout(
            height=height, margin=dict(l=5, r=5, t=8, b=5),
            paper_bgcolor="#FFFEFA", plot_bgcolor="#FFFEFA",
            font=dict(family="Inter", size=10, color="#292821"),
            hoverlabel=dict(bgcolor="white", font_size=11, font_family="Inter")
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        return fig

    def chart_note(high_text, low_text, meaning):
        st.caption(f"**High:** {high_text}  ·  **Low:** {low_text}")
        st.caption(meaning)

    with chart_col:
        st.markdown("### Usage patterns & demand")
        row1a, row1b = st.columns(2, gap="small")
        with row1a:
            st.subheader("Demand by Location")
            if {"Location", "Demand"}.issubset(overview.columns):
                s = overview.groupby("Location")["Demand"].mean().dropna().sort_values()
                fig = go.Figure(go.Bar(x=s.values, y=s.index.astype(str), orientation="h", marker_color="#D8B64A",
                    hovertemplate="%{y}<br>Average demand: %{x:.2f}<extra></extra>"))
                fig.update_layout(xaxis_title="Average demand", yaxis_title="")
                st.plotly_chart(compact_layout(fig, max(210, min(280, 15 * len(s)))), use_container_width=True, key="overview_location")
                if not s.empty: chart_note(f"{s.idxmax()} ({s.max():.2f})", f"{s.idxmin()} ({s.min():.2f})", "Compares mean demand by location; high values may warrant capacity review.")
            else: st.info("Requires Location and Demand columns.")

        with row1b:
            st.subheader("Average Demand by Charger Type")
            if {"ChargerType", "Demand"}.issubset(overview.columns):
                s = overview.groupby("ChargerType")["Demand"].mean().dropna().sort_values(ascending=False)
                fig = go.Figure(go.Pie(labels=s.index.astype(str), values=s.values, hole=0.5, sort=False,
                    marker=dict(colors=["#9B7815", "#C9A227", "#D8C078", "#E8D58A", "#F8EDC5", "#B99529", "#E2C75A"], line=dict(color="#FFFEFA", width=2)),
                    textinfo="label+percent", textposition="inside",
                    hovertemplate="Charger type: %{label}<br>Average demand: %{value:.2f}<extra></extra>"))
                st.plotly_chart(compact_layout(fig, 220), use_container_width=True, key="overview_charger_type")
                if not s.empty: chart_note(f"{s.index[0]} ({s.iloc[0]:.2f})", f"{s.index[-1]} ({s.iloc[-1]:.2f})", "Shows relative average demand values by charger type, not session-count share.")
            else: st.info("Requires ChargerType and Demand columns.")

        row2a, row2b = st.columns(2, gap="small")
        with row2a:
            st.subheader("Demand by Charger Company")
            if {"ChargerCompany", "Demand"}.issubset(overview.columns):
                s = overview.groupby("ChargerCompany")["Demand"].mean().dropna().sort_values()
                fig = go.Figure(go.Bar(x=s.values, y=s.index.astype(str), orientation="h", marker_color="#C9A227",
                    hovertemplate="%{y}<br>Average demand: %{x:.2f}<extra></extra>"))
                fig.update_layout(xaxis_title="Average demand", yaxis_title="")
                st.plotly_chart(compact_layout(fig, max(210, min(280, 15 * len(s)))), use_container_width=True, key="overview_company")
                if not s.empty: chart_note(f"{s.idxmax()} ({s.max():.2f})", f"{s.idxmin()} ({s.min():.2f})", "Compares mean demand by company; investigate differences alongside charger mix and location.")
            else: st.info("Requires ChargerCompany and Demand columns.")

        with row2b:
            st.subheader("Average Demand by Month")
            if not valid_time_demand.empty:
                monthly = valid_time_demand.set_index("StartDatetime")["Demand"].resample("ME").mean().dropna()
                month_labels = monthly.index.strftime("%b %Y").tolist()
                fig = go.Figure(go.Scatter(x=month_labels, y=monthly.values, mode="lines+markers",
                    line=dict(color="#B99529", width=2.5), marker=dict(size=5),
                    customdata=month_labels,
                    hovertemplate="Month: %{customdata}<br>Average demand: %{y:.2f}<extra></extra>"))
                fig.update_layout(
                    xaxis=dict(title="", type="category", showgrid=False, showline=True,
                               categoryorder="array", categoryarray=month_labels),
                    yaxis=dict(title="Average demand", showgrid=False, zeroline=False, showline=True)
                )
                st.plotly_chart(compact_layout(fig, 220), use_container_width=True, key="overview_month")
                if not monthly.empty: chart_note(f"{monthly.idxmax():%b %Y} ({monthly.max():.2f})", f"{monthly.idxmin():%b %Y} ({monthly.min():.2f})", "Highlights month-to-month changes; a peak does not by itself establish seasonality.")
            else: st.info("Requires StartDatetime and Demand columns.")

        row3a, row3b = st.columns(2, gap="small")
        with row3a:
            st.subheader("Weekday vs Weekend")
            if not valid_time_demand.empty:
                wd = valid_time_demand.copy()
                wd["Period"] = np.where(wd["StartDatetime"].dt.dayofweek >= 5, "Weekend", "Weekday")
                s = wd.groupby("Period")["Demand"].mean().reindex(["Weekday", "Weekend"]).dropna()
                fig = go.Figure(go.Bar(x=s.index, y=s.values, marker_color=["#C9A227", "#9B7815"][:len(s)],
                    text=[f"{v:.2f}" for v in s.values], textposition="outside", cliponaxis=False,
                    hovertemplate="%{x}<br>Average demand: %{y:.2f}<extra></extra>"))
                fig.update_layout(yaxis_title="Average demand", xaxis_title="")
                st.plotly_chart(compact_layout(fig, 220), use_container_width=True, key="overview_weekend")
                if not s.empty: chart_note(f"{s.idxmax()} ({s.max():.2f})", f"{s.idxmin()} ({s.min():.2f})", "Compares average demand by day group, not total charging sessions.")
            else: st.info("Requires StartDatetime and Demand columns.")

        with row3b:
            st.subheader("Day × Hour Heatmap")
            if not valid_time_demand.empty:
                heat = valid_time_demand.copy()
                heat["Day"] = heat["StartDatetime"].dt.day_name()
                heat["Hour"] = heat["StartDatetime"].dt.hour
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                matrix = heat.pivot_table(index="Day", columns="Hour", values="Demand", aggfunc="mean").reindex(index=days, columns=range(24))
                fig = go.Figure(go.Heatmap(z=matrix.values, x=list(matrix.columns), y=list(matrix.index),
                    colorscale=[[0, "#FFF9E8"], [0.5, "#F2D56D"], [1, "#B78D13"]],
                    colorbar=dict(title="Demand", thickness=7),
                    hovertemplate="%{y}, %{x}:00<br>Average demand: %{z:.2f}<extra></extra>"))
                fig.update_layout(xaxis=dict(title="Hour", dtick=4), yaxis=dict(title=""))
                st.plotly_chart(compact_layout(fig, 220), use_container_width=True, key="overview_heatmap")
                if matrix.notna().any().any():
                    max_pos = np.unravel_index(np.nanargmax(matrix.to_numpy()), matrix.shape)
                    min_pos = np.unravel_index(np.nanargmin(matrix.to_numpy()), matrix.shape)
                    chart_note(f"{matrix.index[max_pos[0]]} {matrix.columns[max_pos[1]]:02d}:00", f"{matrix.index[min_pos[0]]} {matrix.columns[min_pos[1]]:02d}:00", "Cells show average demand for each weekday/hour combination.")
            else: st.info("Requires StartDatetime and Demand columns.")

        row4a, row4b = st.columns(2, gap="small")
        with row4a:
            st.subheader("Demand by Charging Duration")
            if {"Duration", "Demand"}.issubset(overview.columns):
                dd = overview.dropna(subset=["Duration", "Demand"]).copy()
                dd["Duration Band"] = pd.cut(dd["Duration"], bins=[-np.inf, 30, 60, 120, 240, np.inf], labels=["≤30 min", "31–60 min", "61–120 min", "121–240 min", ">240 min"])
                s = dd.groupby("Duration Band", observed=False)["Demand"].mean().dropna()
                fig = go.Figure(go.Bar(x=s.index.astype(str), y=s.values, marker_color="#D8B64A",
                    hovertemplate="%{x}<br>Average demand: %{y:.2f}<extra></extra>"))
                fig.update_layout(xaxis_title="Duration band", yaxis_title="Average demand")
                st.plotly_chart(compact_layout(fig, 220), use_container_width=True, key="overview_duration")
                if not s.empty: chart_note(f"{s.idxmax()} ({s.max():.2f})", f"{s.idxmin()} ({s.min():.2f})", "Duration bands can reveal differences between shorter and longer sessions; they do not prove duration causes demand.")
            else: st.info("Requires Duration and Demand columns.")

        with row4b:
            st.subheader("Hourly Demand Profile")
            if not hourly.dropna().empty:
                fig = go.Figure(go.Bar(x=hourly.index, y=hourly.values,
                    marker_color=["#9B7815" if h == hourly.idxmax() else "#E5C34F" for h in hourly.index],
                    hovertemplate="%{x}:00<br>Average demand: %{y:.2f}<extra></extra>"))
                fig.update_layout(xaxis=dict(title="Hour", dtick=4), yaxis_title="Average demand")
                st.plotly_chart(compact_layout(fig, 220), use_container_width=True, key="overview_hourly")
                low_hour = hourly.idxmin()
                chart_note(f"{int(hourly.idxmax()):02d}:00 ({hourly.max():.2f})", f"{int(low_hour):02d}:00 ({hourly.min():.2f})", "Use the peak and trough to identify periods for closer operational review.")
            else: st.info("Requires StartDatetime and Demand columns.")

    with insight_col:
        st.markdown("### Key takeaways")
        if {"Location", "Demand"}.issubset(overview.columns):
            loc_means = overview.groupby("Location")["Demand"].mean().dropna().sort_values(ascending=False)
            if not loc_means.empty:
                st.markdown(f"<div class='insight-card'><div class='insight-label'>LOCATION</div><b>Highest average demand</b><div class='insight-body'>{loc_means.index[0]} · {loc_means.iloc[0]:.2f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='insight-card'><div class='insight-label'>LOCATION</div><b>Lowest average demand</b><div class='insight-body'>{loc_means.index[-1]} · {loc_means.iloc[-1]:.2f}</div></div>", unsafe_allow_html=True)
        if not hourly.dropna().empty:
            st.markdown(f"<div class='insight-card'><div class='insight-label'>PEAK PERIOD</div><b>{int(hourly.idxmax()):02d}:00</b><div class='insight-body'>Highest hourly average demand: {hourly.max():.2f}.</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='insight-card'><div class='insight-label'>QUIETEST PERIOD</div><b>{int(hourly.idxmin()):02d}:00</b><div class='insight-body'>Lowest hourly average demand: {hourly.min():.2f}.</div></div>", unsafe_allow_html=True)
        if {"ChargerType", "Demand"}.issubset(overview.columns):
            type_means = overview.groupby("ChargerType")["Demand"].mean().dropna().sort_values(ascending=False)
            if not type_means.empty:
                st.markdown(f"<div class='insight-card'><div class='insight-label'>CHARGER TYPE</div><b>Highest average demand</b><div class='insight-body'>Code {type_means.index[0]} · {type_means.iloc[0]:.2f}</div></div>", unsafe_allow_html=True)
        if {"StartDatetime", "Demand"}.issubset(overview.columns):
            monthly_insight = valid_time_demand.set_index("StartDatetime")["Demand"].resample("ME").mean().dropna()
            if not monthly_insight.empty:
                st.markdown(f"<div class='insight-card'><div class='insight-label'>MONTHLY PATTERN</div><b>Peak month</b><div class='insight-body'>{monthly_insight.idxmax():%B %Y} · {monthly_insight.max():.2f}</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='insight-card'><div class='insight-label'>CONCLUSION</div><b>Plan with evidence</b><div class='insight-body'>Investigate recurring peaks alongside charger availability, location mix and operating constraints. Associations do not establish causes.</div></div>", unsafe_allow_html=True)


elif page == "Regression Prediction":
    st.header("Regression Prediction")
    st.write("Enter charging-session details to estimate demand.")

    with st.form("regression_form"):
        col1, col2 = st.columns(2)

        with col1:
            duration = st.number_input(
                "Duration (minutes)",
                min_value=1.0,
                value=60.0,
                step=5.0
            )

            location_options = sorted(data["Location"].dropna().unique())
            location = st.selectbox("Location", location_options)

            charger_company = st.selectbox(
                "Charger Company (encoded value)", [0, 1], index=1
            )

            charger_type = st.selectbox(
                "Charger Type (encoded value)", [0, 1], index=0
            )

            start_hour = st.selectbox(
                "Start Hour", list(range(24)), index=10
            )

        with col2:
            end_hour = st.selectbox(
                "End Hour", list(range(24)), index=11
            )

            day_of_week = st.selectbox(
                "Day of Week (encoded 0–6)", list(range(7)), index=3
            )

            month = st.selectbox(
                "Month", list(range(1, 13)), index=9
            )

            location_mean = data.loc[
                data["Location"] == location, "Demand"
            ].mean()

            historical_avg_demand = st.number_input(
                "Historical Average Demand",
                min_value=0.0,
                value=float(location_mean) if pd.notna(location_mean) else 0.0,
                step=0.5
            )

        submitted = st.form_submit_button("Predict Demand")

    if submitted:
        # DayOfWeek uses 0=Monday ... 5=Saturday, 6=Sunday.
        # Derive the weekend feature so the user does not enter the same fact twice.
        is_weekend = int(day_of_week >= 5)
        input_data = pd.DataFrame([{
            "Duration": duration,
            "Location": location,
            "ChargerCompany": charger_company,
            "ChargerType": charger_type,
            "StartHour": start_hour,
            "EndHour": end_hour,
            "DayOfWeek": day_of_week,
            "Month": month,
            "IsWeekend": is_weekend,
            "HistoricalAvgDemand": historical_avg_demand
        }])

        try:
            processed_input = regression_preprocessor.transform(input_data)
            prediction = regression_model.predict(processed_input)[0]

            
            st.subheader("Prediction result")
            result_col, chart_col = st.columns([1, 2], gap="large")
            with result_col:
                st.metric("Estimated demand", f"{prediction:.2f}")
                st.markdown(
                    f"<div class='insight-card'><div class='insight-label'>SCENARIO</div>"
                    f"<b>{location}</b><div class='insight-body'>{duration:.0f}-minute session · "
                    f"{start_hour:02d}:00 start</div></div>",
                    unsafe_allow_html=True
                )
            with chart_col:
                st.markdown("**Where does this estimate sit?**")
                location_demands = data.loc[data["Location"] == location, "Demand"].dropna()
                if not location_demands.empty:
                    prediction_fig = go.Figure()
                    prediction_fig.add_trace(go.Histogram(
                        x=location_demands,
                        nbinsx=18,
                        marker=dict(color="#E5C34F", line=dict(color="#FFFEFA", width=1)),
                        opacity=0.9,
                        name="Historical sessions",
                        hovertemplate="Historical demand: %{x:.2f}<br>Sessions in bin: %{y}<extra></extra>"
                    ))
                    prediction_fig.add_vline(
                        x=float(prediction), line_color="#9B7815", line_width=3,
                        line_dash="dash",
                        annotation_text=f"Prediction: {float(prediction):.2f}",
                        annotation_position="top"
                    )
                    prediction_fig.update_layout(
                        height=300, margin=dict(l=5, r=12, t=38, b=8),
                        paper_bgcolor="#FFFEFA", plot_bgcolor="#FFFEFA",
                        font=dict(family="Inter", size=11, color="#292821"),
                        xaxis_title="Demand value", yaxis_title="Number of sessions",
                        showlegend=False, bargap=0.08
                    )
                    prediction_fig.update_xaxes(showgrid=False, zeroline=False)
                    prediction_fig.update_yaxes(showgrid=False, zeroline=False)
                    st.plotly_chart(prediction_fig, use_container_width=True, key="regression_prediction_visual")
                    st.caption("Bars show the historical demand distribution for this location. The dashed line marks the model's estimate for your selected scenario.")
                else:
                    st.info("No historical demand values are available for this location to provide a distribution comparison.")
            st.write(
                f"For **{location}**, the model estimates demand at **{prediction:.2f}**. "
                "The historical distribution provides context; it is not a target or a guarantee of future demand."
            )

            st.markdown(
                """
                <div style="
                    background-color: #FFF5CC;
                    border-left: 5px solid #E0B000;
                    padding: 14px 18px;
                    border-radius: 6px;
                    color: #252525;
                ">
                    <strong>Interpretation</strong><br>
                    The practical meaning of this estimate depends on
                    the definition and unit of the Demand column.
                    Do not interpret it as low or high demand from this
                    number alone.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Input Summary")

            c1, c2, c3 = st.columns(3)
            c1.write(f"**Location:** {location}")
            c2.write(f"**Duration:** {duration:.0f} minutes")
            c3.write(f"**Start hour:** {start_hour}:00")

            c1.write(f"**Charger company code:** {charger_company}")
            c2.write(f"**Charger type code:** {charger_type}")
            c3.write(f"**Month:** {month}")

            st.caption(
                "This is a model estimate, not a guarantee of actual demand."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# --------------------------------------------------
# DEMAND CLASSIFICATION
# --------------------------------------------------

elif page == "Demand Classification":
    st.header("Demand Classification")
    st.write(
        "Predict whether a charging session belongs "
        "to the Low Demand or High Demand category."
    )

    with st.form("classification_form"):
        col1, col2 = st.columns(2)

        with col1:
            duration = st.number_input(
                "Duration (minutes)",
                min_value=1.0,
                value=60.0,
                step=5.0
            )

            location_options = sorted(
                data["Location"].dropna().unique()
            )
            location = st.selectbox(
                "Location",
                location_options,
                key="class_location"
            )

            charger_company = st.selectbox(
                "Charger Company (encoded value)",
                [0, 1],
                key="class_company"
            )

            charger_type = st.selectbox(
                "Charger Type (encoded value)",
                [0, 1],
                key="class_type"
            )

            start_hour = st.selectbox(
                "Start Hour",
                list(range(24)),
                index=10,
                key="class_start"
            )

        with col2:
            end_hour = st.selectbox(
                "End Hour",
                list(range(24)),
                index=11,
                key="class_end"
            )

            day_of_week = st.selectbox(
                "Day of Week (0–6)",
                list(range(7)),
                index=3,
                key="class_day"
            )

            month = st.selectbox(
                "Month",
                list(range(1, 13)),
                index=9,
                key="class_month"
            )

            location_mean = data.loc[
                data["Location"] == location, "Demand"
            ].mean()

            historical_avg_demand = st.number_input(
                "Historical Average Demand",
                min_value=0.0,
                value=(
                    float(location_mean)
                    if pd.notna(location_mean)
                    else 0.0
                ),
                step=0.5,
                key="class_history"
            )

        submitted = st.form_submit_button(
            "Classify Demand"
        )

    if submitted:
        # DayOfWeek uses 0=Monday ... 5=Saturday, 6=Sunday.
        is_weekend = int(day_of_week >= 5)
        input_data = pd.DataFrame([{
            "Duration": duration,
            "Location": location,
            "ChargerCompany": charger_company,
            "ChargerType": charger_type,
            "StartHour": start_hour,
            "EndHour": end_hour,
            "DayOfWeek": day_of_week,
            "Month": month,
            "IsWeekend": is_weekend,
            "HistoricalAvgDemand": historical_avg_demand
        }])

        try:
            processed_input = (
                classification_preprocessor.transform(input_data)
            )

            prediction = classifier.predict(processed_input)[0]

            probabilities = classifier.predict_proba(
                processed_input
            )[0]

            predicted_class = (
                "High Demand" if prediction == 1
                else "Low Demand"
            )

            confidence = float(max(probabilities)) * 100

            st.divider()
            st.subheader("Classification Result")

            st.markdown(
                f"""
                <div style="
                    background-color: #FFF5CC;
                    border-left: 5px solid #E0B000;
                    padding: 18px;
                    border-radius: 8px;
                    color: #252525;
                ">
                    <h3 style="margin: 0;">
                        Predicted Category: {predicted_class}
                    </h3>
                    <p style="margin-bottom: 0;">
                        Model confidence in the predicted class:
                        {confidence:.2f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("Class Probabilities")

            low_probability = float(probabilities[0]) * 100
            high_probability = float(probabilities[1]) * 100

            prob_col1, prob_col2 = st.columns(2)

            with prob_col1:
                st.metric(
                    "Low Demand",
                    f"{low_probability:.2f}%"
                )

            with prob_col2:
                st.metric(
                    "High Demand",
                    f"{high_probability:.2f}%"
                )

            probability_fig = go.Figure(go.Pie(
                labels=["Low Demand", "High Demand"],
                values=[low_probability, high_probability],
                hole=0.48,
                sort=False,
                domain=dict(x=[0.04, 0.76], y=[0.00, 0.88]),
                marker=dict(colors=["#E5C34F", "#9B7815"], line=dict(color="#FFFEFA", width=3)),
                text=[f"{low_probability:.1f}%", f"{high_probability:.1f}%"],
                textinfo="percent",
                textposition="inside",
                insidetextorientation="horizontal",
                hovertemplate="%{label}<br>Estimated probability: %{value:.2f}%<extra></extra>"
            ))
            probability_fig.update_layout(
                height=310, margin=dict(l=8, r=8, t=22, b=5),
                paper_bgcolor="#FFFEFA", plot_bgcolor="#FFFEFA",
                font=dict(family="Inter", size=12, color="#292821"),
                showlegend=True,
                legend=dict(orientation="v", x=0.78, y=0.42, xanchor="left", yanchor="middle")
            )
            st.plotly_chart(probability_fig, use_container_width=True, key="classification_probability_chart")
            st.caption(
                "Probabilities are model estimates, not guarantees of actual demand. "
                "Evaluation metrics such as precision, recall and F1 should be reported "
                "from a separate held-out test set; they are not inferred from this single prediction."
            )

        except Exception as error:
            st.error(
                f"Classification failed: {error}"
            )


# --------------------------------------------------
# TIME-SERIES FORECASTING — EXECUTIVE VIEW
# --------------------------------------------------

elif page == "Time-Series Forecasting":
    st.title("EV Demand | Executive Outlook")
    st.caption("Decision support for demand planning • Historical trend • Forecast outlook • Model accuracy")

    try:
        ts_data = data.copy()
        ts_data["StartDatetime"] = pd.to_datetime(ts_data["StartDatetime"], errors="coerce")
        ts_data["Demand"] = pd.to_numeric(ts_data["Demand"], errors="coerce")
        ts_data = ts_data.dropna(subset=["StartDatetime", "Demand"]).sort_values("StartDatetime")
        monthly_demand = ts_data.set_index("StartDatetime")["Demand"].resample("ME").mean().dropna()

        if len(monthly_demand) < 8:
            st.warning("At least eight months of valid demand data are recommended for this forecast view.")
        else:
            monthly_demand = monthly_demand.astype(float)
            monthly_df = monthly_demand.rename("Actual Demand").to_frame()
            monthly_df["3-Month Moving Average"] = monthly_df["Actual Demand"].rolling(3).mean()

            # A chronological holdout: last four months, with one-step-ahead forecasts.
            test_count = min(4, max(2, len(monthly_demand) // 4))
            train_series = monthly_demand.iloc[:-test_count]
            test_series = monthly_demand.iloc[-test_count:]

            ma_history = list(train_series.values)
            ma_test_values = []
            for actual in test_series.values:
                pred = float(np.mean(ma_history[-3:]))
                ma_test_values.append(pred)
                ma_history.append(float(actual))
            ma_test = pd.Series(ma_test_values, index=test_series.index, name="Moving Average")

            # Exponential smoothing evaluation. If statsmodels is unavailable, use a
            # transparent simple exponential-smoothing fallback with alpha=0.30.
            es_test_values = []
            es_method_label = "Simple Exponential Smoothing"
            try:
                from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                es_history = train_series.copy()
                for date in test_series.index:
                    fitted = SimpleExpSmoothing(
                        es_history, initialization_method="estimated"
                    ).fit(optimized=True)
                    es_test_values.append(float(fitted.forecast(1).iloc[0]))
                    es_history = pd.concat([es_history, test_series.loc[[date]]])
            except Exception:
                alpha = 0.30
                level = float(train_series.iloc[0])
                for value in train_series.iloc[1:]:
                    level = alpha * float(value) + (1 - alpha) * level
                for actual in test_series.values:
                    es_test_values.append(level)
                    level = alpha * float(actual) + (1 - alpha) * level
                es_method_label = "Exponential Smoothing (α = 0.30 fallback)"
            es_test = pd.Series(es_test_values, index=test_series.index, name="Exponential Smoothing")

            def forecast_metrics(actual, predicted):
                actual_arr = np.asarray(actual, dtype=float)
                pred_arr = np.asarray(predicted, dtype=float)
                return float(np.mean(np.abs(actual_arr - pred_arr))), float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))

            ma_mae, ma_rmse = forecast_metrics(test_series, ma_test)
            es_mae, es_rmse = forecast_metrics(test_series, es_test)
            best_method = "3-Month Moving Average" if ma_mae <= es_mae else es_method_label
            best_mae = min(ma_mae, es_mae)

            # Refit/forecast the next three months using the two baseline methods.
            last_date = monthly_demand.index[-1]
            future_dates = pd.date_range(last_date + pd.offsets.MonthEnd(1), periods=3, freq="ME")
            ma_history = list(monthly_demand.values)
            future_ma = []
            for _ in future_dates:
                pred = float(np.mean(ma_history[-3:]))
                future_ma.append(pred)
                ma_history.append(pred)

            # Fit simple exponential smoothing on all observed history for future outlook.
            try:
                from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                fitted_all = SimpleExpSmoothing(
                    monthly_demand, initialization_method="estimated"
                ).fit(optimized=True)
                future_es = np.asarray(fitted_all.forecast(3), dtype=float)
                future_es_label = "Exponential Smoothing"
            except Exception:
                alpha = 0.30
                level = float(monthly_demand.iloc[0])
                for value in monthly_demand.iloc[1:]:
                    level = alpha * float(value) + (1 - alpha) * level
                future_es = np.repeat(level, 3)
                future_es_label = "Exponential Smoothing (α = 0.30 fallback)"

            forecast_df = pd.DataFrame({
                "Moving Average": future_ma,
                future_es_label: future_es
            }, index=future_dates)

            latest = float(monthly_demand.iloc[-1])
            previous = float(monthly_demand.iloc[-2])
            mom_change = latest - previous
            mom_pct = (mom_change / abs(previous) * 100) if previous != 0 else 0.0
            next_forecast = float(forecast_df["Moving Average"].iloc[0] if best_method == "3-Month Moving Average" else forecast_df[future_es_label].iloc[0])

            # Executive summary KPIs
            st.markdown("### Executive summary")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Latest monthly demand", f"{latest:.2f}", f"{mom_change:+.2f} vs prior month")
            k2.metric("Month-on-month change", f"{mom_pct:+.1f}%")
            k3.metric("Next-month forecast", f"{next_forecast:.2f}", best_method)
            k4.metric("Lowest holdout MAE", f"{best_mae:.3f}", best_method)
            st.caption(
                f"Analysis covers {len(monthly_demand)} monthly observations. Forecast values use the same demand units as the source dataset. "
                f"Lowest holdout MAE is the smallest average absolute forecast error on the held-out months: {best_mae:.3f} demand units, "
                f"using {best_method}. Lower is better; it is not a forecast value."
            )

            # Historical performance and executive interpretation
            st.divider()
            hist_col, insight_col = st.columns([1.8, 1], gap="large")
            with hist_col:
                st.markdown("### 01 · Demand trajectory")
                # Use explicit month labels as categorical x-values. This prevents Plotly's
                # date-axis tick placement from showing a neighboring month label.
                history_month_labels = monthly_df.index.strftime("%b %Y")
                fig_history = go.Figure()
                fig_history.add_trace(go.Scatter(
                    x=history_month_labels, y=monthly_df["Actual Demand"],
                    mode="lines+markers", name="Actual demand",
                    line=dict(color="#263746", width=3), marker=dict(size=6),
                    customdata=history_month_labels,
                    hovertemplate="Month: %{customdata}<br>Actual demand: %{y:.3f}<extra></extra>"
                ))
                fig_history.add_trace(go.Scatter(
                    x=history_month_labels, y=monthly_df["3-Month Moving Average"],
                    mode="lines+markers", name="3-month rolling average",
                    line=dict(color="#D9A21B", width=3, dash="solid"), marker=dict(size=5),
                    customdata=history_month_labels,
                    hovertemplate="Month: %{customdata}<br>3-month rolling average: %{y:.3f}<extra></extra>"
                ))
                fig_history.update_layout(
                    height=340, plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(color="#263746"), hovermode="x unified",
                    xaxis=dict(title="Month", type="category", showgrid=False, showline=True, linecolor="#D8DEE5"),
                    yaxis=dict(title="Average demand", showgrid=False, zeroline=False, showline=True, linecolor="#D8DEE5"),
                    legend=dict(orientation="h", y=1.12, x=0),
                    margin=dict(l=8, r=8, t=35, b=8)
                )
                st.plotly_chart(fig_history, use_container_width=True, key="exec_ts_history")
            with insight_col:
                st.markdown("### Executive readout")
                direction = "increased" if mom_change > 0 else "decreased" if mom_change < 0 else "was unchanged"
                st.markdown(
                    f"**Latest movement**\n\nMonthly average demand {direction} by **{abs(mom_change):.2f} ({mom_pct:+.1f}%)** compared with the previous month."
                )
                st.markdown(
                    f"**Model signal**\n\nThe lower-error baseline on the recent holdout was **{best_method}** (MAE **{best_mae:.3f}**). This is a provisional selection, not a guarantee of future accuracy."
                )
                st.markdown(
                    "**Planning implication**\n\nUse the outlook as an initial input for staffing and charging-capacity discussions. Validate against station capacity, charger uptime, occupancy and waiting-time data before making operational commitments."
                )

            # Forecast outlook
            st.divider()
            st.markdown("### 02 · Three-month demand outlook")
            st.caption("Historical actuals are shown separately from future estimates. Forecasts are baseline scenarios, not certainty ranges.")
            fig_forecast = go.Figure()
            recent_history = monthly_demand.iloc[-min(8, len(monthly_demand)):]
            # Keep historical and future months on one categorical axis so labels and
            # hover values always refer to the same month.
            forecast_month_labels = list(recent_history.index.strftime("%b %Y")) + list(forecast_df.index.strftime("%b %Y"))
            history_labels = list(recent_history.index.strftime("%b %Y"))
            future_labels = list(forecast_df.index.strftime("%b %Y"))
            fig_forecast.add_trace(go.Scatter(
                x=history_labels, y=recent_history.values,
                mode="lines+markers", name="Historical actual",
                line=dict(color="#263746", width=3), marker=dict(size=6),
                customdata=history_labels,
                hovertemplate="Month: %{customdata}<br>Historical actual: %{y:.3f}<extra></extra>"
            ))
            fig_forecast.add_trace(go.Scatter(
                x=future_labels, y=forecast_df["Moving Average"],
                mode="lines+markers", name="Moving Average forecast",
                line=dict(color="#D9A21B", width=3, dash="dash"), marker=dict(size=7),
                customdata=future_labels,
                hovertemplate="Month: %{customdata}<br>Moving Average forecast: %{y:.3f}<extra></extra>"
            ))
            fig_forecast.add_trace(go.Scatter(
                x=future_labels, y=forecast_df[future_es_label],
                mode="lines+markers", name=future_es_label,
                line=dict(color="#287D8E", width=3, dash="dot"), marker=dict(size=7),
                customdata=future_labels,
                hovertemplate="Month: %{customdata}<br>Exponential Smoothing forecast: %{y:.3f}<extra></extra>"
            ))
            fig_forecast.update_layout(
                height=360, plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color="#263746"), hovermode="x unified",
                xaxis=dict(title="Month", type="category", categoryorder="array", categoryarray=forecast_month_labels, showgrid=False, showline=True, linecolor="#D8DEE5"),
                yaxis=dict(title="Forecast demand", showgrid=False, zeroline=False, showline=True, linecolor="#D8DEE5"),
                legend=dict(orientation="h", y=1.12, x=0),
                margin=dict(l=8, r=8, t=35, b=8)
            )
            st.plotly_chart(fig_forecast, use_container_width=True, key="exec_ts_forecast")

            forecast_display = forecast_df.copy()
            forecast_display.index = forecast_display.index.strftime("%b %Y")
            forecast_display.index.name = "Forecast month"
            st.dataframe(forecast_display.round(3), use_container_width=True)

            # Model scorecard on the same chronological holdout
            st.divider()
            st.markdown("### 03 · Forecast model scorecard")
            scorecard = pd.DataFrame({
                "Model": ["3-Month Moving Average", es_method_label],
                "MAE": [ma_mae, es_mae],
                "RMSE": [ma_rmse, es_rmse],
                "Test observations": [len(test_series), len(test_series)]
            }).sort_values("MAE").reset_index(drop=True)
            scorecard["MAE"] = scorecard["MAE"].round(3)
            scorecard["RMSE"] = scorecard["RMSE"].round(3)
            st.dataframe(scorecard, use_container_width=True, hide_index=True)

            score_col, methodology_col = st.columns([1, 1], gap="large")
            with score_col:
                st.markdown("**Holdout actual vs forecast**")
                test_compare = pd.DataFrame({
                    "Actual demand": test_series,
                    "Moving Average": ma_test,
                    es_method_label: es_test
                })
                test_compare.index = test_compare.index.strftime("%b %Y")
                test_compare.index.name = "Test month"
                st.dataframe(test_compare.round(3), use_container_width=True)
            with methodology_col:
                st.markdown("**How to read this scorecard**")
                st.markdown(
                    "- **MAE** is the average absolute forecast error.\n"
                    "- **RMSE** penalizes larger errors more strongly.\n"
                    "- Lower scores indicate better performance on these held-out months.\n"
                    f"- Evaluation uses the final **{len(test_series)} months** and rolling one-step-ahead forecasts; actual demand from each test month is added before forecasting the next."
                )
                st.warning("The test window is short. Treat the winning method as a baseline and re-check it with more history before relying on it for investment or capacity decisions.")

    except Exception as error:
        st.error(f"Time-series analysis failed: {error}")
