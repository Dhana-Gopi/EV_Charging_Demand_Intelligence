from pathlib import Path
import joblib

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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

# Shared chart helper — available to every page

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
# --------------------------------------------------
# ALIGNED EXECUTIVE HEADER + NAVIGATION
# --------------------------------------------------


page = st.radio(
        "Dashboard navigation",
        [
            "Executive Overview",
            "Location-Wise Demand Intelligence",
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

    
    def chart_note(high_text, low_text, meaning):
        st.caption(f"**High:** {high_text}  ·  **Low:** {low_text}")
        st.caption(meaning)

        # Precompute aggregates used in charts and narratives
    loc_means = overview.groupby("Location")["Demand"].mean().dropna().sort_values(ascending=False) if {"Location", "Demand"}.issubset(overview.columns) else pd.Series(dtype=float)
    company_means = overview.groupby("ChargerCompany")["Demand"].mean().dropna().sort_values(ascending=False) if {"ChargerCompany", "Demand"}.issubset(overview.columns) else pd.Series(dtype=float)
    type_means = overview.groupby("ChargerType")["Demand"].mean().dropna().sort_values(ascending=False) if {"ChargerType", "Demand"}.issubset(overview.columns) else pd.Series(dtype=float)

    with chart_col:
        st.markdown("### Usage patterns & demand")
        row1a, row1b = st.columns(2, gap="small")
        with row1a:
            # Demand by location: allocate enough chart height for all categories and keep every label visible
            if not loc_means.empty:
                loc_plot = loc_means.sort_values(ascending=True)
                fig = go.Figure(go.Bar(
                        x=loc_plot.values, y=loc_plot.index.astype(str), orientation="h",
                        marker=dict(color=loc_plot.values, colorscale=[[0, "#F3E6B1"], [1, "#9B7815"]]),
                        text=[f"{v:.2f}" for v in loc_plot.values], textposition="outside", cliponaxis=False,
                        hovertemplate="Location: %{y}<br>Average demand: %{x:.2f}<extra></extra>"
                    ))
                fig.update_layout(xaxis_title="Average demand", yaxis_title="", margin=dict(l=150, r=55, t=16, b=35))
                fig.update_yaxes(
                        tickmode="array", tickvals=loc_plot.index.astype(str).tolist(),
                        ticktext=loc_plot.index.astype(str).tolist(), tickfont=dict(size=10),
                        automargin=True, categoryorder="array", categoryarray=loc_plot.index.astype(str).tolist()
                    )
                location_fig = compact_layout(fig, max(280, min(300, 15 * len(loc_plot))))
                location_fig.update_layout(margin=dict(l=150, r=55, t=16, b=35))
                st.plotly_chart(location_fig, use_container_width=True, key="overview_location")
                chart_note(f"{loc_means.index[0]} ({loc_means.iloc[0]:.2f})", f"{loc_means.index[-1]} ({loc_means.iloc[-1]:.2f})", "Bars show mean demand, not total sessions. The location names and all bars are shown so high- and low-demand sites can be compared directly.")
            else:
                st.info("Requires Location and Demand columns.")

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
                company_data = overview.dropna(
                    subset=["ChargerCompany", "Demand"]
                ).copy()
        
                company_demand = (
                    company_data.groupby("ChargerCompany")["Demand"]
                    .mean()
                    .reindex([0, 1])
                    .dropna()
                )
        
                total_demand = company_demand.sum()
        
                if not company_demand.empty and total_demand > 0:
                    company_pct = (company_demand / total_demand) * 100
        
                    fig = go.Figure()
        
                    for company, pct in company_pct.items():
                        x = int(company)
        
                        shaft_width = 0.22
                        head_width = 0.65
                        head_height = 8
                        base = 0
                        tip = float(pct)
                        head_base = max(base, tip - head_height)
        
                        # Solid arrow-shaped bar
                        fig.add_shape(
                            type="path",
                            xref="x",
                            yref="y",
                            path=(
                                f"M {x - shaft_width/2},{base} "
                                f"L {x + shaft_width/2},{base} "
                                f"L {x + shaft_width/2},{head_base} "
                                f"L {x + head_width/2},{head_base} "
                                f"L {x},{tip} "
                                f"L {x - head_width/2},{head_base} "
                                f"L {x - shaft_width/2},{head_base} Z"
                            ),
                            fillcolor="#C9A227",
                            line=dict(color="#C9A227", width=1),
                        )
        
                        # Percentage above the arrow
                        fig.add_annotation(
                            x=x,
                            y=tip + 4,
                            text=f"{pct:.1f}%",
                            showarrow=False,
                            font=dict(size=13, color="#9B7815"),
                        )
        
                    fig.update_layout(
                        xaxis=dict(
                            title="Charger Company",
                            tickmode="array",
                            tickvals=[0, 1],
                            ticktext=["0", "1"],
                            range=[-0.7, 1.7],
                            fixedrange=True,
                        ),
                        yaxis=dict(
                            title="Share of combined average demand (%)",
                            range=[0, 75],
                            dtick=15,
                            ticksuffix="%",
                            fixedrange=True,
                        ),
                        margin=dict(l=15, r=20, t=35, b=45),
                        showlegend=False,
                    )
        
                    st.plotly_chart(
                        compact_layout(fig, 300),
                        use_container_width=True,
                        key="overview_company",
                    )
        
                else:
                    st.info("No valid charger company demand data available.")
            else:
                st.info("Required columns ChargerCompany and Demand are missing.")
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
            st.subheader("Calendar Heatmap · Weekday vs Weekend Demand")
        if not valid_time_demand.empty:
            # Pool available years: rows are Monday–Sunday and columns are Jan–Dec.
            calendar_data = valid_time_demand.copy()
            calendar_data["MonthNumber"] = calendar_data["StartDatetime"].dt.month
            calendar_data["WeekdayNumber"] = calendar_data["StartDatetime"].dt.dayofweek
            month_names = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            demand_grid = calendar_data.pivot_table(
                index="WeekdayNumber", columns="MonthNumber", values="Demand", aggfunc="mean"
            ).reindex(index=range(7), columns=range(1, 13))
            record_grid = calendar_data.pivot_table(
                index="WeekdayNumber", columns="MonthNumber", values="Demand", aggfunc="count"
            ).reindex(index=range(7), columns=range(1, 13))
            tile_text = np.empty(demand_grid.shape, dtype=object)
            for row_i in range(demand_grid.shape[0]):
                for col_i in range(demand_grid.shape[1]):
                    value = demand_grid.iloc[row_i, col_i]
                    tile_text[row_i, col_i] = f"{value:.1f}" if pd.notna(value) else "—"
            fig = go.Figure(go.Heatmap(
                z=demand_grid.to_numpy(dtype=float),
                x=month_names,
                y=weekday_names,
                text=tile_text,
                customdata=record_grid.to_numpy(dtype=float),
                texttemplate="%{text}",
                textfont=dict(size=9, color="#292821"),
                colorscale=[[0, "#FFF8DF"], [0.5, "#F0D56C"], [1, "#9B7815"]],
                colorbar=dict(title="Avg demand", thickness=8),
                xgap=4, ygap=4,
                hovertemplate="Day: %{y}<br>Month: %{x}<br>Average demand: %{z:.2f}<br>Records: %{customdata:.0f}<extra></extra>"
            ))
            fig.update_layout(
                xaxis=dict(side="top", tickangle=-35, title="Month", tickfont=dict(size=8)),
                yaxis=dict(title="Day of week", autorange="reversed", tickfont=dict(size=9)),
                margin=dict(l=85, r=12, t=78, b=18)
            )
            calendar_fig = compact_layout(fig, 330)
            calendar_fig.update_layout(margin=dict(l=85, r=12, t=78, b=18))
            st.plotly_chart(calendar_fig, use_container_width=True, key="overview_week_calendar")
            weekday_mean = calendar_data.loc[calendar_data["WeekdayNumber"] < 5, "Demand"].mean()
            weekend_mean = calendar_data.loc[calendar_data["WeekdayNumber"] >= 5, "Demand"].mean()
            if pd.notna(weekday_mean) and pd.notna(weekend_mean):
                higher = "Weekday" if weekday_mean >= weekend_mean else "Weekend"
                lower = "Weekend" if higher == "Weekday" else "Weekday"
                chart_note(
                    f"{higher} average ({max(weekday_mean, weekend_mean):.2f})",
                    f"{lower} average ({min(weekday_mean, weekend_mean):.2f})",
                    "Each cell shows average demand for that weekday and month across all available years. Saturday and Sunday are weekend rows; hover to see the number of records behind a cell."
                )
        else:
            st.info("Requires StartDatetime and Demand columns.")

        
    row4a, row4b = st.columns(2, gap="small")

    with row4a:
        st.subheader("Demand by Charging Duration · High to Low")
    
        if {"Duration", "Demand"}.issubset(overview.columns):
            dd = overview.dropna(subset=["Duration", "Demand"]).copy()
            dd = dd[dd["Duration"] >= 0]
    
            dd["Duration Band"] = pd.cut(
                dd["Duration"],
                bins=[-0.001, 30, 60, 120, 240, np.inf],
                labels=[
                    "≤30 min", "31–60 min", "61–120 min",
                    "121–240 min", ">240 min"
                ]
            )
    
            duration_means = (
                dd.groupby("Duration Band", observed=False)["Demand"]
                .mean()
                .dropna()
                .sort_values(ascending=False)
            )
    
            if not duration_means.empty:
                fig = go.Figure(
                    go.Bar(
                        x=duration_means.values,
                        y=duration_means.index.astype(str),
                        orientation="h",
                        marker=dict(
                            color=duration_means.values,
                            colorscale=[
                                [0, "#F3E6B1"],
                                [1, "#9B7815"]
                            ]
                        ),
                        text=[f"{v:.2f}" for v in duration_means.values],
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate=(
                            "Duration: %{y}<br>"
                            "Average demand: %{x:.2f}<extra></extra>"
                        )
                    )
                )
    
                fig.update_layout(
                    xaxis_title="Average demand",
                    yaxis_title="",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=12, r=45, t=12, b=35)
                )
    
                st.plotly_chart(
                    compact_layout(fig, 300),
                    use_container_width=True,
                    key="overview_duration"
                )
    
                st.caption(
                    "Ranked from highest to lowest average demand. "
                    "Association does not establish causation."
                )
            else:
                st.info("No valid duration values are available.")
        else:
            st.info("Requires Duration and Demand columns.")
    
    
    with row4b:
        st.subheader("Hourly Demand Profile")
    
        if not hourly.dropna().empty:
            fig = go.Figure(
                go.Bar(
                    x=hourly.index,
                    y=hourly.values,
                    marker_color=[
                        "#9B7815" if h == hourly.idxmax() else "#E5C34F"
                        for h in hourly.index
                    ],
                    hovertemplate=(
                        "%{x}:00<br>"
                        "Average demand: %{y:.2f}<extra></extra>"
                    )
                )
            )
    
            fig.update_layout(
                xaxis=dict(title="Hour", dtick=4),
                yaxis_title="Average demand",
                margin=dict(l=12, r=20, t=12, b=35)
            )
    
            st.plotly_chart(
                compact_layout(fig, 300),
                use_container_width=True,
                key="overview_hourly"
            )
    
            low_hour = hourly.idxmin()
    
            chart_note(
                f"{int(hourly.idxmax()):02d}:00 ({hourly.max():.2f})",
                f"{int(low_hour):02d}:00 ({hourly.min():.2f})",
                "Use peak and low-demand hours to identify periods for "
                "operational review."
            )
        else:
            st.info("Requires StartDatetime and Demand columns.")

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




elif page == "Location-Wise Demand Intelligence":

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    # ---------------------------------------------------------
    # 1. PAGE TITLE AND THEME
    # ---------------------------------------------------------
    st.title("Location-Wise Demand Intelligence")
    st.caption(
        "Explore charging demand by location, charger equipment, "
        "time of day, charging duration and weekday/weekend patterns."
    )

    BG_COLOR = "#FFFEFA"
    TEXT_COLOR = "#292821"
    GOLD = "#C9A227"
    DARK_GOLD = "#9B7815"
    SAGE = "#8C9F91"
    LIGHT_GOLD = "#E9D9A5"

    def style_location_chart(fig, height=300):
        fig.update_layout(
            height=height,
            margin=dict(l=10, r=10, t=35, b=10),
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            font=dict(
                family="Inter",
                size=11,
                color=TEXT_COLOR
            ),
            title_font=dict(
                family="Inter",
                size=13,
                color=TEXT_COLOR
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=11,
                font_family="Inter"
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=10)
            )
        )

        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            automargin=True
        )

        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            automargin=True
        )

        return fig

    def display_code(value):
        """Display numeric company/type codes without trailing .0."""
        if pd.isna(value):
            return "Unknown"

        try:
            numeric_value = float(value)
            if numeric_value.is_integer():
                return str(int(numeric_value))
        except (ValueError, TypeError):
            pass

        return str(value)

    # ---------------------------------------------------------
    # 2. CHECK AND PREPARE DATA
    # ---------------------------------------------------------
    if "data" not in locals() and "data" not in globals():
        st.error(
            "The main dataset 'data' was not found. "
            "Check the variable name used when loading your dataset."
        )
        st.stop()

    required_columns = [
        "Location",
        "Demand",
        "ChargerCompany",
        "ChargerType",
        "StartDatetime"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    # Support either common name for the duration column.
    if "Duration" in data.columns:
        duration_column = "Duration"
    elif "Duration(min)" in data.columns:
        duration_column = "Duration(min)"
    else:
        duration_column = None

    if duration_column is None:
        missing_columns.append("Duration or Duration(min)")

    if missing_columns:
        st.error(
            "Required dataset columns are missing: "
            + ", ".join(missing_columns)
        )
        st.stop()

    intelligence = data.copy()

    intelligence["Demand"] = pd.to_numeric(
        intelligence["Demand"], errors="coerce"
    )

    intelligence["Duration"] = pd.to_numeric(
        intelligence[duration_column], errors="coerce"
    )

    intelligence["StartDatetime"] = pd.to_datetime(
        intelligence["StartDatetime"], errors="coerce"
    )

    intelligence = intelligence.dropna(
        subset=[
            "Location",
            "Demand",
            "Duration",
            "ChargerCompany",
            "ChargerType",
            "StartDatetime"
        ]
    ).copy()

    intelligence = intelligence[
        intelligence["Duration"] >= 0
    ].copy()

    intelligence["StartHour"] = (
        intelligence["StartDatetime"].dt.hour
    )

    intelligence["Month"] = (
        intelligence["StartDatetime"].dt.to_period("M")
    )

    intelligence["DayGroup"] = np.where(
        intelligence["StartDatetime"].dt.dayofweek >= 5,
        "Weekend",
        "Weekday"
    )

    if intelligence.empty:
        st.warning("No valid records are available for this analysis.")
        st.stop()

    all_months = pd.period_range(
        intelligence["Month"].min(),
        intelligence["Month"].max(),
        freq="M"
    )

    # ---------------------------------------------------------
    # 3. ANALYSIS MODE AND FILTERS
    # ---------------------------------------------------------
    st.markdown("### Choose your analysis")

    analysis_mode = st.radio(
        "Analysis mode",
        [
            "Location Analysis",
            "Charger Equipment Analysis",
            "Demand-Range Analysis"
        ],
        horizontal=True,
        key="location_analysis_mode"
    )

    selected_location = None
    selected_equipment = None
    selected_equipment_value = None

    # ---------------- LOCATION FILTER ----------------
    if analysis_mode == "Location Analysis":

        locations = sorted(
            intelligence["Location"].astype(str).unique().tolist()
        )

        location_options = ["Select a location..."] + locations

        selected_location = st.selectbox(
            "Select a location",
            location_options,
            index=0,
            key="location_page_location"
        )

        if selected_location == "Select a location...":
            st.info(
                "Choose a location to view its charger combinations, "
                "hourly demand, duration patterns and monthly trends."
            )
            st.stop()

        location_data = intelligence[
            intelligence["Location"].astype(str) == selected_location
        ].copy()

    # ---------------- EQUIPMENT FILTER ----------------
    elif analysis_mode == "Charger Equipment Analysis":

        equipment_choice = st.selectbox(
            "Filter by equipment",
            ["Charger Type", "Charger Company"],
            key="location_page_equipment_choice"
        )

        if equipment_choice == "Charger Type":
            selected_equipment = "ChargerType"
            equipment_label = "Charger type"
        else:
            selected_equipment = "ChargerCompany"
            equipment_label = "Charger company"

        available_values = sorted(
            intelligence[selected_equipment].dropna().unique().tolist(),
            key=lambda value: display_code(value)
        )

        equipment_options = ["All"] + available_values

        selected_equipment_value = st.selectbox(
            f"Select {equipment_label}",
            equipment_options,
            format_func=lambda value: (
                "All" if value == "All"
                else display_code(value)
            ),
            key="location_page_equipment_value"
        )

        location_data = intelligence.copy()

        if selected_equipment_value != "All":
            location_data = location_data[
                location_data[selected_equipment]
                == selected_equipment_value
            ].copy()

        # Optional additional equipment filter.
        other_column = (
            "ChargerCompany"
            if selected_equipment == "ChargerType"
            else "ChargerType"
        )

        other_label = (
            "Charger company"
            if other_column == "ChargerCompany"
            else "Charger type"
        )

        other_values = sorted(
            location_data[other_column].dropna().unique().tolist(),
            key=lambda value: display_code(value)
        )

        if other_values:
            other_options = ["All"] + other_values

            selected_other_value = st.selectbox(
                f"Optional {other_label} filter",
                other_options,
                format_func=lambda value: (
                    "All" if value == "All"
                    else display_code(value)
                ),
                key="location_page_other_equipment"
            )

            if selected_other_value != "All":
                location_data = location_data[
                    location_data[other_column] == selected_other_value
                ].copy()

    # ---------------- DEMAND RANGE FILTER ----------------
    else:

        demand_min = float(intelligence["Demand"].min())
        demand_max = float(intelligence["Demand"].max())

        if demand_min == demand_max:
            st.info(
                f"All valid records have the same demand value: "
                f"{demand_min:.2f}."
            )
            demand_range = (demand_min, demand_max)
        else:
            demand_range = st.slider(
                "Select demand range",
                min_value=demand_min,
                max_value=demand_max,
                value=(demand_min, demand_max),
                step=max((demand_max - demand_min) / 100, 0.01),
                key="location_page_demand_range"
            )

        location_filter_options = ["All"] + sorted(
            intelligence["Location"].astype(str).unique().tolist()
        )

        selected_location_filter = st.selectbox(
            "Optional location filter",
            location_filter_options,
            key="location_page_demand_location"
        )
        
        location_data = intelligence[
            intelligence["Demand"].between(
                demand_range[0], demand_range[1]
            )
        ].copy()

        if selected_location_filter != "All":
            location_data = location_data[
                location_data["Location"].astype(str)
                == selected_location_filter
            ].copy()

    # ---------------------------------------------------------
    # 4. HANDLE EMPTY FILTER RESULTS
    # ---------------------------------------------------------
    if location_data.empty:
        st.warning(
            "No records match these filters. Try another location, "
            "equipment selection or demand range."
        )
        st.stop()

    # ---------------------------------------------------------
    # 5. FILTER SUMMARY
    # ---------------------------------------------------------
    total_records = len(location_data)
    total_locations = location_data["Location"].nunique()
    overall_average = location_data["Demand"].mean()

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    summary_col1.metric(
        "Records analysed",
        f"{total_records:,}"
    )

    summary_col2.metric(
        "Locations covered",
        f"{total_locations:,}"
    )

    summary_col3.metric(
        "Average demand",
        f"{overall_average:.2f}"
    )

    st.caption(
        "Average demand is the mean of the Demand values in the "
        "filtered records. It should not automatically be interpreted "
        "as energy consumption, revenue or charger occupancy."
    )

    st.divider()

    # ---------------------------------------------------------
    # 6. DEMAND BY CHARGER COMPANY AND CHARGER TYPE
    # ---------------------------------------------------------
    st.markdown("### Charger Equipment Comparison")

    company_col, type_col = st.columns(2)

    with company_col:

        st.markdown("**Average demand by charger company**")

        company_summary = (
            location_data.groupby("ChargerCompany")["Demand"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )

        company_summary["Label"] = company_summary[
            "ChargerCompany"
        ].apply(
            lambda value: f"Company {display_code(value)}"
        )

        fig_company = px.bar(
            company_summary,
            x="Label",
            y="mean",
            text="mean",
            custom_data=["count"],
            color_discrete_sequence=[GOLD]
        )

        fig_company.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{x}<br>"
                "Average demand: %{y:.2f}<br>"
                "Records: %{customdata[0]:,}"
                "<extra></extra>"
            )
        )

        fig_company.update_layout(
            xaxis_title="Charger company code",
            yaxis_title="Average demand"
        )

        style_location_chart(fig_company, 280)

        st.plotly_chart(
            fig_company,
            use_container_width=True,
            key="location_company_comparison"
        )

    with type_col:

        st.markdown("**Average demand by charger type**")

        type_summary = (
            location_data.groupby("ChargerType")["Demand"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )

        type_summary["Label"] = type_summary[
            "ChargerType"
        ].apply(
            lambda value: f"Type {display_code(value)}"
        )

        fig_type = px.bar(
            type_summary,
            x="Label",
            y="mean",
            text="mean",
            custom_data=["count"],
            color_discrete_sequence=[SAGE]
        )

        fig_type.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{x}<br>"
                "Average demand: %{y:.2f}<br>"
                "Records: %{customdata[0]:,}"
                "<extra></extra>"
            )
        )

        fig_type.update_layout(
            xaxis_title="Charger type code",
            yaxis_title="Average demand"
        )

        style_location_chart(fig_type, 280)

        st.plotly_chart(
            fig_type,
            use_container_width=True,
            key="location_type_comparison"
        )

    # ---------------------------------------------------------
    # 7. REAL COMPANY + TYPE COMBINATIONS IN THE DATA
    # ---------------------------------------------------------
    st.markdown("### Charger Company and Type Combinations")

    combination = (
        location_data.groupby(
            ["ChargerCompany", "ChargerType"],
            dropna=False
        )["Demand"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(
            ["mean", "count"],
            ascending=[False, False]
        )
    )

    combination["Combination"] = combination.apply(
        lambda row: (
            f"Company {display_code(row['ChargerCompany'])}"
            f" / Type {display_code(row['ChargerType'])}"
        ),
        axis=1
    )

    if combination.empty:
        st.warning("No charger combinations are available.")
        st.stop()

    best_combo = combination.iloc[0]

    best_company = best_combo["ChargerCompany"]
    best_type = best_combo["ChargerType"]
    best_combo_mean = float(best_combo["mean"])
    best_combo_count = int(best_combo["count"])

    combo_fig = px.bar(
        combination.sort_values("mean", ascending=True),
        x="mean",
        y="Combination",
        orientation="h",
        text="mean",
        custom_data=["count"],
        color_discrete_sequence=[GOLD]
    )

    combo_fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "%{y}<br>"
            "Average demand: %{x:.2f}<br>"
            "Records: %{customdata[0]:,}"
            "<extra></extra>"
        )
    )

    combo_fig.update_layout(
        xaxis_title="Average demand",
        yaxis_title="Company / type combination"
    )

    style_location_chart(
        combo_fig,
        height=max(260, 55 * len(combination))
    )

    st.plotly_chart(
        combo_fig,
        use_container_width=True,
        key="location_company_type_combinations"
    )

    best_col1, best_col2, best_col3 = st.columns(3)

    best_col1.metric(
        "Highest-average combination",
        (
            f"Company {display_code(best_company)} "
            f"/ Type {display_code(best_type)}"
        )
    )

    best_col2.metric(
        "Combination average demand",
        f"{best_combo_mean:.2f}"
    )

    best_col3.metric(
        "Records in combination",
        f"{best_combo_count:,}"
    )

    st.caption(
        "Only combinations present in the filtered data are displayed. "
        "The highest average is an observed result, not proof that the "
        "equipment itself caused higher demand."
    )

    # For location mode, examine the highest-average combo at that location.
    # For the other modes, examine the entire filtered dataset so the
    # time-based charts represent the selected filters.
    if analysis_mode == "Location Analysis":

        best_pair_data = location_data[
            (location_data["ChargerCompany"] == best_company)
            & (location_data["ChargerType"] == best_type)
        ].copy()

        chart_scope_label = (
            f"Company {display_code(best_company)} / "
            f"Type {display_code(best_type)} at {selected_location}"
        )

    else:
        best_pair_data = location_data.copy()

        if analysis_mode == "Charger Equipment Analysis":
            chart_scope_label = "the selected equipment filters"
        else:
            chart_scope_label = "the selected demand range and location filter"

    # ---------------------------------------------------------
    # 8. HOURLY DEMAND FOR THE ANALYSIS SCOPE
    # ---------------------------------------------------------
    st.divider()
    st.markdown("### At What Hour Is Average Demand Highest?")

    
    st.caption(
        "Hourly demand patterns across all charger companies and "
        "charger types at the selected location. Hover over each "
        "bar to see the number of records."
    )

    hourly_combo = (
        location_data.groupby("StartHour")["Demand"]
        .agg(["mean", "count"])
        .reindex(range(24))
    )



    valid_hours = hourly_combo.dropna(subset=["mean"])

    peak_hour = None
    peak_demand = None
    peak_hour_count = 0

    if not valid_hours.empty:

        peak_hour = int(valid_hours["mean"].idxmax())
        peak_demand = float(valid_hours.loc[peak_hour, "mean"])
        peak_hour_count = int(valid_hours.loc[peak_hour, "count"])

    hourly_fig = go.Figure()

    hourly_fig.add_trace(
        go.Bar(
            x=[f"{hour:02d}:00" for hour in hourly_combo.index],
            y=hourly_combo["mean"],
            marker_color=GOLD,
            customdata=hourly_combo[["count"]].fillna(0).to_numpy(),
            hovertemplate=(
                "Hour: %{x}<br>"
                "Average demand: %{y:.2f}<br>"
                "Records: %{customdata[0]:.0f}"
                "<extra></extra>"
            )
        )
    )

    if peak_hour is not None:
        hourly_fig.add_trace(
            go.Scatter(
                x=[f"{peak_hour:02d}:00"],
                y=[peak_demand],
                mode="markers+text",
                marker=dict(
                    color=DARK_GOLD,
                    size=10
                ),
                text=[f"Peak: {peak_demand:.2f}"],
                textposition="top center",
                name="Highest average",
                hovertemplate=(
                    "Peak hour: %{x}<br>"
                    "Average demand: %{y:.2f}"
                    "<extra></extra>"
                )
            )
        )

    hourly_fig.update_layout(
        xaxis_title="Hour of day",
        yaxis_title="Average demand",
        showlegend=False
    )

    style_location_chart(hourly_fig, 320)

    st.plotly_chart(
        hourly_fig,
        use_container_width=True,
        key="location_hourly_demand"
    )

    if peak_hour is not None:

        hour_col1, hour_col2 = st.columns(2)

        hour_col1.metric(
            "Highest-average hour",
            f"{peak_hour:02d}:00"
        )

        hour_col2.metric(
            "Average demand at that hour",
            f"{peak_demand:.2f}"
        )

        if peak_hour_count < 10:
            st.warning(
                f"The peak hour is based on only {peak_hour_count} records. "
                "Treat it as a preliminary observation rather than a "
                "reliable recurring peak."
            )
        else:
            st.caption(
                f"The highest-average hour has {peak_hour_count:,} records."
            )

    else:
        st.info("There are no valid hourly demand observations.")

    # ---------------------------------------------------------
    # 9. DEMAND BY CHARGING DURATION
    # ---------------------------------------------------------
    st.divider()
    st.markdown("### Which Charging Duration Has Higher Demand?")

    duration_bins = [
        -0.001, 30, 60, 120, 240, np.inf
    ]

    duration_labels = [
        "≤30 min",
        "31–60 min",
        "61–120 min",
        "121–240 min",
        ">240 min"
    ]

    
    duration_data = location_data.copy()


    duration_data["DurationRange"] = pd.cut(
        duration_data["Duration"],
        bins=duration_bins,
        labels=duration_labels,
        include_lowest=True
    )

    duration_summary = (
        duration_data.groupby(
            "DurationRange",
            observed=False
        )["Demand"]
        .agg(["mean", "count"])
        .reindex(duration_labels)
        .dropna(subset=["mean"])
        .reset_index()
    )

    duration_summary = duration_summary.sort_values(
        "mean",
        ascending=True
    )

    duration_fig = px.bar(
        duration_summary,
        x="mean",
        y="DurationRange",
        orientation="h",
        text="mean",
        custom_data=["count"],
        color_discrete_sequence=[SAGE]
    )

    duration_fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "Duration: %{y}<br>"
            "Average demand: %{x:.2f}<br>"
            "Records: %{customdata[0]:,}"
            "<extra></extra>"
        )
    )

    duration_fig.update_layout(
        xaxis_title="Average demand",
        yaxis_title="Charging duration"
    )

    style_location_chart(duration_fig, 300)

    st.plotly_chart(
        duration_fig,
        use_container_width=True,
        key="location_duration_demand"
    )

    if not duration_summary.empty:

        top_duration = duration_summary.sort_values(
            "mean", ascending=False
        ).iloc[0]

        st.caption(
            f"Highest observed average: {top_duration['DurationRange']} "
            f"with demand {top_duration['mean']:.2f} across "
            f"{int(top_duration['count']):,} records. "
            "This describes an association, not proof that longer "
            "charging causes higher demand."
        )

    # ---------------------------------------------------------
    # 10. MONTHLY DEMAND PATTERN
    # ---------------------------------------------------------
    st.divider()
    st.markdown("### How Does Demand Change Month by Month?")

    
    st.caption(
    "Monthly demand patterns across all charger companies and "
    "charger types at the selected location."
     )

    monthly_combo = (
    location_data.groupby("Month")["Demand"]
    .agg(["mean", "count"])
    .reindex(all_months)
    )


    valid_months = monthly_combo.dropna(subset=["mean"])

    peak_month = None
    peak_value = None
    peak_month_demand = None
    peak_month_count = 0

    if not valid_months.empty:

        peak_month = valid_months["mean"].idxmax()
        peak_value = float(valid_months.loc[peak_month, "mean"])
        peak_month_demand = peak_value
        peak_month_count = int(valid_months.loc[peak_month, "count"])

    monthly_fig = go.Figure()

    monthly_fig.add_trace(
        go.Scatter(
            x=[str(month) for month in monthly_combo.index],
            y=monthly_combo["mean"],
            mode="lines+markers",
            name="Monthly average demand",
            line=dict(
                color=GOLD,
                width=2.5
            ),
            marker=dict(size=6),
            customdata=monthly_combo[["count"]].fillna(0).to_numpy(),
            connectgaps=False,
            hovertemplate=(
                "Month: %{x}<br>"
                "Average demand: %{y:.2f}<br>"
                "Records: %{customdata[0]:.0f}"
                "<extra></extra>"
            )
        )
    )

    if peak_month is not None:
        monthly_fig.add_trace(
            go.Scatter(
                x=[str(peak_month)],
                y=[peak_month_demand],
                mode="markers+text",
                marker=dict(
                    color=DARK_GOLD,
                    size=10
                ),
                text=[f"Peak: {peak_month_demand:.2f}"],
                textposition="top center",
                name="Highest monthly average",
                hovertemplate=(
                    "Peak month: %{x}<br>"
                    "Average demand: %{y:.2f}"
                    "<extra></extra>"
                )
            )
        )

    monthly_fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average demand",
        showlegend=False
    )

    style_location_chart(monthly_fig, 320)

    st.plotly_chart(
        monthly_fig,
        use_container_width=True,
        key="location_monthly_demand"
    )

    if peak_month is not None:

        month_col1, month_col2 = st.columns(2)

        month_col1.metric(
            "Highest-average month",
            str(peak_month)
        )

        month_col2.metric(
            "Average demand that month",
            f"{peak_month_demand:.2f}"
        )

        
        
        if peak_month_count < 10:
            st.warning(
                f"The peak month is based on {int(peak_month_count)} records "
                "at this location. Treat it cautiously when the sample size is small."
            )
        else:
            st.caption(
                f"The peak month contains {peak_month_count:,} records."
        )

    else:
        st.info("There are no valid monthly demand observations.")

    # ---------------------------------------------------------
    # 11. AVERAGE DEMAND BY LOCATION
    # Show this when the page is not already focused on one location.
    # ---------------------------------------------------------
    if analysis_mode != "Location Analysis":

        st.divider()
        st.markdown("### Average Demand by Location")

        location_summary = (
            location_data.groupby("Location")["Demand"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("mean", ascending=True)
        )

        location_fig = px.bar(
            location_summary,
            x="mean",
            y="Location",
            orientation="h",
            text="mean",
            custom_data=["count"],
            color_discrete_sequence=[GOLD]
        )

        location_fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "Location: %{y}<br>"
                "Average demand: %{x:.2f}<br>"
                "Records: %{customdata[0]:,}"
                "<extra></extra>"
            )
        )

        location_fig.update_layout(
            xaxis_title="Average demand",
            yaxis_title="Location"
        )

        style_location_chart(
            location_fig,
            height=max(350, 32 * len(location_summary))
        )

        st.plotly_chart(
            location_fig,
            use_container_width=True,
            key="filtered_location_comparison"
        )

        st.caption(
            "Locations are ranked by average demand in the filtered records. "
            "Record counts matter: a high average from very few records "
            "may not represent normal demand."
        )

    # ---------------------------------------------------------
    # 12. WEEKDAY VS WEEKEND
    # ---------------------------------------------------------
    st.divider()
    st.markdown("### Weekday vs Weekend Demand")

    
    weekday_summary = (
        location_data.groupby("DayGroup")["Demand"]
        .agg(["mean", "sum", "count"])
    )


    weekday_row = weekday_summary.loc["Weekday"] \
        if "Weekday" in weekday_summary.index else None

    weekend_row = weekday_summary.loc["Weekend"] \
        if "Weekend" in weekday_summary.index else None

    weekday_avg = (
        float(weekday_row["mean"])
        if weekday_row is not None else None
    )

    weekend_avg = (
        float(weekend_row["mean"])
        if weekend_row is not None else None
    )

    day_col1, day_col2 = st.columns(2)

    with day_col1:

        st.markdown("**Average demand comparison**")

        day_chart_data = []

        if weekday_row is not None:
            day_chart_data.append({
                "DayGroup": "Weekday",
                "AverageDemand": float(weekday_row["mean"]),
                "Records": int(weekday_row["count"])
            })

        if weekend_row is not None:
            day_chart_data.append({
                "DayGroup": "Weekend",
                "AverageDemand": float(weekend_row["mean"]),
                "Records": int(weekend_row["count"])
            })

        if day_chart_data:

            day_chart_df = pd.DataFrame(day_chart_data)

            day_fig = px.bar(
                day_chart_df,
                x="DayGroup",
                y="AverageDemand",
                text="AverageDemand",
                custom_data=["Records"],
                color="DayGroup",
                color_discrete_map={
                    "Weekday": GOLD,
                    "Weekend": SAGE
                }
            )

            day_fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "%{x}<br>"
                    "Average demand: %{y:.2f}<br>"
                    "Records: %{customdata[0]:,}"
                    "<extra></extra>"
                )
            )

            day_fig.update_layout(
                xaxis_title="",
                yaxis_title="Average demand",
                showlegend=False
            )

            style_location_chart(day_fig, 280)

            st.plotly_chart(
                day_fig,
                use_container_width=True,
                key="location_weekday_weekend_average"
            )

        else:
            st.info("Not enough valid records for a weekday/weekend comparison.")

    with day_col2:

        st.markdown("**Share of total recorded demand**")

        
        share_data = (
            location_data.groupby("DayGroup")["Demand"]
            .sum()
            .reindex(["Weekday", "Weekend"])
            .dropna()
            .reset_index()
        )


        share_data.columns = ["DayGroup", "TotalDemand"]

        share_data = share_data[share_data["TotalDemand"] >= 0]

        if (
            not share_data.empty
            and share_data["TotalDemand"].sum() > 0
        ):

            share_fig = px.pie(
                share_data,
                names="DayGroup",
                values="TotalDemand",
                hole=0.55,
                color="DayGroup",
                color_discrete_map={
                    "Weekday": GOLD,
                    "Weekend": SAGE
                }
            )

            share_fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate=(
                    "%{label}<br>"
                    "Total demand: %{value:.2f}<br>"
                    "Share: %{percent}"
                    "<extra></extra>"
                )
            )

            share_fig.update_layout(
                showlegend=True
            )

            style_location_chart(share_fig, 280)

            st.plotly_chart(
                share_fig,
                use_container_width=True,
                key="location_weekday_weekend_share"
            )

        else:
            st.info(
                "A share chart cannot be calculated because the total "
                "recorded demand is zero or unavailable."
            )

    # Display record counts to give context to the averages.
    count_col1, count_col2 = st.columns(2)

    count_col1.metric(
        "Weekday records",
        f"{int(weekday_row['count']):,}"
        if weekday_row is not None else "0"
    )

    count_col2.metric(
        "Weekend records",
        f"{int(weekend_row['count']):,}"
        if weekend_row is not None else "0"
    )

    # ---------------------------------------------------------
    # 13. BUSINESS CONCLUSION
    # ---------------------------------------------------------
    st.divider()
    st.markdown("### Business Final Conclusion")
    st.markdown("**What the filtered data suggests**")

    business_insights = []
    business_actions = []

    # A. Weekday vs weekend interpretation
    if weekday_avg is not None and weekend_avg is not None:

        difference = weekend_avg - weekday_avg
        tolerance = 0.5

        if abs(difference) <= tolerance:

            business_insights.append(
                f"Weekday average demand ({weekday_avg:.2f}) and "
                f"weekend average demand ({weekend_avg:.2f}) are "
                f"broadly similar (difference: {abs(difference):.2f})."
            )

            business_actions.append(
                "Compare hourly demand, waiting times and charger "
                "occupancy before changing weekday or weekend staffing."
            )

        elif difference > 0:

            business_insights.append(
                f"Weekend average demand ({weekend_avg:.2f}) is higher "
                f"than weekday demand ({weekday_avg:.2f}) "
                f"by {difference:.2f}."
            )

            business_actions.append(
                "Review weekend occupancy and waiting times. Consider "
                "operational changes only if the pattern is consistent "
                "and charger capacity is constrained."
            )

        else:

            business_insights.append(
                f"Weekday average demand ({weekday_avg:.2f}) is higher "
                f"than weekend demand ({weekend_avg:.2f}) "
                f"by {abs(difference):.2f}."
            )

            business_actions.append(
                "Review weekday occupancy and waiting times. Use quieter "
                "periods for maintenance if operational data supports it."
            )

    # B. Best observed equipment combination
    business_insights.append(
        f"Company {display_code(best_company)} with charger Type "
        f"{display_code(best_type)} has the highest observed average "
        f"demand among the combinations in the current filtered data "
        f"({best_combo_mean:.2f}, based on {best_combo_count:,} records)."
    )

    if best_combo_count < 10:
        business_actions.append(
            "The highest-average charger combination has fewer than "
            "10 records. Gather more observations before using it to "
            "make equipment purchasing or expansion decisions."
        )
    else:
        business_actions.append(
            f"Evaluate Company {display_code(best_company)} / Type "
            f"{display_code(best_type)} using utilization, uptime, "
            "energy delivered, maintenance costs and revenue per charger "
            "before considering additional deployment."
        )

    # C. Peak hour interpretation
    if peak_hour is not None and peak_demand is not None:

        business_insights.append(
            f"The highest observed hourly average in the current "
            f"analysis scope is at {peak_hour:02d}:00 "
            f"({peak_demand:.2f}, {peak_hour_count:,} records)."
        )

        if peak_hour_count < 10:
            business_actions.append(
                "Validate the apparent peak hour with more records "
                "before changing staffing or charger availability."
            )
        else:
            business_actions.append(
                "Check charger occupancy and waiting times around "
                f"{peak_hour:02d}:00 to determine whether additional "
                "capacity or staffing is actually needed."
            )

    # D. Peak month interpretation
    if peak_month is not None and peak_month_demand is not None:

        business_insights.append(
            f"The highest observed monthly average is in {peak_month} "
            f"({peak_month_demand:.2f}, {peak_month_count:,} records)."
        )

        if peak_month_count < 10:
            business_actions.append(
                "Treat the peak month cautiously because it contains "
                "fewer than 10 records. Check the data coverage before "
                "drawing seasonal conclusions."
            )
        else:
            business_actions.append(
                "Compare the monthly pattern across additional periods "
                "before planning seasonal staffing or capacity changes."
            )

    # E. Show conclusion and actions
    st.markdown("**Key observations**")

    if business_insights:
        for insight in business_insights:
            st.markdown(f"- {insight}")
    else:
        st.info("Not enough information is available to draw conclusions.")

    st.markdown("**Recommended next steps**")

    if business_actions:
        for action in business_actions:
            st.markdown(f"- {action}")

    # Scope-aware disclaimer
    if analysis_mode == "Location Analysis":
        conclusion_scope = f"the selected location: {selected_location}"
    elif analysis_mode == "Charger Equipment Analysis":
        conclusion_scope = "the selected charger equipment filters"
    else:
        conclusion_scope = "the selected demand range and location filter"

    st.caption(
        f"These conclusions are based on observed patterns in "
        f"{conclusion_scope}. They describe associations in the available "
        "data, not proof of causation. Confirm sample sizes, charger "
        "occupancy, waiting times, uptime, energy delivered and revenue "
        "before making operational or investment decisions."
    )

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

            c1.write(f"**Historical average demand:** "f"{historical_avg_demand:.2f}")

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
