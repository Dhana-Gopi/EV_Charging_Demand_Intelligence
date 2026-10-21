"""EV Charging Demand Intelligence | Streamlit dashboard.
Run: streamlit run ev_app.py
Place ChargingRecords.csv and the six trained .pkl files beside this file.
"""
from pathlib import Path
import pickle
import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='EV Charging Demand Intelligence', page_icon='⚡', layout='wide', initial_sidebar_state='collapsed')
BASE = Path(__file__).resolve().parent
YELLOW, DARK, GREY, PALE, WHITE = '#F2C300', '#222222', '#666666', '#FFFBE6', '#FFFFFF'
YELLOWS = ['#F2C300','#D6A800','#FFE071','#B78A00','#FFF0A8']
DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
FEATURES = ['Location','ChargerCompany','ChargerType','StartHour','DayOfWeek','Month','IsWeekend']

st.markdown(f'''<style>
.stApp {{background:{WHITE}; color:{DARK}}}
.block-container {{padding:1.1rem 1rem 2rem;max-width:100%}}
h1,h2,h3 {{color:{DARK}!important}}
[data-testid="stMetric"] {{background:{PALE};border:1px solid #F0E8C3;border-radius:9px;padding:10px}}
[data-testid="stMetricLabel"] {{font-size:12px;font-weight:700}}
[data-testid="stMetricValue"] {{font-size:22px;font-weight:850}}
.insight {{background:{PALE};border-left:4px solid {YELLOW};border-radius:6px;padding:9px 12px;margin:5px 0 12px;font-size:12px;line-height:1.5}}
.chart-title {{font-size:14px;font-weight:800;margin:4px 0}}
.header {{text-align:center;font-size:29px;font-weight:900;font-style:italic;text-decoration:underline;padding:15px 0 2px}}
.subheader {{text-align:center;color:{GREY};font-size:13px;margin-bottom:12px}}
[data-baseweb="tab-highlight"] {{background:{YELLOW}!important}}
.board-insight {{background:#FFFBE6;border-left:4px solid #F2C300;border-radius:7px;padding:12px 13px;margin:0 0 12px;font-size:13px;line-height:1.6}} .board-title {{font-size:13px;font-weight:800;margin-bottom:6px;color:#222}}
</style>''', unsafe_allow_html=True)

def note(message):
    st.markdown(f'<div class="insight">{html.escape(str(message))}</div>',unsafe_allow_html=True)

def chart(fig, height=300):
    fig.update_layout(height=height, margin=dict(l=8,r=8,t=18,b=8),paper_bgcolor=WHITE,plot_bgcolor=WHITE,
                      font=dict(color=DARK,size=11),legend=dict(orientation='h',y=-0.22),colorway=YELLOWS)
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(gridcolor='#F2F2F2',zeroline=False)
    st.plotly_chart(fig,width='stretch',config={'displayModeBar':False})

def location_bar(frame, value, title=None, counts=False):
    """Horizontal ranking: full names remain readable in compact dashboard columns."""
    frame = frame.sort_values(value, ascending=True).copy()
    fig = go.Figure(go.Bar(
        x=frame[value], y=frame['Location'].astype(str), orientation='h',
        marker_color=YELLOW,
        customdata=frame[['Sessions']].to_numpy() if 'Sessions' in frame else None,
        hovertemplate='%{y}<br>Value: %{x:,.2f}<extra></extra>',
        text=[f'{v:,.0f}' if counts else f'{v:.2f}' for v in frame[value]],
        textposition='outside', cliponaxis=False))
    fig.update_layout(xaxis_title=title or value, yaxis_title='',
                      margin=dict(l=10,r=36,t=12,b=20))
    fig.update_yaxes(automargin=True, tickfont=dict(size=10))
    fig.update_xaxes(automargin=True)
    return fig

def heading(text):
    st.markdown(f'<div class="chart-title">{html.escape(text)}</div>',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    path = BASE/'ChargingRecords.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Place your original ChargingRecords.csv beside ev_app.py.')
    df = pd.read_csv(path)
    needed = {'StartDatetime','EndDatetime','Demand','Location','ChargerCompany','ChargerType'}
    if not needed.issubset(df.columns):
        raise ValueError('Missing CSV columns: '+', '.join(sorted(needed-set(df.columns))))
    for c in ['StartDatetime','EndDatetime']:
        df[c] = pd.to_datetime(df[c],errors='coerce')
    df['Demand'] = pd.to_numeric(df['Demand'],errors='coerce')
    df = df.dropna(subset=['StartDatetime','EndDatetime','Demand']).copy()
    calculated = (df['EndDatetime']-df['StartDatetime']).dt.total_seconds()/60
    df = df.loc[calculated>=0].copy()
    df['Duration'] = calculated.loc[df.index]
    # Match the notebook: remove exact duplicate rows after recalculating Duration.
    df = df.drop_duplicates().copy()
    df['Year'] = df['StartDatetime'].dt.year
    df['Month'] = df['StartDatetime'].dt.month
    df['MonthLabel'] = df['StartDatetime'].dt.strftime('%Y-%m')
    df['StartHour'] = df['StartDatetime'].dt.hour
    df['DayIndex'] = df['StartDatetime'].dt.dayofweek
    df['DayName'] = df['StartDatetime'].dt.day_name()
    df['IsWeekend'] = (df['DayIndex']>=5).astype(int)
    return df

@st.cache_resource(show_spinner=False)
def load_models():
    folder = BASE
    filenames = {'reg':'lightgbm_demand_model.pkl','pre':'demand_preprocessor.pkl',
                 'reg_features':'demand_feature_columns.pkl','cls':'lightgbm_demand_classifier.pkl',
                 'cls_features':'classification_feature_columns.pkl','threshold':'demand_classification_threshold.pkl'}
    result={}
    for k,name in filenames.items():
        p=folder/name
        if not p.exists():
            raise FileNotFoundError(f'Missing trained model file: {p}')
        with p.open('rb') as f: result[k]=pickle.load(f)
    if list(result['reg_features'])!=list(result['cls_features']):
        raise ValueError('Regression and classification saved feature orders differ.')
    if list(result['reg_features'])!=FEATURES:
        raise ValueError('Saved feature columns do not match the expected seven inputs: '+str(result['reg_features']))
    return result

def metrics(df):
    cols=st.columns(6)
    vals=[('Sessions',f'{len(df):,}'),('Avg Demand',f'{df.Demand.mean():.2f}'),
          ('Total Demand',f'{df.Demand.sum():,.0f}'),('Avg Duration',f'{df.Duration.mean():.1f} min'),
          ('Locations',f'{df.Location.nunique():,}'),('Peak Start Hour',f'{int(df.groupby("StartHour").Demand.mean().idxmax()):02d}:00')]
    for col,(name,value) in zip(cols,vals): col.metric(name,value)

def bar(df,x,y,orientation='v',color=None):
    fig=px.bar(df,x=x,y=y,orientation=orientation,color=color,color_discrete_sequence=YELLOWS,
               text_auto='.2f' if y!='Sessions' else True)
    fig.update_traces(marker_line_width=0,textposition='outside',cliponaxis=False)
    return fig

def hourly_line(g):
    fig=px.line(g,x='StartHour',y='Demand',markers=True,color_discrete_sequence=[YELLOW])
    fig.update_xaxes(tickmode='array',tickvals=list(range(0,24,3)),
                     ticktext=[f'{h:02d}:00' for h in range(0,24,3)],title='Charging start hour')
    fig.update_yaxes(title='Average demand')
    return fig


def board_section(title, narrative, implication=None):
    safe_title=html.escape(title)
    safe_body=html.escape(narrative)
    tail=(f'<div style="margin-top:7px"><b>Management implication:</b> {html.escape(implication)}</div>' if implication else '')
    st.markdown(f'<div class="board-insight"><div class="board-title">{safe_title}</div><div>{safe_body}</div>{tail}</div>',unsafe_allow_html=True)

def board_story(df, overall, filtered=False, selection=None):
    """Compute board-ready interpretations from the exact displayed subset."""
    n=len(df); start=df.StartDatetime.min(); end=df.StartDatetime.max()
    loc=df.groupby('Location').agg(avg=('Demand','mean'),sessions=('Demand','size')).sort_values('avg',ascending=False)
    busy=loc.sort_values('sessions',ascending=False)
    scope=(f'Selected view: location {selection[0]}, year {selection[1]}, company {selection[2]}, type {selection[3]}. ' if filtered else '')
    share=f' This represents {n/len(overall):.1%} of all cleaned records.' if filtered else ''
    board_section('01 | Scope and purpose',f'{scope}From {start:%d %b %Y} to {end:%d %b %Y}, this view covers {n:,} charging sessions across {len(loc)} location category/categories. Average demand is {df.Demand.mean():.2f} units per session; average duration is {df.Duration.mean():.1f} minutes.{share} Demand units follow the source dataset.', 'Use these measures to assess charging-session behavior, not station utilization or installed capacity.')
    top=loc.iloc[0]; high=loc.index[0]; peak=busy.iloc[0]; busy_name=busy.index[0]
    if len(loc)>1:
        narrative=f'{high} leads in average demand at {top.avg:.2f} units across {int(top.sessions):,} sessions. {busy_name} has the highest activity at {int(peak.sessions):,} sessions (average demand {peak.avg:.2f}). High per-session demand and high traffic are different priorities.'
    else:
        narrative=f'{high} contributes {int(top.sessions):,} sessions, averaging {top.avg:.2f} demand units. Its historical full-dataset average is {overall.loc[overall.Location==high,"Demand"].mean():.2f}; the current filters may select only part of its activity.'
    board_section('02 | Location performance',narrative,'Review throughput and busy periods at high-volume locations; examine high-demand locations separately. Do not infer capacity shortages from averages alone.')
    for field,label in [('ChargerCompany','company'),('ChargerType','type')]:
        g=df.groupby(field).agg(avg=('Demand','mean'),sessions=('Demand','size')).sort_values('avg',ascending=False)
        if len(g)>1:
            leader=g.iloc[0]; second=g.iloc[-1]
            desc=f'{label.title()} {g.index[0]} averages {leader.avg:.2f} units over {int(leader.sessions):,} sessions, compared with {label} {g.index[-1]} at {second.avg:.2f} units over {int(second.sessions):,} sessions. This is an observed difference, not proof that one option causes higher demand.'
        else:
            desc=f'Only {label} {g.index[0]} is represented in the current selection: {int(g.iloc[0].sessions):,} sessions averaging {g.iloc[0].avg:.2f} units. No comparison with excluded {label}s is possible.'
        board_section('03 | Charger '+label.title(),desc,'Compare demand together with session volumes and location mix. Numeric codes are not verified equipment specifications.')
    combo=df.groupby(['ChargerCompany','ChargerType']).agg(avg=('Demand','mean'),sessions=('Demand','size')).sort_values('sessions',ascending=False)
    k=combo.index[0]; v=combo.iloc[0]
    board_section('04 | Observed company/type mix',f'The most frequently observed combination is Company {k[0]} / Type {k[1]}, with {int(v.sessions):,} sessions and {v.avg:.2f} average demand. There are {len(combo)} observed combinations in this selection; absent combinations are not assumed to exist.','Investigate the configurations that handle the most sessions, while checking per-session demand separately.')
    h=df.groupby('StartHour').agg(avg=('Demand','mean'),sessions=('Demand','size'))
    peak_hour=int(h.avg.idxmax()); peak_n=int(h.loc[peak_hour,'sessions'])
    busy_hour=int(h.sessions.idxmax())
    board_section('05 | Hourly charging behavior',f'Sessions starting at {peak_hour:02d}:00 have the highest average demand ({h.loc[peak_hour,"avg"]:.2f} units; {peak_n:,} sessions). The largest number of sessions begins at {busy_hour:02d}:00 ({int(h.loc[busy_hour,"sessions"]):,} sessions). These are different indicators.','Investigate staffing, scheduling, and power requirements at relevant hours. Start-hour demand is not a measurement of instantaneous grid load.')
    durations=df.groupby('Location').Duration.mean().sort_values(ascending=False)
    board_section('06 | Duration and turnover',f'Average session duration is {df.Duration.mean():.1f} minutes. {durations.index[0]} records the longest location-level mean at {durations.iloc[0]:.1f} minutes within the selection.','Long stays may affect turnover; confirm actual charger occupancy and queues before claiming congestion.')
    months=df.groupby('MonthLabel').agg(avg=('Demand','mean'),sessions=('Demand','size'))
    days=df.groupby('DayName').Demand.mean().sort_values(ascending=False)
    if len(months)>1:
        best=months.avg.idxmax(); worst=months.avg.idxmin()
        temporal=f'Among {len(months)} observed months, {best} has the highest mean demand ({months.loc[best,"avg"]:.2f}; {int(months.loc[best,"sessions"]):,} sessions) and {worst} the lowest ({months.loc[worst,"avg"]:.2f}). '
    else:
        temporal=f'The selection contains {len(months)} observed month, so no month-to-month trend can be inferred. '
    temporal+=f'{days.index[0]} has the highest weekday mean ({days.iloc[0]:.2f}). Partial or low-volume months should not be treated as representative.'
    board_section('07 | Calendar patterns',temporal,'Use these observations for historical planning; this is not a future demand forecast.')
    caution=(' The selection has fewer than 100 sessions, so averages may be unstable.' if n<100 else (' This selection has fewer than 500 sessions; interpret subgroup comparisons cautiously.' if n<500 else ''))
    board_section('08 | Decision and limitations',f'Prioritize reviewing the busiest locations and hours, then assess high-demand configurations and long-duration sessions.{caution} The dataset does not provide installed charger counts, charging power ratings, queues, occupancy, or evidence of causality.','Validate equipment capacity and operational data before recommending expansion, replacement, or staffing changes.')

def summary_note(df):
    loc=df.groupby('Location').agg(Avg=('Demand','mean'),Sessions=('Demand','size')).sort_values('Avg',ascending=False)
    top=loc.index[0]
    busy=loc.sort_values('Sessions',ascending=False).index[0]
    return (f'{top} has the highest average demand ({loc.iloc[0].Avg:.2f}; {int(loc.iloc[0].Sessions):,} sessions), '
            f'while {busy} records the most sessions ({int(loc.loc[busy,"Sessions"]):,}). '
            'Average demand and session volume answer different operational questions; small samples need caution.')

try:
    data=load_data()
except Exception as exc:
    st.error(str(exc));st.stop()

st.markdown('<div class="header">⚡ EV CHARGING DEMAND INTELLIGENCE</div>',unsafe_allow_html=True)
st.markdown('<div class="subheader">Usage patterns • Location intelligence • Regression • Demand classification</div>',unsafe_allow_html=True)
note('Objective: Explore historical EV charging demand, compare locations and charger combinations, and estimate demand for a selected charging-session scenario. Predictions are not time-series forecasts.')
page=st.radio('Dashboard navigation',['Executive Overview','Location-Wise Demand Analysis','Demand Prediction','Demand Classification'],horizontal=True,label_visibility='collapsed')
st.divider()

if page=='Executive Overview':
    st.subheader('📊 Executive Overview')
    metrics(data)
    st.caption(f'Historical records: {data.StartDatetime.min():%d %b %Y} – {data.StartDatetime.max():%d %b %Y}. Demand uses the units supplied by the source dataset.')
    left,right=st.columns([3.1,.9],gap='medium')
    with left:
        a,b,c=st.columns(3)
        with a:
            heading('Average Demand by Location')
            g=data.groupby('Location',as_index=False).agg(AvgDemand=('Demand','mean'),Sessions=('Demand','size')).sort_values('AvgDemand',ascending=False)
            chart(location_bar(g,'AvgDemand','Average demand'),355)
            note(f'{g.iloc[0].Location} has the highest average demand ({g.iloc[0].AvgDemand:.2f}) but only {g.iloc[0].Sessions:,} sessions. Compare sample size and session volume before considering infrastructure expansion.')
        with b:
            heading('Average Demand by Start Hour')
            g=data.groupby('StartHour',as_index=False).Demand.mean()
            chart(hourly_line(g),300)
            note(f'Peak recorded start hour: {int(g.loc[g.Demand.idxmax(),"StartHour"]):02d}:00 (average {g.Demand.max():.2f}).')
        with c:
            heading('Demand by Charger Type')
            g=data.groupby('ChargerType',as_index=False).agg(AvgDemand=('Demand','mean'),Sessions=('Demand','size'))
            g['ChargerType']=g.ChargerType.map(lambda v:f'Type {v}')
            chart(bar(g,'ChargerType','AvgDemand'))
            note(f'{g.loc[g.AvgDemand.idxmax(),"ChargerType"]} has higher mean demand; volume and installed charger counts are not equivalent.')
        a,b,c=st.columns(3)
        with a:
            heading('Charging Sessions by Location')
            g=data.groupby('Location').size().reset_index(name='Sessions').sort_values('Sessions',ascending=False)
            chart(location_bar(g,'Sessions','Sessions',counts=True),355)
            note(f'{g.iloc[0].Location} handles the most recorded sessions ({g.iloc[0].Sessions:,}). Prioritize reviewing its busiest hours and existing capacity; volume alone does not prove congestion.')
        with b:
            heading('Company / Type Demand')
            g=data.groupby(['ChargerCompany','ChargerType'],as_index=False).Demand.mean()
            g['Combination']=g.apply(lambda r:f'C{r.ChargerCompany} / T{r.ChargerType}',axis=1)
            chart(bar(g,'Combination','Demand'))
            note(f'Top observed combination: {g.loc[g.Demand.idxmax(),"Combination"]}, mean demand {g.Demand.max():.2f}. Only observed combinations are shown.')
        with c:
            heading('Average Duration by Location')
            g=data.groupby('Location',as_index=False).Duration.mean().sort_values('Duration',ascending=False)
            chart(location_bar(g,'Duration','Minutes'),355)
            note(f'{g.iloc[0].Location} has the longest mean recorded duration ({g.iloc[0].Duration:.1f} minutes). Duration is not charger utilization.')
        a,b=st.columns(2)
        with a:
            heading('Monthly Average Demand (Historical)')
            g=data.groupby('MonthLabel',as_index=False).Demand.mean().sort_values('MonthLabel')
            chart(px.line(g,x='MonthLabel',y='Demand',markers=True,color_discrete_sequence=[YELLOW]),280)
            note('Historical monthly averages describe observed charging sessions, not future forecasts. Check sample sizes for unusual months.')
        with b:
            heading('Weekday Average Demand')
            g=data.groupby('DayName',as_index=False).Demand.mean()
            g['DayName']=pd.Categorical(g.DayName,categories=DAYS,ordered=True)
            g=g.sort_values('DayName')
            chart(bar(g,'DayName','Demand'),280)
            note(f'{g.loc[g.Demand.idxmax(),"DayName"]} has the highest recorded mean demand ({g.Demand.max():.2f}).')
    with right:
        st.markdown('### 💡 Executive Briefing')
        board_story(data,data)

elif page=='Location-Wise Demand Analysis':
    st.subheader('📍 Location-Wise Demand Analysis')
    f1,f2,f3,f4=st.columns(4)
    with f1: loc=st.selectbox('Location',['All']+sorted(data.Location.astype(str).unique().tolist()))
    with f2: year=st.selectbox('Year',['All']+sorted(data.Year.unique().tolist()))
    with f3: company=st.selectbox('Charger Company',['All']+sorted(data.ChargerCompany.unique().tolist()))
    with f4: ctype=st.selectbox('Charger Type',['All']+sorted(data.ChargerType.unique().tolist()))
    d=data.copy()
    if loc!='All': d=d[d.Location.astype(str)==loc]
    if year!='All': d=d[d.Year==year]
    if company!='All': d=d[d.ChargerCompany==company]
    if ctype!='All': d=d[d.ChargerType==ctype]
    if d.empty:
        st.warning('No charging sessions match this combination. Change one or more filters.');st.stop()
    metrics(d)
    st.caption(f'Selection: Location {loc} | Year {year} | Company {company} | Type {ctype} | {len(d):,} sessions')
    left,right=st.columns([3.1,.9],gap='medium')
    with left:
        a,b,c=st.columns(3)
        with a:
            heading('Mean Demand by Location')
            g=d.groupby('Location',as_index=False).agg(Avg=('Demand','mean'),Sessions=('Demand','size')).sort_values('Avg',ascending=False)
            chart(location_bar(g,'Avg','Average demand'),355)
            note(f'Within the current filter selection, {g.iloc[0].Location} has the highest mean demand ({g.iloc[0].Avg:.2f}; {g.iloc[0].Sessions:,} sessions). This is a filtered comparison, not an overall ranking.')
        with b:
            heading('Demand by Charger Company')
            g=d.groupby('ChargerCompany',as_index=False).agg(Avg=('Demand','mean'),Sessions=('Demand','size'))
            g['Company']=g.ChargerCompany.map(lambda x:f'Company {x}')
            chart(bar(g,'Company','Avg'))
            note(f'{g.loc[g.Avg.idxmax(),"Company"]} has the highest mean ({g.Avg.max():.2f}) among selected sessions; comparisons are conditional on the filters.')
        with c:
            heading('Demand by Charger Type')
            g=d.groupby('ChargerType',as_index=False).agg(Avg=('Demand','mean'),Sessions=('Demand','size'))
            g['Type']=g.ChargerType.map(lambda x:f'Type {x}')
            chart(bar(g,'Type','Avg'))
            note(f'{g.loc[g.Avg.idxmax(),"Type"]} has mean demand {g.Avg.max():.2f}; type labels are numeric codes from the dataset.')
        a,b,c=st.columns(3)
        with a:
            heading('Observed Company / Type Combinations')
            g=d.groupby(['ChargerCompany','ChargerType'],as_index=False).agg(Avg=('Demand','mean'),Sessions=('Demand','size'))
            g['Combo']=g.apply(lambda r:f'C{r.ChargerCompany} / T{r.ChargerType}',axis=1)
            chart(bar(g,'Combo','Avg'))
            note(f'{g.loc[g.Avg.idxmax(),"Combo"]} has highest mean demand ({g.Avg.max():.2f}); no unobserved combinations are invented.')
        with b:
            heading('Hourly Demand Pattern')
            g=d.groupby('StartHour',as_index=False).Demand.mean()
            chart(hourly_line(g),300)
            note(f'Peak start hour for this selection: {int(g.loc[g.Demand.idxmax(),"StartHour"]):02d}:00; mean {g.Demand.max():.2f}.')
        with c:
            heading('Charging Duration by Location')
            g=d.groupby('Location',as_index=False).Duration.mean().sort_values('Duration',ascending=False)
            chart(location_bar(g,'Duration','Minutes'),355)
            note(f'Longest average session duration: {g.iloc[0].Location}, {g.iloc[0].Duration:.1f} minutes.')
        a,b=st.columns(2)
        with a:
            heading('Monthly Demand and Session Counts')
            g=d.groupby('MonthLabel',as_index=False).agg(Avg=('Demand','mean'),Sessions=('Demand','size')).sort_values('MonthLabel')
            chart(px.line(g,x='MonthLabel',y='Avg',markers=True,hover_data=['Sessions'],color_discrete_sequence=[YELLOW]))
            note(f'Highest monthly mean: {g.loc[g.Avg.idxmax(),"MonthLabel"]} ({g.Avg.max():.2f}, n={int(g.loc[g.Avg.idxmax(),"Sessions"]):,}). Sparse months require caution.')
        with b:
            heading('Session Volume by Company and Type')
            g=d.groupby(['ChargerCompany','ChargerType'],as_index=False).size()
            g['Combo']=g.apply(lambda r:f'C{r.ChargerCompany} / T{r.ChargerType}',axis=1)
            g=g.rename(columns={'size':'Sessions'})
            chart(bar(g,'Combo','Sessions'))
            note(f'{g.loc[g.Sessions.idxmax(),"Combo"]} has most recorded sessions ({g.Sessions.max():,}); this is not a count of installed chargers.')
    with right:
        st.markdown('### 💡 Filtered Executive Briefing')
        board_story(d,data,filtered=True,selection=(loc,year,company,ctype))

else:
    is_class=page=='Demand Classification'
    st.subheader('🔎 Demand Classification' if is_class else '📈 Demand Prediction')
    st.caption('Choose session-start information. Both models use exactly seven features; no end time or completed duration is required.')
    try:
        models=load_models()
    except Exception as exc:
        st.error(str(exc));st.stop()
    a,b,c,d=st.columns(4)
    with a: location=st.selectbox('Location',sorted(data.Location.unique().tolist()))
    with b: charger_company=st.selectbox('Charger Company',sorted(data.ChargerCompany.unique().tolist()),format_func=lambda x:f'Company {x}')
    with c: charger_type=st.selectbox('Charger Type',sorted(data.ChargerType.unique().tolist()),format_func=lambda x:f'Type {x}')
    with d: start_hour=st.slider('Charging Start Hour',0,23,18)
    a,b,c=st.columns(3)
    with a: day=st.selectbox('Day of Week',list(range(7)),format_func=lambda i:DAYS[i])
    with b: month=st.selectbox('Month',list(range(1,13)),format_func=lambda i:pd.Timestamp(2026,i,1).strftime('%B'))
    with c: st.metric('Weekend (derived)', 'Yes' if day>=5 else 'No')
    record=pd.DataFrame([{'Location':location,'ChargerCompany':charger_company,'ChargerType':charger_type,
                          'StartHour':start_hour,'DayOfWeek':day,'Month':month,'IsWeekend':int(day>=5)}],columns=FEATURES)
    st.caption('Weekend is automatically derived from the chosen day to avoid inconsistent inputs.')
    if st.button('Predict Demand Class' if is_class else 'Predict Charging Demand',type='primary'):
        try:
            transformed=models['pre'].transform(record)
            left,right=st.columns([1,1],gap='large')
            if is_class:
                pred=int(models['cls'].predict(transformed)[0])
                probabilities=models['cls'].predict_proba(transformed)[0]
                classes=list(models['cls'].classes_)
                low=float(probabilities[classes.index(0)])
                high=float(probabilities[classes.index(1)])
                threshold=float(models['threshold'])
                with left:
                    st.metric('Predicted Class','High Demand' if pred==1 else 'Low Demand')
                    st.metric('Training Demand Threshold',f'{threshold:.2f}')
                    note(f'The model classifies this scenario as {"High" if pred else "Low"} Demand. The training-set median ({threshold:.2f}) defined the class labels; this is not a probability cutoff.')
                with right:
                    fig=go.Figure(go.Bar(x=['Low Demand','High Demand'],y=[low*100,high*100],marker_color=['#FFE071',YELLOW],text=[f'{low:.1%}',f'{high:.1%}'],textposition='outside'))
                    fig.update_yaxes(range=[0,110],title='Model probability (%)')
                    heading('Predicted Class Probabilities');chart(fig,320)
                    note(f'High-demand probability: {high:.1%}; low-demand probability: {low:.1%}. These are model estimates, not guarantees or measured charging demand.')
            else:
                prediction=float(models['reg'].predict(transformed)[0])
                baseline=float(data.Demand.mean())
                with left:
                    st.metric('Predicted Demand',f'{prediction:.2f} units')
                    st.metric('Historical Overall Mean',f'{baseline:.2f} units',delta=f'{prediction-baseline:+.2f} units')
                    note(f'For this selected session scenario, the model estimates {prediction:.2f} demand units. This is {abs(prediction-baseline):.2f} units {"above" if prediction>=baseline else "below"} the overall historical mean.')
                with right:
                    heading('Prediction vs Historical Baseline')
                    fig=go.Figure(go.Bar(x=['Historical mean','Model prediction'],y=[baseline,prediction],marker_color=['#FFE071',YELLOW],text=[f'{baseline:.2f}',f'{prediction:.2f}'],textposition='outside'))
                    chart(fig,320)
                    note('The baseline is the full cleaned dataset mean, not a matched-location control group. Predictions may have error and should not be treated as actual outcomes.')
            st.info('These models estimate demand from session-start attributes. Their outputs do not forecast monthly demand, queues, or the number of chargers required.')
        except Exception as exc:
            st.error(f'Prediction could not be completed: {exc}')
    else:
        note('Select the inputs above and click Predict to view the model output, a comparison chart, and an interpretation.')
