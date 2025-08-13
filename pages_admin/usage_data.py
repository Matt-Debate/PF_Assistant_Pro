import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from auth import get_usage_logs
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Usage Data", layout="wide")
st.title("📊 Usage Data Panel")

# Load data
data = get_usage_logs()
df = pd.DataFrame(data, columns=["username", "tool", "timestamp", "success", "feedback"])

if df.empty:
    st.warning("No usage data found.")
    st.stop()

# Convert timestamps
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.day_name()
else:
    st.error("No 'timestamp' column found in data.")
    st.stop()

# Section 1: Raw Logs
st.subheader("📄 Raw Logs")
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False)
st.download_button("⬇️ Download Raw Logs (CSV)", csv, "raw_logs.csv", "text/csv")

# Section 2: Usage Per Tool (Bar Chart)
st.subheader("🛠️ Usage Per Tool")
tool_counts = df['tool'].value_counts().reset_index()
tool_counts.columns = ['Tool', 'Count']
st.plotly_chart(px.bar(tool_counts, x='Tool', y='Count', title='Tool Usage Counts'), use_container_width=True)

# Section 3: Daily Usage (Line Chart)
st.subheader("📆 Daily Usage")
daily_counts = df.groupby('date').size().reset_index(name='count')
fig = px.line(daily_counts, x='date', y='count', markers=True, title="Daily Total Usage")
fig.update_traces(mode='lines+markers', hovertemplate='Date: %{x}<br>Count: %{y}')
fig.update_layout(xaxis_title='Date', yaxis_title='Usage Count')
st.plotly_chart(fig, use_container_width=True)

# Section 4: Heatmap of Usage by Hour and Day of Week
st.subheader("🔥 Hourly Usage Heatmap")
heatmap_data = df.groupby(['weekday', 'hour']).size().unstack(fill_value=0).reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)
heatmap_fig = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='Blues',
    colorbar=dict(title='Usage Count')
))
heatmap_fig.update_layout(
    xaxis_title='Hour of Day',
    yaxis_title='Day of Week',
    title='Usage Heatmap by Hour and Day of Week'
)
st.plotly_chart(heatmap_fig, use_container_width=True)

# CSV export of heatmap data
csv_heatmap = heatmap_data.reset_index().to_csv(index=False)
st.download_button("⬇️ Download Heatmap Data (CSV)", csv_heatmap, "heatmap_data.csv", "text/csv")
