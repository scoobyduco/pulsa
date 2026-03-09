import streamlit as st
import pandas as pd
from datetime import timedelta

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_excel("running_engine_output.xlsx")

df["Date"] = pd.to_datetime(df["Date"])

st.title("🏃 Cycle-Aware Running Dashboard")

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------

st.sidebar.header("Filters")

time_filter = st.sidebar.selectbox(
    "Select time window",
    ["All Data", "Last 30 Days", "Last Cycle"]
)

# -----------------------------
# APPLY FILTER
# -----------------------------

df_filtered = df.copy()

if time_filter == "Last 30 Days":

    cutoff = df["Date"].max() - timedelta(days=30)

    df_filtered = df[df["Date"] >= cutoff]

elif time_filter == "Last Cycle":

    if "cyclePhase" in df.columns:

        cycle_starts = df[df["cyclePhase"] == "Menstrual"]

        if len(cycle_starts) >= 2:

            last_start = cycle_starts.iloc[-1]["Date"]
            prev_start = cycle_starts.iloc[-2]["Date"]

            df_filtered = df[
                (df["Date"] >= prev_start) &
                (df["Date"] <= last_start)
            ]

# -----------------------------
# TODAY STATUS
# -----------------------------

latest = df_filtered.iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric("Readiness Score", int(latest["readiness_score"]))
col2.metric("3 Day Load", int(latest["load_3d"]))
col3.metric("Session Type", latest["session_flag"])

# -----------------------------
# READINESS TIMELINE
# -----------------------------

st.header("Readiness Timeline")

st.line_chart(
    df_filtered.set_index("Date")["readiness_score"]
)

# -----------------------------
# EFFICIENCY TREND
# -----------------------------

st.header("Efficiency Trend")

st.line_chart(
    df_filtered.set_index("Date")["efficiency_index"]
)

# -----------------------------
# LOAD VS HR
# -----------------------------

st.header("Load vs Heart Rate")

st.scatter_chart(
    df_filtered,
    x="load_3d",
    y="avgHr"
)

# -----------------------------
# SESSION TYPES
# -----------------------------

st.header("Session Classification")

session_counts = df_filtered["session_flag"].value_counts()

st.bar_chart(session_counts)

# -----------------------------
# DATA TABLE
# -----------------------------

st.header("Dataset Preview")

st.dataframe(df_filtered.tail(20))
