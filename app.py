import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Cycle-Aware Running Dashboard",
    layout="wide"
)

st.title("🏃‍♀️ Cycle-Aware Athlete Monitoring")

st.write(
"""
This dashboard explores how **menstrual cycle physiology, training load,
recovery, and performance interact** in endurance running.
"""
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_excel("running_engine_output.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    # Unlogged periods leave cycleDay counting up forever; those phases are unknown
    stale = df["cycleDay"] > 45
    df.loc[stale, ["cycleDay", "cyclePhase", "cycleProgress"]] = np.nan
    return df.sort_values("Date")


df = load_data()

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------

st.sidebar.header("Filters")

time_filter = st.sidebar.selectbox(
    "Time Window",
    ["All Data", "Last 30 Days", "Last Cycle"]
)

df_filtered = df.copy()

if time_filter == "Last 30 Days":
    cutoff = df["Date"].max() - timedelta(days=30)
    df_filtered = df[df["Date"] >= cutoff]

elif time_filter == "Last Cycle":
    # Most recent complete cycle: from the second-to-last period start up to the last one
    cycle_starts = df.loc[df["cycleDay"] == 1, "Date"]
    if len(cycle_starts) >= 2:
        prev_start, last_start = cycle_starts.iloc[-2], cycle_starts.iloc[-1]
        df_filtered = df[
            (df["Date"] >= prev_start) &
            (df["Date"] < last_start)
        ]

if df_filtered.empty:
    st.warning("No data in this time window.")
    st.stop()


def fmt_int(v):
    return "—" if pd.isna(v) else int(v)


def latest_value(col):
    s = df_filtered[col].dropna()
    return s.iloc[-1] if len(s) else np.nan

# ---------------------------------------------------
# SECTION 1 — CURRENT STATUS
# ---------------------------------------------------

st.header("1️⃣ Current Physiological Status")

latest = df_filtered.iloc[-1]
st.caption(f"Data as of {latest['Date'].date()}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Readiness Score", fmt_int(latest_value("readiness_score")))
col2.metric("Cycle Phase", latest_value("cyclePhase") if pd.notna(latest_value("cyclePhase")) else "—")
col3.metric("3-Day Load", fmt_int(latest_value("load_3d")))
col4.metric("Session Flag", latest["session_flag"])

# ---------------------------------------------------
# SECTION 2 — PHYSIOLOGY
# ---------------------------------------------------

st.header("2️⃣ Physiological Signals")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Readiness Timeline")
    st.line_chart(
        df_filtered.set_index("Date")["readiness_score"]
    )

with col2:
    st.subheader("Resting Heart Rate")
    fig, ax = plt.subplots()
    ax.plot(
        df_filtered["Date"],
        df_filtered["restingHeartRate"]
    )
    ax.set_title("Resting HR")
    st.pyplot(fig)

# ---------------------------------------------------
# SECTION 3 — CYCLE PHYSIOLOGY
# ---------------------------------------------------

st.header("3️⃣ Physiology Across Cycle Phases")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    df.boxplot(column="restingHeartRate", by="cyclePhase", ax=ax)
    plt.suptitle("")
    ax.set_title("Resting HR by Cycle Phase")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    df.boxplot(column="total_sleep_hours", by="cyclePhase", ax=ax)
    plt.suptitle("")
    ax.set_title("Sleep Duration by Cycle Phase")
    st.pyplot(fig)

# ---------------------------------------------------
# SECTION 4 — TRAINING LOAD
# ---------------------------------------------------

st.header("4️⃣ Training Load")

col1, col2 = st.columns(2)

with col1:
    st.subheader("3-Day Training Load")
    st.line_chart(
        df_filtered.set_index("Date")["load_3d"]
    )

with col2:
    st.subheader("Load vs Heart Rate")
    st.scatter_chart(
        df_filtered,
        x="load_3d",
        y="avgHr"
    )

# ---------------------------------------------------
# SECTION 5 — PERFORMANCE
# ---------------------------------------------------

st.header("5️⃣ Running Performance")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    df.boxplot(column="efficiency", by="cyclePhase", ax=ax)
    plt.suptitle("")
    ax.set_title("Efficiency by Cycle Phase")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    d = df.dropna(subset=["cycleProgress", "efficiency"])
    ax.scatter(d["cycleProgress"], d["efficiency"], alpha=0.3)
    if len(d) > 1:
        z = np.polyfit(d["cycleProgress"], d["efficiency"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(d["cycleProgress"].min(), d["cycleProgress"].max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2)
    ax.set_title("Efficiency vs Cycle Progress")
    st.pyplot(fig)

# ---------------------------------------------------
# SECTION 6 — SESSION TYPES
# ---------------------------------------------------

st.header("6️⃣ Session Classification")

session_counts = df_filtered["session_flag"].value_counts()
st.bar_chart(session_counts)

# ---------------------------------------------------
# SECTION 7 — CYCLE-AWARE INTERPRETATION
# ---------------------------------------------------

st.header("7️⃣ Cycle-Aware Interpretation")

cycle = latest_value("cyclePhase")
cycle = cycle if pd.notna(cycle) else "Unknown"
rhr_z = latest_value("rhr_z")
rhr_z = 0 if pd.isna(rhr_z) else rhr_z
load = latest_value("load_3d")
load = 0 if pd.isna(load) else load
sleep_debt = latest_value("sleep_debt_3day")
sleep_debt = 0 if pd.isna(sleep_debt) else sleep_debt

# Physiological signal
if rhr_z > 1.5:
    physiology_signal = "Elevated HR"
else:
    physiology_signal = "Normal physiology"

# Load classification
if load > 2500:
    load_state = "High"
elif load > 1500:
    load_state = "Moderate"
else:
    load_state = "Low"

# Sleep classification
if sleep_debt > 5:
    sleep_state = "High"
elif sleep_debt > 3:
    sleep_state = "Moderate"
else:
    sleep_state = "Low"

# Interpretation logic
if physiology_signal == "Elevated HR":
    if cycle == "Luteal" and load_state != "High":
        interpretation = "Hormonal physiology likely."
    elif load_state == "High":
        interpretation = "Training fatigue likely."
    elif sleep_state != "Low":
        interpretation = "Recovery deficit likely."
    else:
        interpretation = "Physiological stress detected."
else:
    interpretation = "Physiology within normal range."

# Display summary
st.markdown("### Interpretation Summary")

st.write(f"**Physiology signal:** {physiology_signal}")
st.write(f"**Cycle phase:** {cycle}")
st.write(f"**Training load:** {load_state}")
st.write(f"**Sleep debt:** {sleep_state}")

st.write("---")

st.success(f"**Interpretation:** {interpretation}")

# ---------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------

st.header("Dataset Preview")

st.dataframe(df_filtered.tail(20))
