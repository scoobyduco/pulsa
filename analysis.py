"""
Batch analysis: generates static figures from running_engine_output.xlsx.
Aligns with app.py visualizations for reports/export.

Also provides build_learn_context() for LEARN page — data-driven, no phase inference.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

PHASE_ORDER = ["Menstrual", "Follicular", "Ovulatory", "Luteal"]


def build_learn_context(df: pd.DataFrame) -> dict:
    """
    Build context for LEARN page. Uses df["cycle_phase"] as single source of truth.
    NO assign_phase(), NO % splits, NO cycle_day → phase inference.

    Expects columns: cycle_phase, cycle_day, resting_hr, readiness_score, performance_score.
    Maps common aliases: cyclePhase→cycle_phase, restingHeartRate→resting_hr, etc.
    """
    if df is None or len(df) == 0:
        return _empty_context()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    # Normalize column names (app data uses camelCase)
    col_map = {
        "cycle_phase": ["cyclePhase", "cycle_phase"],
        "cycle_day": ["cycleDay", "cycle_day"],
        "resting_hr": ["restingHeartRate", "resting_hr"],
        "readiness_score": ["readiness_score"],
        "performance_score": ["performance_smoothed", "performance_score"],
    }
    for canon, aliases in col_map.items():
        if canon not in df.columns:
            for a in aliases:
                if a in df.columns:
                    df[canon] = df[a]
                    break

    # Validate required columns — hard-require phase + day; return empty on failure (never raise)
    hard_required = ["cycle_phase", "cycle_day"]
    hard_missing = [c for c in hard_required if c not in df.columns or df[c].dropna().empty]
    if hard_missing:
        return _empty_context()
    # Fill optional columns with NaN if missing so downstream code doesn't crash
    for opt in ["resting_hr", "readiness_score", "performance_score"]:
        if opt not in df.columns:
            df[opt] = float("nan")

    df = df.dropna(subset=["cycle_phase", "cycle_day"])
    if len(df) < 3:
        return _empty_context()

    # -------- TIME WINDOWS --------
    short_term = df.tail(7)
    baseline = df.tail(30)

    # -------- CURRENT PHASE (from cycle_phase only) --------
    current_phase = str(df.iloc[-1]["cycle_phase"])
    if current_phase not in PHASE_ORDER:
        current_phase = "Ovulatory"

    # -------- YOUR BODY TODAY --------
    hr_current = short_term["resting_hr"].mean()
    hr_baseline = baseline["resting_hr"].mean()
    hr_delta_raw = hr_current - hr_baseline

    readiness_current = short_term["readiness_score"].mean()
    readiness_baseline = baseline["readiness_score"].mean()

    # Prefer performance_smoothed for deltas when performance_score is all-NaN in short window (better coverage)
    perf_col = "performance_score"
    if short_term["performance_score"].isna().all() and "performance_smoothed" in df.columns:
        perf_col = "performance_smoothed"
    performance_current = short_term[perf_col].mean()
    performance_baseline = baseline[perf_col].mean()
    performance_delta_raw = performance_current - performance_baseline

    # Coerce NaN deltas to 0 so UI shows sensible defaults (e.g. "Around baseline")
    hr_delta = 0.0 if pd.isna(hr_delta_raw) else float(hr_delta_raw)
    performance_delta = 0.0 if pd.isna(performance_delta_raw) else float(performance_delta_raw)

    # -------- PHASE-BASED AGGREGATION --------
    phase_stats_raw = df.groupby("cycle_phase").agg({
        "resting_hr": "mean",
        "readiness_score": "mean",
        "performance_score": "mean",
    }).to_dict("index")

    # Sort by phase_order
    phase_stats = {}
    for p in PHASE_ORDER:
        if p in phase_stats_raw:
            phase_stats[p] = phase_stats_raw[p]

    rdy_delta_val = float(readiness_current - readiness_baseline) if pd.notna(readiness_current) and pd.notna(readiness_baseline) else 0.0

    return {
        "cycle": {
            "cycle_day": int(df.iloc[-1]["cycle_day"]),
            "phase": current_phase,
        },
        "today": {
            "window": "7d_vs_30d",
            "hr_delta": round(hr_delta, 1),
            "readiness_delta": rdy_delta_val,
            "performance_delta": performance_delta,
        },
        "phase_stats": {
            "window": "historical_by_phase",
            "data": phase_stats,
        },
    }


def _empty_context() -> dict:
    return {
        "cycle": {"cycle_day": 17, "phase": "Ovulatory"},
        "today": {"window": "7d_vs_30d", "hr_delta": 0, "readiness_delta": 0, "performance_delta": 0},
        "phase_stats": {"window": "historical_by_phase", "data": {}},
    }


# -----------------------------------------------------------------------------
# Batch figure generation (run as script)
# -----------------------------------------------------------------------------

def _run_batch_analysis():
    df = pd.read_excel("running_engine_output.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])

    os.makedirs("figures", exist_ok=True)

    # 1 Readiness over time
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["readiness_score"])
    plt.title("Readiness Score Over Time")
    plt.xlabel("Date")
    plt.ylabel("Readiness")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/readiness_over_time.png")
    plt.close()

    # 2 Resting HR over time
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["restingHeartRate"])
    plt.title("Resting Heart Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Resting HR")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/resting_hr_over_time.png")
    plt.close()

    # 3 Load vs HR
    plt.figure(figsize=(8, 5))
    plt.scatter(df["load_3d"], df["avgHr"])
    plt.title("Training Load vs HR")
    plt.xlabel("3-Day Load")
    plt.ylabel("Average HR")
    plt.tight_layout()
    plt.savefig("figures/load_vs_hr.png")
    plt.close()

    # 4 Efficiency over time
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["efficiency"])
    plt.title("Running Efficiency Over Time")
    plt.xlabel("Date")
    plt.ylabel("Efficiency Index")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/efficiency_over_time.png")
    plt.close()

    # 5 Resting HR by cycle phase
    plt.figure(figsize=(8, 5))
    df.boxplot(column="restingHeartRate", by="cyclePhase")
    plt.title("Resting HR by Cycle Phase")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("figures/rhr_by_cycle_phase.png")
    plt.close()

    # 6 Sleep by cycle phase (if column exists)
    if "total_sleep_hours" in df.columns:
        plt.figure(figsize=(8, 5))
        df.boxplot(column="total_sleep_hours", by="cyclePhase")
        plt.title("Sleep Duration by Cycle Phase")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("figures/sleep_by_cycle_phase.png")
        plt.close()

    # 7 Efficiency by cycle phase
    plt.figure(figsize=(8, 5))
    df.boxplot(column="efficiency", by="cyclePhase")
    plt.title("Efficiency by Cycle Phase")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("figures/efficiency_by_cycle_phase.png")
    plt.close()

    # 8 Session classification
    plt.figure(figsize=(6, 4))
    df["session_flag"].value_counts().plot(kind="bar")
    plt.title("Run Session Classification")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("figures/session_types.png")
    plt.close()

    print("Analysis complete. Figures saved to figures/")


if __name__ == "__main__":
    _run_batch_analysis()
