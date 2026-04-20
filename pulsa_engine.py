"""
Pulsa Engine — Core brain for cycle-aware running.

DATA → FEATURES → INSIGHTS → PLAN

Takes df_run_daily from engineering.py and produces:
- Feature engineering (pace, efficiency)
- Normalization
- Performance score
- Aggregations by phase / run type / cycle day
- Insights (body, running, performance, training)
- Training windows (peak / fatigue)
- Weekly plan generator

Product principle: Training is NOT cycle-driven.
Cycle ADJUSTS intensity and effort; it does NOT remove key sessions.
"""

import numpy as np
import pandas as pd
from typing import Optional

# Minimum data thresholds
MIN_POINTS_PER_PHASE = 5
MIN_RUNS_FOR_INSIGHTS = 10
ROLLING_WINDOW = 5
PHASE_ORDER = ["Menstrual", "Follicular", "Ovulatory", "Luteal"]


# -----------------------------------------------------------------------------
# 1. Feature Engineering
# -----------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features: pace, efficiency, optional pace_at_fixed_hr.
    """
    df = df.copy()

    # Pace = duration / distance (min/km); prefer pace_min_per_km if present
    if "pace_min_per_km" in df.columns:
        df["pace"] = df["pace_min_per_km"]
        df.drop(columns=["pace_min_per_km"], inplace=True, errors="ignore")
    elif "distance_km" in df.columns and "duration_min" in df.columns:
        df["pace"] = np.where(df["distance_km"] > 0, df["duration_min"] / df["distance_km"], np.nan)
    else:
        df["pace"] = np.nan

    # Efficiency = pace / HR (lower = better at same HR)
    if "avgHr" in df.columns and "pace" in df.columns:
        df["efficiency"] = np.where(
            (df["avgHr"] > 0) & (df["pace"] > 0),
            df["pace"] / df["avgHr"],
            df.get("efficiency_index", np.nan)
        )
    else:
        df["efficiency"] = df.get("efficiency_index", np.nan)
    # Drop the old duplicate; keep 'efficiency' as canonical
    df.drop(columns=["efficiency_index"], inplace=True, errors="ignore")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# -----------------------------------------------------------------------------
# 2. Normalization (robust, handles NaN and outliers)
# -----------------------------------------------------------------------------

def robust_normalize(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Rolling min-max normalization. Handles NaN, uses rolling window.
    Returns values in [0, 1] where higher = better (for efficiency we invert).
    """
    rolling_min = series.rolling(window, min_periods=max(3, window // 2)).min()
    rolling_max = series.rolling(window, min_periods=max(3, window // 2)).max()
    spread = rolling_max - rolling_min
    norm = np.where(spread > 0, (series - rolling_min) / spread, 0.5)
    return pd.Series(np.clip(norm, 0, 1), index=series.index)


def invert_normalize(series: pd.Series, window: int = 14) -> pd.Series:
    """For 'lower is better' metrics (e.g. efficiency = pace/HR): 1 - norm."""
    norm = robust_normalize(series, window)
    return 1 - norm  # higher normalized = better performance


# -----------------------------------------------------------------------------
# 3. Performance Score
# -----------------------------------------------------------------------------

def compute_performance_score(
    df: pd.DataFrame,
    window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """
    performance_score = 0.5 * efficiency_norm + 0.3 * readiness_norm - 0.2 * rhr_penalty
    Smooth with rolling average for stability.
    """
    df = df.copy()

    # Efficiency: lower raw = better → invert normalization
    eff = df["efficiency"] if "efficiency" in df.columns else df.get("efficiency_index", np.nan)
    if pd.notna(eff).any():
        df["efficiency_norm"] = invert_normalize(eff, window=14)
    else:
        df["efficiency_norm"] = 0.5  # neutral

    # Readiness: higher = better
    rdy = df.get("readiness_score", pd.Series(70, index=df.index))
    df["readiness_norm"] = robust_normalize(rdy.clip(0, 100), window=14)

    # RHR penalty: higher RHR = worse. Use z-score style penalty.
    rhr = df.get("restingHeartRate", pd.Series(60, index=df.index))
    rhr_rolling = rhr.rolling(14, min_periods=5).mean()
    rhr_std = rhr.rolling(14, min_periods=5).std()
    rhr_penalty = np.where(
        rhr_std > 0,
        np.clip((rhr - rhr_rolling) / rhr_std, 0, 3),  # only penalize elevated
        0
    )
    df["rhr_penalty"] = rhr_penalty

    # Combine
    df["performance_score"] = (
        0.5 * df["efficiency_norm"].fillna(0.5) +
        0.3 * df["readiness_norm"].fillna(0.5) -
        0.2 * df["rhr_penalty"].clip(0, 2)
    )
    df["performance_score"] = df["performance_score"].clip(0, 1)

    # Smoothed (rolling)
    df["performance_smoothed"] = df["performance_score"].rolling(window, min_periods=1).mean()

    return df


# -----------------------------------------------------------------------------
# 4. Aggregations
# -----------------------------------------------------------------------------

def metrics_by_phase(df: pd.DataFrame) -> dict:
    """Aggregate metrics by cycle phase. Returns 'insufficient data' if < MIN_POINTS."""
    if "cyclePhase" not in df.columns:
        return {"status": "insufficient data", "reason": "No cycle data"}

    phase_counts = df.dropna(subset=["cyclePhase"]).groupby("cyclePhase").size()
    valid_phases = [p for p in PHASE_ORDER if p in phase_counts.index and phase_counts[p] >= MIN_POINTS_PER_PHASE]
    if len(valid_phases) < 2:
        return {"status": "insufficient data", "reason": f"Need ≥{MIN_POINTS_PER_PHASE} points in ≥2 phases"}

    out = {}
    metrics = ["restingHeartRate", "readiness_score", "total_sleep_hours", "efficiency", "performance_score"]
    for m in metrics:
        if m not in df.columns:
            continue
        out[m] = {
            p: df[df["cyclePhase"] == p][m].mean()
            for p in valid_phases
            if df[df["cyclePhase"] == p][m].notna().sum() >= MIN_POINTS_PER_PHASE
        }
    out["status"] = "ok"
    return out


def metrics_by_run_type(df: pd.DataFrame) -> dict:
    """Classify runs and aggregate. Uses distance + aerobic effect as proxy for type."""
    runs = df.dropna(subset=["distance_km", "Date"])
    if len(runs) < MIN_RUNS_FOR_INSIGHTS:
        return {"status": "insufficient data", "reason": f"Need ≥{MIN_RUNS_FOR_INSIGHTS} runs"}

    # Simple classification
    def classify(row):
        d, te = row.get("distance_km", 0), row.get("aerobicTrainingEffect", 2)
        if pd.isna(d) or d <= 0:
            return "unknown"
        if d >= 15:
            return "long"
        if d >= 8 and (te >= 3 or pd.isna(te)):
            return "tempo"
        if d >= 5 and (te >= 2.5 or pd.isna(te)):
            return "moderate"
        return "easy"

    runs = runs.copy()
    runs["run_type"] = runs.apply(classify, axis=1)
    out = runs.groupby("run_type").agg({
        "distance_km": "mean",
        "pace": "mean",
        "efficiency": "mean",
        "avgHr": "mean",
    }).to_dict()
    out["status"] = "ok"
    return out


def performance_by_cycle_day(df: pd.DataFrame) -> dict:
    """Performance score by cycle day (1-28)."""
    if "cycleDay" not in df.columns or "performance_smoothed" not in df.columns:
        return {"status": "insufficient data", "reason": "Missing cycle day or performance"}

    sub = df.dropna(subset=["cycleDay", "performance_smoothed"])
    if len(sub) < 10:
        return {"status": "insufficient data", "reason": "Need ≥10 run days"}

    by_day = sub.groupby(sub["cycleDay"].astype(int).clip(1, 28))["performance_smoothed"].mean()
    return {"status": "ok", "by_day": by_day.to_dict()}


# -----------------------------------------------------------------------------
# 5. Insights Engine
# -----------------------------------------------------------------------------

def generate_insights(df: pd.DataFrame) -> dict:
    """
    Structured insights per cycle phase:
      {
        "Menstrual": {"body": [], "running": [], "performance": [], "training": []},
        "Follicular": {...},
        ...
      }

    Notes:
    - For body/running/performance we use ONLY the subset of rows where cyclePhase == phase.
    - Training messages are phase-guidance (phase-specific, not data-differentials).
    """
    insights = {p: {"body": [], "running": [], "performance": [], "training": []} for p in PHASE_ORDER}

    # Global insufficient-data fallback
    if "cyclePhase" not in df.columns or len(df.dropna(subset=["cyclePhase"])) < 10:
        for p in PHASE_ORDER:
            insights[p]["body"].append(f"Not enough cycle data yet to personalize your {p} phase.")
            insights[p]["running"].append(f"Log more runs to see efficiency patterns in {p}.")
            insights[p]["performance"].append(f"More data needed for performance insights in {p}.")
            insights[p]["training"].append("Once you have 2+ weeks of data, Pulsa will suggest training windows.")
        return insights

    # Build phase-by-phase insights from the phase subset only
    for phase in PHASE_ORDER:
        phase_df = df[df.get("cyclePhase") == phase].dropna(subset=["cyclePhase"])

        # BODY (phase-only)
        if "restingHeartRate" in phase_df.columns:
            rhr = phase_df["restingHeartRate"].dropna()
            if len(rhr) >= 2:
                insights[phase]["body"].append(f"Resting HR averages about {rhr.mean():.0f} bpm in {phase}.")
        if "readiness_score" in phase_df.columns:
            rdy = phase_df["readiness_score"].dropna()
            if len(rdy) >= 2:
                insights[phase]["body"].append(f"Your readiness score averages about {rdy.mean():.0f}/100 in {phase}.")
        if "total_sleep_hours" in phase_df.columns:
            slp = phase_df["total_sleep_hours"].dropna()
            if len(slp) >= 2:
                insights[phase]["body"].append(f"Sleep averages about {slp.mean():.1f} hours in {phase}.")

        # RUNNING (phase-only)
        phase_runs = phase_df.dropna(subset=["efficiency"], how="all") if "efficiency" in phase_df.columns else phase_df
        if "efficiency" in phase_runs.columns:
            eff = phase_runs["efficiency"].dropna()
            if len(eff) >= 2:
                # Efficiency: lower is better (pace/HR). Present as a measured tendency.
                insights[phase]["running"].append(f"Run efficiency averages about {eff.mean():.3f} in {phase}.")
        if "avgHr" in phase_runs.columns:
            hr = phase_runs["avgHr"].dropna()
            if len(hr) >= 2:
                insights[phase]["running"].append(f"Average run HR is about {hr.mean():.0f} bpm in {phase}.")

        # PERFORMANCE (phase-only)
        if "performance_smoothed" in phase_df.columns:
            perf = phase_df["performance_smoothed"].dropna()
            if len(perf) >= 2:
                insights[phase]["performance"].append(f"Performance score averages about {perf.mean():.2f} in {phase}.")

        # TRAINING (phase guidance)
        if phase in ("Follicular", "Ovulatory"):
            insights[phase]["training"].append(
                f"{phase} is a great window for quality. If your body feels good, try tempo or intervals."
            )
        elif phase == "Luteal":
            insights[phase]["training"].append(
                "Keep the week structure, but reduce intensity slightly if fatigue signals show up."
            )
        elif phase == "Menstrual":
            insights[phase]["training"].append(
                "Honor how you feel. Same structure is okay — swap a workout for easy if needed."
            )

        # Per-phase fallbacks so UI always has something
        if not insights[phase]["body"]:
            insights[phase]["body"].append(f"Your {phase} body insights will get sharper as you log more days.")
        if not insights[phase]["running"]:
            insights[phase]["running"].append(f"Your {phase} running insights will appear as you log more HR-based runs.")
        if not insights[phase]["performance"]:
            insights[phase]["performance"].append(f"Performance insights for {phase} improve with more run days.")
        if not insights[phase]["training"]:
            insights[phase]["training"].append("Cycle adjusts effort, not structure.")

    return insights


# -----------------------------------------------------------------------------
# 6. Training Windows
# -----------------------------------------------------------------------------

def compute_training_windows(df: pd.DataFrame) -> dict:
    """
    peak_window = top 20% performance days
    fatigue_window = bottom 20%
    """
    if "performance_smoothed" not in df.columns or "Date" not in df.columns:
        return {"peak_days": [], "low_days": [], "status": "insufficient data"}

    sub = df.dropna(subset=["performance_smoothed", "Date"]).copy()
    if len(sub) < 5:
        return {"peak_days": [], "low_days": [], "status": "insufficient data"}

    q80 = sub["performance_smoothed"].quantile(0.80)
    q20 = sub["performance_smoothed"].quantile(0.20)
    peak = sub[sub["performance_smoothed"] >= q80]["Date"].dt.strftime("%Y-%m-%d").tolist()
    low = sub[sub["performance_smoothed"] <= q20]["Date"].dt.strftime("%Y-%m-%d").tolist()

    return {"peak_days": peak[:20], "low_days": low[:20], "status": "ok"}


# -----------------------------------------------------------------------------
# 6b. Compute Weekly Target (Goal-Driven)
# -----------------------------------------------------------------------------

# DEPRECATED: These functions have moved to plan_engine.py.
# Kept as thin forwards for backward compatibility.
def compute_weekly_km(goal: dict, current_phase: str = "Unknown") -> int:
    from plan_engine import compute_weekly_km as _pe_cwk
    return _pe_cwk(goal, current_phase)

def validate_goal(goal: dict) -> tuple[bool, str]:
    from plan_engine import validate_goal as _pe_vg
    return _pe_vg(goal)


# -----------------------------------------------------------------------------
# 7. Weekly Plan Generator
# -----------------------------------------------------------------------------

# DEPRECATED: generate_week_plan has been replaced by plan_engine.generate_plan().
# Kept as a thin forward for backward compatibility.
def generate_week_plan(
    profile: dict,
    current_phase: str,
    weekly_km: float,
) -> list[dict]:
    from plan_engine import generate_plan as _pe_gen
    _fp = _pe_gen(
        goal_type="maintenance",
        baseline_km=float(weekly_km),
        runs_per_week=4,
    )
    _days_map = _fp.get("weeks", [{}])[0].get("days", {}) if _fp.get("weeks") else {}
    plan = []
    for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        wo = _days_map.get(d, {"type": "rest", "km": 0, "details": ""})
        plan.append({
            "day": d,
            "type": wo.get("type", "rest"),
            "distance": wo.get("km", 0),
            "intensity_note": "",
        })
    return [x for x in plan if x["type"] != "rest" or x["distance"] > 0]


# -----------------------------------------------------------------------------
# 8. Main Pipeline
# -----------------------------------------------------------------------------

def run_pulsa_engine(df_run_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    Full pipeline: features, performance score, insights, windows.
    Returns (df_enriched, insights_dict, training_windows_dict).
    """
    df = add_features(df_run_daily)
    df = compute_performance_score(df)
    insights = generate_insights(df)
    windows = compute_training_windows(df)
    return df, insights, windows
