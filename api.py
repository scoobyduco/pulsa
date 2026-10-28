"""
Pulsa API — FastAPI backend serving the Rork frontend.
Reads pre-computed Excel files (Option B: your data, shared).

Run locally:   uvicorn api:app --reload --port 8000
Deploy:        Railway reads this file via Procfile
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from plan_engine import calculate_vdot, generate_plan

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Pulsa API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten after testing
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data loading ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE_DIR, "user_settings.json")
DEBRIEF_PATH = os.path.join(BASE_DIR, "workout_debriefs.json")

# Garmin keeps counting cycle days when a period isn't logged, so a missed
# log produces "day 200" rows that all fall into Luteal. Treat them as unknown.
MAX_CYCLE_DAY = 45


def _sanitize_cycle(df: pd.DataFrame) -> pd.DataFrame:
    if "cycleDay" not in df.columns:
        return df
    bad = pd.to_numeric(df["cycleDay"], errors="coerce") > MAX_CYCLE_DAY
    for col in ("cycleDay", "cyclePhase", "cycleProgress", "daysUntilNextPeriod"):
        if col in df.columns:
            df.loc[bad, col] = np.nan
    return df


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load daily master + running master. Cached at module level."""
    daily_path   = os.path.join(BASE_DIR, "daily_engine_output.xlsx")
    running_path = os.path.join(BASE_DIR, "running_engine_output.xlsx")

    # Fallback paths
    if not os.path.exists(daily_path):
        daily_path = os.path.join(BASE_DIR, "df_daily_master.xlsx")
    if not os.path.exists(running_path):
        running_path = os.path.join(BASE_DIR, "df_running_master.xlsx")

    daily   = pd.read_excel(daily_path)   if os.path.exists(daily_path)   else pd.DataFrame()
    running = pd.read_excel(running_path) if os.path.exists(running_path) else pd.DataFrame()

    for df in (daily, running):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        _sanitize_cycle(df)

    return daily, running


_daily_df, _running_df = _load()


def daily() -> pd.DataFrame:
    if "Date" not in _daily_df.columns:
        return _daily_df
    return _daily_df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")


def runs() -> pd.DataFrame:
    """Actual runs only. running_engine_output has one row per day; non-run days have no distance."""
    r = _running_df
    if "Date" not in r.columns:
        return r
    if "distance_km" in r.columns:
        r = r[pd.to_numeric(r["distance_km"], errors="coerce") > 0]
    return r.sort_values("Date")


def _today() -> date:
    """Indirection so tests can pin the date."""
    return date.today()


def _settings() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        return {}
    try:
        with open(SETTINGS_PATH) as f:
            s = json.load(f)
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def _parse_date(v: Any) -> date | None:
    if not v:
        return None
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


# ─── Helpers ──────────────────────────────────────────────────────────────────
PHASE_COLORS = {
    "Menstrual":  "#E24B4A",
    "Follicular": "#1D9E75",
    "Ovulatory":  "#D4537E",
    "Luteal":     "#EF9F27",
}
PHASE_EMOJI = {
    "Menstrual":  "🩸",
    "Follicular": "🌱",
    "Ovulatory":  "🌸",
    "Luteal":     "🌙",
}
PHASE_ORDER = ["Menstrual", "Follicular", "Ovulatory", "Luteal"]


def _safe(val: Any, default: Any = None) -> Any:
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return val


def _latest_value(df: pd.DataFrame, *cols: str) -> Any:
    """Most recent non-null value of the first column that has one.
    The latest row is often a partial day (e.g. no resting HR yet)."""
    for col in cols:
        if col in df.columns:
            s = df[col].dropna()
            if len(s):
                return s.iloc[-1]
    return None


def _cycle_length(df: pd.DataFrame) -> int:
    for col in ("actualCycleLength", "applicableMenstrualCycleLength"):
        if col in df.columns:
            v = df[col].dropna().tail(1)
            if len(v) and 21 <= float(v.iloc[0]) <= 45:
                return int(round(float(v.iloc[0])))
    try:
        cl = int(_settings().get("cycle", {}).get("cycle_length", 28))
    except (TypeError, ValueError):
        cl = 28
    return int(max(21, min(cl, 45)))


def _cycle_anchor(df: pd.DataFrame) -> date | None:
    """Start date of the most recent known period.

    Respects user_settings.cycle.use_garmin_cycle: when false the manually
    entered last_period_start wins; otherwise the Garmin cycle data does.
    Each source falls back to the other.
    """
    cyc = _settings().get("cycle", {})
    manual = _parse_date(cyc.get("last_period_start"))

    garmin = None
    if "cycleDay" in df.columns and "Date" in df.columns:
        known = df.dropna(subset=["cycleDay"])
        if len(known):
            row = known.iloc[-1]
            garmin = pd.Timestamp(row["Date"]).date() - timedelta(days=int(float(row["cycleDay"])) - 1)

    if cyc.get("use_garmin_cycle", False):
        return garmin or manual
    return manual or garmin


def _cycle_day_on(day: date, anchor: date | None, cl: int) -> int:
    if anchor is None:
        return 14
    return (day - anchor).days % cl + 1


def _phase_day_bounds(cycle_length: int = 28) -> dict:
    cl = int(max(21, min(cycle_length, 45)))
    m_end = int(cl * 0.2)
    f_end = int(cl * 0.5)
    o_end = int(cl * 0.65)
    return {
        "Menstrual":  (1, m_end),
        "Follicular": (m_end + 1, f_end),
        "Ovulatory":  (f_end + 1, o_end),
        "Luteal":     (o_end + 1, cl),
    }


def _phase_for_day(cycle_day: int, cl: int) -> str:
    for ph, (ps, pe) in _phase_day_bounds(cl).items():
        if ps <= cycle_day <= pe:
            return ph
    return "Luteal"


def _cycle_state(df: pd.DataFrame, day: date | None = None) -> dict:
    """Single source of truth for 'where am I in my cycle' across endpoints."""
    day = day or _today()
    cl = _cycle_length(df)
    anchor = _cycle_anchor(df)
    cd = _cycle_day_on(day, anchor, cl)
    return {"anchor": anchor, "cycleLength": cl, "cycleDay": cd, "phase": _phase_for_day(cd, cl)}


def _resolve_phase(df: pd.DataFrame) -> str:
    return _cycle_state(df)["phase"]


def _data_as_of(df: pd.DataFrame) -> str | None:
    if "Date" not in df.columns or not len(df):
        return None
    return str(pd.Timestamp(df["Date"].max()).date())


# ─── Plan building (shared by /plan, /today, /workout) ────────────────────────
WTYPE_EMOJI = {
    "easy": "🏃‍♀️", "tempo": "⚡", "interval": "🔥", "long": "🛤️",
    "fartlek": "🎢", "hills": "⛰️", "race": "🏁", "rest": "🧘",
}
WTYPE_LABEL = {
    "easy": "Easy run", "tempo": "Tempo run", "interval": "Intervals", "long": "Long run",
    "fartlek": "Fartlek", "hills": "Hill repeats", "race": "Race day", "rest": "Rest",
}
WTYPE_EFFORT = {
    "easy": "Easy", "long": "Moderate", "tempo": "Hard", "interval": "Very hard",
    "fartlek": "Hard", "hills": "Hard", "race": "Race", "rest": "Recovery",
}
QUALITY_TYPES = {"tempo", "interval", "fartlek", "hills"}
DAY_KEYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DIST_MAP = {"5 km": "5k", "10 km": "10k",
            "Half marathon (21 km)": "21k", "Marathon (42 km)": "42k"}


def _vdot_from_runs(r: pd.DataFrame, days: int = 180) -> float | None:
    """Fitness estimate from recent runs. Training runs understate race fitness,
    so take the 90th percentile rather than the median. None → plan_engine estimates from volume."""
    if not {"distance_km", "duration_min", "Date"} <= set(r.columns):
        return None
    sub = r[(r["distance_km"] >= 3) & (r["duration_min"] > 0)]
    sub = sub[sub["Date"] >= sub["Date"].max() - pd.Timedelta(days=days)] if len(sub) else sub
    if len(sub) < 5:
        return None
    vdots = [calculate_vdot(int(m * 60), float(k)) for m, k in zip(sub["duration_min"], sub["distance_km"])]
    return round(float(np.percentile(vdots, 90)), 1)


def _build_plan(d: pd.DataFrame) -> dict:
    """Generate the plan from user settings and map the current week to API rows."""
    today = _today()
    cyc = _cycle_state(d, today)
    cl, anchor = cyc["cycleLength"], cyc["anchor"]

    tr = _settings().get("training", {})
    has_race  = bool(tr.get("has_race", False))
    race_dist = tr.get("race_dist_sel", "10 km")
    baseline  = float(tr.get("baseline_km", 25) or 25)
    rpw       = int(tr.get("days_per_week", 4) or 4)
    goal_time = tr.get("goal_time", "") or ""

    race_date_obj = _parse_date(tr.get("race_date")) if has_race else None
    # A race in the past can't be trained for — fall back to maintenance.
    if race_date_obj and race_date_obj < today:
        race_date_obj = None
        has_race = False

    if has_race and race_dist == "Custom":
        km = float(tr.get("race_dist_custom", 10) or 10)
        goal_type = "5k" if km <= 7 else "10k" if km <= 15 else "21k" if km <= 30 else "42k"
    else:
        goal_type = DIST_MAP.get(race_dist, "10k") if has_race else "maintenance"

    fp = generate_plan(
        goal_type=goal_type,
        goal_time_str=goal_time if has_race else "",
        baseline_km=baseline,
        runs_per_week=rpw,
        current_vdot=_vdot_from_runs(runs()),
        race_date=race_date_obj,
        today=today,
        last_period_start=anchor,
        cycle_length=cl,
    )

    cur_week = fp["weeks"][0] if fp.get("weeks") else {}
    days_map = cur_week.get("days", {})

    week = []
    monday = today - timedelta(days=today.weekday())
    for i, day_key in enumerate(DAY_KEYS):
        wo = days_map.get(day_key, {"type": "rest", "km": 0})
        wt = str(wo.get("type", "rest")).lower()
        day_date = monday + timedelta(days=i)
        day_phase = _phase_for_day(_cycle_day_on(day_date, anchor, cl), cl)
        week.append({
            "workoutId":  f"{day_key}_{day_date.isoformat()}",
            "date":       str(day_date),
            "dayName":    day_key,
            "type":       wt,
            "emoji":      WTYPE_EMOJI.get(wt, "🏃‍♀️"),
            "distanceKm": float(wo.get("km", 0) or 0),
            "details":    wo.get("details", "") or "",
            "isKey":      bool(wo.get("is_key", False)),
            "phase":      day_phase,
            "phaseColor": PHASE_COLORS[day_phase],
            "isRest":     wt == "rest",
            "isToday":    day_date == today,
        })

    return {
        "fp":        fp,
        "week":      week,
        "totalKm":   float(cur_week.get("total_km", baseline)),
        "hasRace":   has_race,
        "raceDist":  race_dist,
        "raceDate":  race_date_obj,
        "goalTime":  goal_time,
    }


def _paces_raw(fp: dict) -> dict:
    raw = (fp.get("paces") or {}).get("_raw", {})
    return {
        "easy":     float(raw.get("easy_fast", 378)),
        "easy_slow": float(raw.get("easy_slow", 409)),
        "tempo":    float(raw.get("tempo", 320)),
        "interval": float(raw.get("interval", 295)),
        "mp":       float(raw.get("mp", 341)),
    }


def _estimate_minutes(wt: str, km: float, paces: dict) -> int:
    """Rough total time: easy-pace km plus a faster share for quality days."""
    if wt == "rest" or km <= 0:
        return 0
    sec_per_km = {
        "easy": paces["easy"], "long": paces["easy_slow"],
        "tempo": 0.6 * paces["easy"] + 0.4 * paces["tempo"],
        "interval": 0.7 * paces["easy"] + 0.3 * paces["interval"],
        "race": paces["mp"],
    }.get(wt, paces["easy"])
    return max(1, int(round(km * sec_per_km / 60)))


# ─── /today ───────────────────────────────────────────────────────────────────
@app.get("/today")
def get_today():
    d = daily()
    today = _today()

    cyc = _cycle_state(d, today)
    phase, cycle_day, cl = cyc["phase"], cyc["cycleDay"], cyc["cycleLength"]
    bounds = _phase_day_bounds(cl)

    # Metrics (latest known value, the newest row is often a partial day)
    rhr_today = _safe(_latest_value(d, "restingHeartRate", "resting_hr"))
    readiness = _safe(_latest_value(d, "readiness_score"))
    sleep_h   = _safe(_latest_value(d, "total_sleep_hours"))

    baseline_30 = d.tail(30)
    rhr_base  = _safe(float(baseline_30["restingHeartRate"].mean())) if "restingHeartRate" in baseline_30.columns else None
    rdy_base  = _safe(float(baseline_30["readiness_score"].mean()))  if "readiness_score" in baseline_30.columns else None
    hr_delta  = (float(rhr_today) - rhr_base) if rhr_today is not None and rhr_base is not None else 0.0
    rdy_delta = (float(readiness) - rdy_base) if readiness is not None and rdy_base is not None else 0.0

    def hr_ctx(delta):
        if abs(delta) <= 3:
            return "Within your normal range"
        return f"+{int(delta)} bpm vs baseline" if delta > 0 else f"{int(delta)} bpm vs baseline"

    def rdy_ctx(delta):
        if delta > 5:
            return "Above your usual"
        if delta >= -5:
            return "In line with your usual"
        return "Below your usual"

    def sleep_ctx(h):
        if h is None:
            return "—"
        if h >= 7:
            return "In line with your baseline"
        if h >= 6:
            return "Slightly lower"
        return "Lower than usual"

    # Recommendation: today's slot from the plan, with phase-aware coaching copy
    phase_body = {
        "Menstrual":  "Hormones are at their lowest — go easy and protect your energy.",
        "Follicular": "Estrogen is rising — a great window to push quality.",
        "Ovulatory":  "Peak estrogen — your strongest window. Go for it.",
        "Luteal":     "Progesterone raises your HR — go by effort, not pace.",
    }
    try:
        plan = _build_plan(d)
    except Exception:  # bad training settings shouldn't take down the home screen
        plan = {"fp": {}, "week": []}
    paces = _paces_raw(plan["fp"])
    slot = next((w for w in plan["week"] if w["isToday"]), None) or {"type": "rest", "distanceKm": 0.0}
    rec_type = slot["type"]
    rec_dist = float(slot["distanceKm"])
    rec_body = phase_body[phase]

    if rec_type == "rest":
        rec_body = "Rest day — recovery is where the training turns into fitness."
    elif rec_type in QUALITY_TYPES and readiness is not None and float(readiness) < 60:
        # Cycle adjusts effort, readiness can too — keep the session, drop the intensity.
        rec_type = "easy"
        rec_body = "Recovery signals are low today — keep the run but make it easy."
    elif rec_type in QUALITY_TYPES and phase in ("Menstrual", "Luteal"):
        rec_body = f"{phase_body[phase]} Keep the session, adjust the effort."

    # Recovery trend (7 days)
    trend_7 = d.dropna(subset=["readiness_score"]).tail(7) if "readiness_score" in d.columns else d.iloc[0:0]
    recovery_trend = []
    for _, row in trend_7.iterrows():
        ph = _safe(row.get("cyclePhase")) or _safe(row.get("cycle_phase")) or phase
        recovery_trend.append({
            "date":      str(pd.Timestamp(row["Date"]).date()),
            "readiness": float(row["readiness_score"]),
            "phase":     str(ph),
        })

    # Trend insight
    rdys = [x["readiness"] for x in recovery_trend]
    if len(rdys) >= 2:
        if rdys[-1] > rdys[0] + 3:
            trend_insight = "Recovery climbing steadily this week — a supportive pattern."
        elif rdys[-1] < rdys[0] - 3:
            trend_insight = "Recovery lower this week — protect sleep and keep load manageable."
        else:
            trend_insight = "Recovery holding steady this week — small shifts are normal."
    else:
        trend_insight = "Keep logging to see your recovery trend."

    # Why this happens
    copy_variant = "aligned"
    if abs(hr_delta) > 5 or abs(rdy_delta) > 8:
        copy_variant = "outlier"
    elif abs(hr_delta) > 2 or abs(rdy_delta) > 4:
        copy_variant = "mixed"

    bottom_lines = {
        "aligned": "This matches the typical pattern for this phase. Keep your plan structured.",
        "outlier": "Your response differs from the typical pattern. Trust your sensations today.",
        "mixed":   "Mixed signals are normal month to month. Use effort cues together with data.",
    }

    # Coming next
    next_phase = PHASE_ORDER[(PHASE_ORDER.index(phase) + 1) % 4]
    _, cur_end = bounds[phase]
    days_to_next = max(1, cur_end - cycle_day + 1)
    next_blurbs = {
        "Menstrual":  "Energy often dips as hormones reset.",
        "Follicular": "Energy often rises — good time to build.",
        "Ovulatory":  "Strength and speed can feel sharper.",
        "Luteal":     "Effort may feel higher at the same pace.",
    }

    name = _settings().get("profile", {}).get("name") or "Runner"
    data_as_of = _data_as_of(d)

    return {
        "greetingName": name,
        "cycleDay":     cycle_day,
        "cyclePhase":   phase,
        "phaseColor":   PHASE_COLORS[phase],
        "todayDate":    f"{today.day} {today.strftime('%b')}",
        "phaseNote":    next_blurbs.get(phase, ""),
        "dataAsOf":     data_as_of,
        "dataStale":    bool(data_as_of) and (today - date.fromisoformat(data_as_of)).days > 2,
        "recommendation": {
            "type":        rec_type,
            "title":       WTYPE_LABEL.get(rec_type, rec_type.capitalize()),
            "distanceKm":  rec_dist,
            "durationMin": _estimate_minutes(rec_type, rec_dist, paces),
            "effortLevel": WTYPE_EFFORT.get(rec_type, "Moderate"),
            "body":        rec_body,
            "workoutId":   slot.get("workoutId"),
        },
        "metrics": {
            "restingHr":        int(rhr_today) if rhr_today is not None else None,
            "restingHrContext": hr_ctx(hr_delta),
            "sleepHours":       round(float(sleep_h), 1) if sleep_h is not None else None,
            "sleepContext":     sleep_ctx(sleep_h),
            "readinessScore":   int(readiness) if readiness is not None else None,
            "readinessContext": rdy_ctx(rdy_delta),
        },
        "recoveryTrend":  recovery_trend,
        "trendInsight":   trend_insight,
        "whyHappens": {
            "copyVariant":    copy_variant,
            "hrDelta":        round(hr_delta, 1),
            "readinessDelta": round(rdy_delta, 1),
            "bottomLine":     bottom_lines[copy_variant],
        },
        "comingNext": {
            "phase":  next_phase,
            "inDays": days_to_next,
            "blurb":  next_blurbs[next_phase],
        },
    }


# ─── /plan ────────────────────────────────────────────────────────────────────
@app.get("/plan")
def get_plan():
    tr = _settings().get("training", {})
    try:
        plan = _build_plan(daily())
    except Exception as e:
        return {
            "goalCard": {"hasRace": False, "raceDist": tr.get("race_dist_sel", "10 km"),
                         "raceDate": "", "goalTime": "", "weeksAway": 0},
            "week": [],
            "totals": {"totalKm": 0, "runCount": 0},
            "error": str(e),
        }

    week = plan["week"]
    race_date = plan["raceDate"]
    goal_card = {
        "hasRace":   plan["hasRace"],
        "raceDist":  plan["raceDist"],
        "raceDate":  race_date.isoformat() if race_date else "",
        "goalTime":  plan["goalTime"] if plan["hasRace"] else "",
        "weeksAway": max(0, (race_date - _today()).days // 7) if race_date else 0,
    }
    return {
        "goalCard": goal_card,
        "week":     week,
        "totals":   {"totalKm": round(plan["totalKm"], 1),
                     "runCount": sum(1 for w in week if not w["isRest"])},
    }


# ─── /insights ────────────────────────────────────────────────────────────────
@app.get("/insights")
def get_insights():
    d = daily()
    r = runs()

    cl = _cycle_length(d)
    bounds = _phase_day_bounds(cl)

    # Performance columns are produced by pulsa_engine and only live in the running output
    performance_rhythm = []
    perf_col = None
    for c in ("performance_smoothed", "performance_score"):
        if c in r.columns and r[c].notna().any():
            perf_col = c
            break

    sub = pd.DataFrame()
    if perf_col and "cycleDay" in r.columns:
        sub = r.dropna(subset=["cycleDay", perf_col]).copy()
        sub["cd"] = sub["cycleDay"].astype(float).astype(int)
        sub = sub[sub["cd"].between(1, 28)]
    if len(sub):
        by_day = sub.groupby("cd")[perf_col].mean()
        full = pd.Series({i: by_day.get(i, np.nan) for i in range(1, 29)})
        full = full.interpolate(method="linear", limit_direction="both").ffill().bfill()
        for cd in range(1, 29):
            performance_rhythm.append({"cycleDay": cd, "score": round(float(full[cd]), 3)})

    peak_day    = max(performance_rhythm, key=lambda x: x["score"]) if performance_rhythm else {"cycleDay": 14, "score": 0.7}
    fatigue_day = min(performance_rhythm, key=lambda x: x["score"]) if performance_rhythm else {"cycleDay": 28, "score": 0.5}
    rhythm_insight = (
        f"+{abs(peak_day['score'] - fatigue_day['score']):.0%} between your peak (day {peak_day['cycleDay']}) "
        f"and lowest (day {fatigue_day['cycleDay']}) — a real, measurable difference."
        if performance_rhythm else "Keep logging runs to see your performance rhythm."
    )

    PLAYBOOK_CFG = [
        ("Follicular", "PUSH", "#E8F5E9", "#2E7D32",
         "Intervals, tempo, long runs",
         "Rising estrogen boosts power — your fastest Zone 2 pace lives here"),
        ("Ovulatory",  "PEAK", "#FCE4EC", "#AD1457",
         "Race, time trial, test yourself",
         "Peak strength and motivation — your highest output window"),
        ("Luteal",     "EASE", "#FFF8E1", "#E65100",
         "Easy runs, steady volume",
         "Progesterone rises, HR elevated — effort costs more"),
        ("Menstrual",  "REST", "#EDE7F6", "#4527A0",
         "Rest or gentle movement",
         "HRV recovering, sleep quality dips — protect the reset"),
    ]

    rdy_by_phase = {}
    if "readiness_score" in d.columns and "cyclePhase" in d.columns:
        g = d.dropna(subset=["cyclePhase", "readiness_score"]).groupby("cyclePhase")["readiness_score"].mean()
        rdy_by_phase = {p: int(round(float(g[p]))) for p in g.index if p in PHASE_ORDER}

    playbook = []
    for ph, mode, color, text_color, action, reason in PLAYBOOK_CFG:
        ps, pe = bounds.get(ph, (1, 7))
        playbook.append({
            "phase":       ph,
            "mode":        mode,
            "dayRange":    f"{ps}–{pe}",
            "action":      action,
            "reason":      reason,
            "avgReadiness": rdy_by_phase.get(ph, 0),
            "color":       color,
        })

    def phase_metric(df, col, lower_is_better=False):
        if col not in df.columns or "cyclePhase" not in df.columns:
            return []
        sub = df.dropna(subset=["cyclePhase", col])
        g = sub.groupby("cyclePhase")[col].mean()
        vals = {p: float(g[p]) for p in PHASE_ORDER if p in g.index}
        if not vals:
            return []
        if lower_is_better:
            best = min(vals, key=vals.get)
            worst = max(vals, key=vals.get)
        else:
            best = max(vals, key=vals.get)
            worst = min(vals, key=vals.get)
        return [
            {
                "phase":     ph,
                "value":     round(vals[ph], 1),
                "color":     PHASE_COLORS[ph],
                "highlight": "best" if ph == best else ("worst" if ph == worst else None),
            }
            for ph in PHASE_ORDER if ph in vals
        ]

    rhr_col = "restingHeartRate" if "restingHeartRate" in d.columns else "resting_hr"
    rhr_by_phase = phase_metric(d, rhr_col, lower_is_better=True)

    z2_by_phase = []
    if "avgHr" in r.columns and "cyclePhase" in r.columns:
        pace_col = next((c for c in ("pace", "pace_min_per_km") if c in r.columns), None)
        if pace_col:
            z2 = r[(r["avgHr"].between(120, 155)) & (r[pace_col] > 0)].dropna(subset=["cyclePhase", pace_col])
            if len(z2) >= 4:
                g = z2.groupby("cyclePhase")[pace_col].mean()
                vals = {p: float(g[p]) for p in PHASE_ORDER if p in g.index}
                best  = min(vals, key=vals.get)
                worst = max(vals, key=vals.get)
                for ph in PHASE_ORDER:
                    if ph in vals:
                        pace_s = vals[ph] * 60
                        z2_by_phase.append({
                            "phase":      ph,
                            "pacePerKmS": int(pace_s),
                            "color":      PHASE_COLORS[ph],
                            "highlight":  "best" if ph == best else ("worst" if ph == worst else None),
                        })

    if rhr_by_phase:
        hi = max(rhr_by_phase, key=lambda x: x["value"])
        lo = min(rhr_by_phase, key=lambda x: x["value"])
        diff = hi["value"] - lo["value"]
        summary = (
            f"In your {hi['phase'].lower()} phase every signal shifts — "
            f"resting HR +{diff:.0f} bpm. Same run, different body."
            if diff >= 1 else
            "Your metrics are consistent across phases — keep logging to see patterns emerge."
        )
    else:
        summary = "Keep logging to see your metrics by phase."

    # Distinct period start dates seen in the data
    n_cycles = 0
    if "cycleDay" in d.columns:
        known = d.dropna(subset=["cycleDay"])
        n_cycles = int((known["Date"] - pd.to_timedelta(known["cycleDay"].astype(float) - 1, unit="D")).nunique())
    n_runs = len(r)

    return {
        "performanceRhythm": performance_rhythm,
        "peakDay":           peak_day,
        "fatigueDay":        fatigue_day,
        "rhythmInsight":     rhythm_insight,
        "playbook":          playbook,
        "cyclesCount":       n_cycles,
        "runsCount":         n_runs,
        "underTheHood": {
            "summary":           summary,
            "restingHrByPhase":  rhr_by_phase,
            "hrvByPhase":        [],
            "zone2PaceByPhase":  z2_by_phase,
        },
    }


# ─── /learn ───────────────────────────────────────────────────────────────────
@app.get("/learn")
def get_learn():
    d = daily()
    cyc = _cycle_state(d)
    phase, cycle_day, cl = cyc["phase"], cyc["cycleDay"], cyc["cycleLength"]
    bounds = _phase_day_bounds(cl)

    PHASE_PREVIEW = {
        "Menstrual":  "Lower energy, hormones reset",
        "Follicular": "Energy rising, recovery improves",
        "Ovulatory":  "Peak energy and strength",
        "Luteal":     "HR elevated, stay steady",
    }
    PHASE_TAGS = {
        "Menstrual":  ["easy runs", "rest days"],
        "Follicular": ["tempo runs", "build"],
        "Ovulatory":  ["intervals", "races"],
        "Luteal":     ["steady runs", "aerobic"],
    }
    PHASE_KEY_MAP = {
        "Menstrual": "menstrual", "Follicular": "follicular",
        "Ovulatory": "ovulatory", "Luteal": "luteal",
    }

    timeline = [
        {
            "key":       PHASE_KEY_MAP[ph],
            "name":      ph,
            "color":     PHASE_COLORS[ph],
            "isCurrent": ph == phase,
        }
        for ph in PHASE_ORDER
    ]

    phase_grid = []
    for ph in PHASE_ORDER:
        ps, pe = bounds[ph]
        phase_grid.append({
            "key":       PHASE_KEY_MAP[ph],
            "emoji":     PHASE_EMOJI[ph],
            "name":      ph,
            "dayRange":  f"Days {ps}–{pe}",
            "preview":   PHASE_PREVIEW[ph],
            "tags":      PHASE_TAGS[ph],
            "isCurrent": ph == phase,
        })

    return {
        "currentPhase": {
            "key":         PHASE_KEY_MAP[phase],
            "emoji":       PHASE_EMOJI[phase],
            "name":        phase,
            "cycleDay":    cycle_day,
            "cycleLength": cl,
        },
        "timeline":     timeline,
        "phaseGrid":    phase_grid,
        "researchStat": "78% of female athletes say their cycle affects performance — yet most plans ignore it.",
        "research": [
            {"authors": "McNulty et al.", "year": 2020, "journal": "Sports Medicine",
             "url": "https://link.springer.com/article/10.1007/s40279-020-01319-3"},
            {"authors": "Sims & Heather", "year": 2018, "journal": "Experimental Physiology",
             "url": "https://physoc.onlinelibrary.wiley.com/doi/10.1113/EP086797"},
            {"authors": "Elliott-Sale et al.", "year": 2021, "journal": "IJSNEM",
             "url": "https://journals.humankinetics.com/view/journals/ijsnem/31/1/article-p55.xml"},
        ],
    }


# ─── /learn/phase/{key} ───────────────────────────────────────────────────────
@app.get("/learn/phase/{key}")
def get_phase_detail(key: str):
    d = daily()

    KEY_MAP = {
        "menstrual": "Menstrual", "follicular": "Follicular",
        "ovulatory": "Ovulatory", "luteal": "Luteal",
    }
    phase = KEY_MAP.get(key.lower())
    if phase is None:
        raise HTTPException(status_code=404, detail=f"Unknown phase '{key}'")
    cl = _cycle_length(d)
    bounds = _phase_day_bounds(cl)
    ps, pe = bounds[phase]

    BODY_BULLETS = {
        "Menstrual":  ["Lower energy and recovery capacity", "Hormones at lowest", "Body is resetting"],
        "Follicular": ["Energy is rising", "Recovery improves", "Estrogen builds steadily"],
        "Ovulatory":  ["Peak energy and strength", "You may feel your strongest", "Estrogen peaks / LH surges"],
        "Luteal":     ["Heart rate slightly elevated", "Recovery may be slower", "Progesterone rises then falls"],
    }
    BEST_FOR = {
        "Menstrual":  ["easy runs", "walking", "rest days"],
        "Follicular": ["tempo runs", "harder workouts", "increasing volume"],
        "Ovulatory":  ["intervals", "races", "key sessions"],
        "Luteal":     ["steady runs", "aerobic efforts", "consistency"],
    }
    COACHING = {
        "Menstrual":  "Go lighter if needed — rest is productive right now.",
        "Follicular": "This is a great phase to push slightly — your body is primed to adapt.",
        "Ovulatory":  "This is your peak performance window. Enjoy it.",
        "Luteal":     "Adjust effort — not volume. The same run will feel harder, and that's normal.",
    }
    SCIENCE = {
        "Menstrual":  "With estrogen and progesterone at their lowest, your body conserves energy for recovery.",
        "Follicular": "Rising estrogen supports muscle repair and glycogen storage.",
        "Ovulatory":  "Peak estrogen enhances neuromuscular function, reaction time, and pain tolerance.",
        "Luteal":     "Progesterone raises core temperature and resting heart rate, making the same effort feel harder.",
    }

    rdy_by_phase = {}
    if "readiness_score" in d.columns and "cyclePhase" in d.columns:
        g = d.dropna(subset=["cyclePhase", "readiness_score"]).groupby("cyclePhase")["readiness_score"].mean()
        rdy_by_phase = {p: float(g[p]) for p in PHASE_ORDER if p in g.index}

    pattern_bars = []
    if rdy_by_phase:
        vmax = max(rdy_by_phase.values())
        vmin = min(rdy_by_phase.values())
        for ph in PHASE_ORDER:
            v = rdy_by_phase.get(ph, (vmax + vmin) / 2)
            pattern_bars.append({
                "phase":     ph,
                "value":     round(v, 1),
                "color":     PHASE_COLORS[ph],
                "isCurrent": ph == phase,
            })

    ph_rdy = rdy_by_phase.get(phase)
    global_rdy = float(np.mean(list(rdy_by_phase.values()))) if rdy_by_phase else None
    if ph_rdy and global_rdy:
        pct = (ph_rdy - global_rdy) / global_rdy * 100
        if abs(pct) >= 2:
            direction = "higher" if pct > 0 else "lower"
            data_insight = f"Your readiness averages {abs(int(pct))}% {direction} than your overall baseline in this phase."
        else:
            data_insight = "Your readiness is consistent across your cycle in this phase."
    else:
        data_insight = "Log more days to see your personal pattern for this phase."

    return {
        "key":          key.lower(),
        "emoji":        PHASE_EMOJI[phase],
        "name":         phase,
        "dayRange":     f"Days {ps}–{pe}",
        "description":  SCIENCE[phase],
        "bodyBullets":  BODY_BULLETS[phase],
        "bestFor":      BEST_FOR[phase],
        "coachingNote": COACHING[phase],
        "patternBars":  pattern_bars,
        "dataInsight":  data_insight,
        "scienceNote":  SCIENCE[phase],
    }


# ─── /workout/{id} ────────────────────────────────────────────────────────────
SEG_COLOR = {"warmup": "#D3D1C7", "cooldown": "#D3D1C7", "recovery": "#5DCAA5"}
WORK_COLOR = {"easy": "#5DCAA5", "tempo": "#EF9F27", "interval": "#EF9F27", "long": "#7F77DD",
              "fartlek": "#EF9F27", "hills": "#D85A30", "race": "#D4537E"}


def _seg(kind: str, title: str, minutes: float, cue: str | None, wt: str = "easy") -> dict:
    return {
        "kind":        kind,
        "title":       title,
        "durationMin": max(1, int(round(minutes))),
        "cue":         cue,
        "color":       SEG_COLOR.get(kind) or WORK_COLOR.get(wt, "#5DCAA5"),
    }


def _repeat(n: int, work: dict, rec: dict, cues: list[str]) -> list[dict]:
    """n work blocks separated by recoveries; cues go on the first, second and last block."""
    out = []
    for i in range(n):
        cue = cues[0] if i == 0 else cues[-1] if i == n - 1 else cues[1] if i == 1 else None
        out.append(dict(work, cue=cue))
        if i < n - 1:
            out.append(dict(rec, cue=rec["cue"] if i == 0 else None))
    return out


def _workout_segments(wt: str, km: float, details: str, paces: dict) -> list[dict]:
    """Session structure sized to the planned distance and the plan's pace zones."""
    if wt == "rest" or km <= 0:
        return []
    easy_min = lambda dist: dist * paces["easy"] / 60  # noqa: E731

    if wt == "easy":
        return [_seg("work", "Easy run", easy_min(km), "Conversational pace — you should be able to chat")]

    if wt == "race":
        return [
            _seg("warmup", "Warm up", 10, "Easy jog plus a few strides"),
            _seg("work", "Race", km * paces["mp"] / 60, "Start controlled, finish strong", wt),
        ]

    if wt == "long":
        total = km * paces["easy_slow"] / 60
        return [
            _seg("warmup", "Warm up", 10, "Start slow, let your body wake up"),
            _seg("work", "Steady run", max(10, total - 15), "Find a sustainable rhythm", wt),
            _seg("cooldown", "Cool down", 5, "Slow to a jog, then walk"),
        ]

    if wt == "tempo":
        m = re.search(r"w/(\d+(?:\.\d+)?)\s*km", details or "")
        tempo_km = float(m.group(1)) if m else round(km * 0.4)
        tempo_total = tempo_km * paces["tempo"] / 60
        blocks = max(1, int(round(tempo_total / 8)))
        work = _seg("work", "Tempo interval", tempo_total / blocks, None, wt)
        rec = _seg("recovery", "Recovery jog", 2, "Slow down, let HR drop")
        cool = max(5, easy_min(km - tempo_km) - 10 - 2 * (blocks - 1))
        return ([_seg("warmup", "Warm up", 10, "Easy pace, find your rhythm")]
                + _repeat(blocks, work, rec, ["Comfortably hard", "Same effort as first", "Last one — hold steady"])
                + [_seg("cooldown", "Cool down", cool, "Easy jog, then walk")])

    if wt == "interval":
        m = re.search(r"(\d+)\s*×\s*(\d+)\s*m", details or "")
        reps, rep_m = (int(m.group(1)), int(m.group(2))) if m else (5, 1000)
        work = _seg("work", "Hard interval", rep_m / 1000 * paces["interval"] / 60, None, wt)
        rec = _seg("recovery", "Recovery", 2, "Walk or jog slowly")
        cool = max(5, easy_min(max(0.0, km - reps * rep_m / 1000)) - 10 - 2 * (reps - 1))
        return ([_seg("warmup", "Warm up", 10, "Easy pace, loosen up")]
                + _repeat(reps, work, rec, ["Hard effort — 85–90% max HR", "Same effort — don't start fast", "Dig deep"])
                + [_seg("cooldown", "Cool down", cool, "Easy jog, then walk")])

    if wt in ("fartlek", "hills"):
        reps = 6
        if wt == "fartlek":
            work = _seg("work", "Surge", 2, None, wt)
            rec = _seg("recovery", "Float", 2, "Easy running, not walking")
            cues = ["Quick but relaxed", "Play with the pace", "Last surge — enjoy it"]
        else:
            work = _seg("work", "Hill repeat", 1, None, wt)
            rec = _seg("recovery", "Jog down", 2, "Easy jog back to the start")
            cues = ["Strong arms, short steps", "Drive with your knees", "Last one — tall posture"]
        block = reps * (work["durationMin"] + rec["durationMin"]) - rec["durationMin"]
        cool = max(5, easy_min(km) - 10 - block)
        return ([_seg("warmup", "Warm up", 10, "Easy pace, loosen up")]
                + _repeat(reps, work, rec, cues)
                + [_seg("cooldown", "Cool down", cool, "Easy jog, then walk")])

    return [_seg("work", WTYPE_LABEL.get(wt, "Run"), easy_min(km), None, wt)]


@app.get("/workout/{workout_id}")
def get_workout(workout_id: str):
    d = daily()

    PHASE_TIPS = {
        "Follicular": "Your body is primed for intensity — a great window to push quality.",
        "Ovulatory":  "Peak energy window. You might feel stronger than usual — trust it.",
        "Luteal":     "HR may feel 3–5 bpm higher today. Go by effort, not numbers.",
        "Menstrual":  "Energy may be lower. It's okay to cut short or swap for easy if needed.",
    }

    # Accepts "Tue_2026-10-27" (as sent in /plan) or a bare day key like "Tue"
    day_key = workout_id.split("_")[0][:3].capitalize()
    if day_key not in DAY_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown workout '{workout_id}'")

    plan = _build_plan(d)
    paces = _paces_raw(plan["fp"])
    wo = next(w for w in plan["week"] if w["dayName"] == day_key)
    wt, dist = wo["type"], wo["distanceKm"]
    segs = _workout_segments(wt, dist, wo["details"], paces)
    total = sum(s["durationMin"] for s in segs)

    return {
        "id":             workout_id,
        "dateLabel":      day_key,
        "date":           wo["date"],
        "type":           wt,
        "distanceKm":     dist,
        "details":        wo["details"],
        "structureLabel": WTYPE_LABEL.get(wt, wt.capitalize()),
        "timeEstimate":   f"{int(total * 0.92)}–{int(math.ceil(total * 1.08))} min" if total else "—",
        "segments":       segs,
        "phase":          wo["phase"],
        "phaseTip":       PHASE_TIPS.get(wo["phase"], "Listen to your body and adapt as needed."),
        "override":       None,
    }


# ─── /debrief ─────────────────────────────────────────────────────────────────
def _read_debriefs() -> list:
    if not os.path.exists(DEBRIEF_PATH):
        return []
    try:
        with open(DEBRIEF_PATH) as f:
            existing = json.load(f)
    except (OSError, ValueError):
        return []
    return existing if isinstance(existing, list) else []


@app.post("/debrief")
def save_debrief(debrief: dict):
    """Save workout debrief to JSON file."""
    existing = _read_debriefs()
    existing.append(dict(debrief, savedAt=pd.Timestamp.now().isoformat(timespec="seconds")))
    tmp = DEBRIEF_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    os.replace(tmp, DEBRIEF_PATH)
    return {"status": "saved", "count": len(existing)}


@app.get("/debriefs")
def list_debriefs():
    return {"debriefs": _read_debriefs()}


# ─── /health ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    d = daily()
    r = runs()
    return {
        "status":        "ok",
        "daily_rows":    len(d),
        "run_rows":      len(r),
        "latest_date":   str(d.iloc[-1]["Date"].date()) if len(d) else None,
        "current_phase": _resolve_phase(d),
    }
