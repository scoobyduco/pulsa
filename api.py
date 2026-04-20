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
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Pulsa API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten after testing
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data loading ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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

    return daily, running


_daily_df, _running_df = _load()


def daily() -> pd.DataFrame:
    return _daily_df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")


def runs() -> pd.DataFrame:
    return _running_df.sort_values("Date")


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


def _latest_row(df: pd.DataFrame) -> pd.Series:
    if len(df) == 0:
        return pd.Series(dtype=object)
    return df.iloc[-1]


def _resolve_phase(df: pd.DataFrame) -> str:
    """Get current cycle phase from latest data row."""
    for col in ("cyclePhase", "cycle_phase"):
        if col in df.columns:
            recent = df.dropna(subset=[col]).tail(14)
            for _, row in recent.iloc[::-1].iterrows():
                v = str(row[col]).strip()
                if v in PHASE_ORDER:
                    return v
    # Fallback: compute from user_settings.json if available
    settings_path = os.path.join(BASE_DIR, "user_settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            s = json.load(f)
        lps_str = s.get("cycle", {}).get("last_period_start")
        cl = int(s.get("cycle", {}).get("cycle_length", 28))
        if lps_str:
            lps = date.fromisoformat(lps_str)
            cd = (date.today() - lps).days % cl + 1
            if cd <= 5:
                return "Menstrual"
            if cd <= int(cl * 0.5):
                return "Follicular"
            if cd <= int(cl * 0.65):
                return "Ovulatory"
            return "Luteal"
    return "Follicular"


def _resolve_cycle_day(df: pd.DataFrame) -> int:
    for col in ("cycleDay", "cycle_day"):
        if col in df.columns:
            recent = df.dropna(subset=[col]).tail(5)
            if len(recent):
                return int(float(recent.iloc[-1][col]))
    settings_path = os.path.join(BASE_DIR, "user_settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            s = json.load(f)
        lps_str = s.get("cycle", {}).get("last_period_start")
        cl = int(s.get("cycle", {}).get("cycle_length", 28))
        if lps_str:
            lps = date.fromisoformat(lps_str)
            return (date.today() - lps).days % cl + 1
    return 14


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


def _cycle_length(df: pd.DataFrame) -> int:
    for col in ("actualCycleLength", "applicableMenstrualCycleLength"):
        if col in df.columns:
            v = df[col].dropna().tail(1)
            if len(v) and 21 <= float(v.iloc[0]) <= 45:
                return int(round(float(v.iloc[0])))
    settings_path = os.path.join(BASE_DIR, "user_settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            return int(json.load(f).get("cycle", {}).get("cycle_length", 28))
    return 28


# ─── /today ───────────────────────────────────────────────────────────────────
@app.get("/today")
def get_today():
    d = daily()

    latest = _latest_row(d)
    phase = _resolve_phase(d)
    cycle_day = _resolve_cycle_day(d)
    cl = _cycle_length(d)
    bounds = _phase_day_bounds(cl)

    # Metrics
    rhr_today = _safe(latest.get("restingHeartRate") or latest.get("resting_hr"))
    readiness = _safe(latest.get("readiness_score"))
    sleep_h   = _safe(latest.get("total_sleep_hours"))

    baseline_30 = d.tail(30)
    rhr_base  = float(baseline_30["restingHeartRate"].mean()) if "restingHeartRate" in baseline_30.columns else None
    rdy_base  = float(baseline_30["readiness_score"].mean())  if "readiness_score" in baseline_30.columns else None
    hr_delta  = (float(rhr_today) - rhr_base) if rhr_today and rhr_base else 0.0
    rdy_delta = (float(readiness) - rdy_base) if readiness and rdy_base else 0.0

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

    # Recommendation
    phase_rec = {
        "Menstrual":  ("easy",     "Easy run",    5.0, 35, "Easy",      "Hormones are at their lowest — go easy and protect your energy."),
        "Follicular": ("tempo",    "Tempo run",   8.0, 45, "Hard",      "Estrogen is rising — a great window to push quality."),
        "Ovulatory":  ("interval", "Intervals",   6.0, 40, "Very hard", "Peak estrogen — your strongest window. Go for it."),
        "Luteal":     ("easy",     "Steady run",  7.0, 45, "Moderate",  "Progesterone raises your HR — go by effort, not pace."),
    }
    rec_type, rec_title, rec_dist, rec_dur, rec_effort, rec_body = phase_rec[phase]

    # Recovery trend (7 days)
    trend_7 = d.tail(7)
    recovery_trend = []
    for _, row in trend_7.iterrows():
        ph = str(row.get("cyclePhase") or row.get("cycle_phase") or phase)
        rdy = _safe(row.get("readiness_score"), 60)
        recovery_trend.append({
            "date":      str(pd.Timestamp(row["Date"]).date()),
            "readiness": float(rdy),
            "phase":     ph,
        })

    # Trend insight
    rdys = [x["readiness"] for x in recovery_trend if x["readiness"]]
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
    cur_start, cur_end = bounds[phase]
    days_to_next = max(1, cur_end - cycle_day + 1)
    next_blurbs = {
        "Menstrual":  "Energy often dips as hormones reset.",
        "Follicular": "Energy often rises — good time to build.",
        "Ovulatory":  "Strength and speed can feel sharper.",
        "Luteal":     "Effort may feel higher at the same pace.",
    }

    # Greeting name
    name = "Runner"
    settings_path = os.path.join(BASE_DIR, "user_settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            name = json.load(f).get("profile", {}).get("name", "Runner") or "Runner"

    return {
        "greetingName": name,
        "cycleDay":     cycle_day,
        "cyclePhase":   phase,
        "phaseColor":   PHASE_COLORS[phase],
        "todayDate":    date.today().strftime("%-d %b"),
        "phaseNote":    next_blurbs.get(phase, ""),
        "recommendation": {
            "type":        rec_type,
            "title":       rec_title,
            "distanceKm":  rec_dist,
            "durationMin": rec_dur,
            "effortLevel": rec_effort,
            "body":        rec_body,
        },
        "metrics": {
            "restingHr":        int(rhr_today) if rhr_today else None,
            "restingHrContext": hr_ctx(hr_delta),
            "sleepHours":       round(float(sleep_h), 1) if sleep_h else None,
            "sleepContext":     sleep_ctx(sleep_h),
            "readinessScore":   int(readiness) if readiness else None,
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
    d = daily()
    phase = _resolve_phase(d)
    cycle_day = _resolve_cycle_day(d)
    cl = _cycle_length(d)

    try:
        import sys
        sys.path.insert(0, BASE_DIR)
        from plan_engine import generate_plan

        settings_path = os.path.join(BASE_DIR, "user_settings.json")
        s = {}
        if os.path.exists(settings_path):
            with open(settings_path) as f:
                s = json.load(f)

        tr = s.get("training", {})
        has_race  = tr.get("has_race", False)
        race_dist = tr.get("race_dist_sel", "10 km")
        baseline  = float(tr.get("baseline_km", 25))
        rpw       = int(tr.get("days_per_week", 4))
        goal_time = tr.get("goal_time", "") or ""

        dist_map = {"5 km": "5k", "10 km": "10k",
                    "Half marathon (21 km)": "21k", "Marathon (42 km)": "42k"}
        goal_type = dist_map.get(race_dist, "10k") if has_race else "maintenance"

        race_date_obj = None
        rd = tr.get("race_date")
        if has_race and rd:
            try:
                race_date_obj = date.fromisoformat(str(rd))
            except Exception:
                pass

        lps = None
        lps_str = s.get("cycle", {}).get("last_period_start")
        if lps_str:
            try:
                lps = date.fromisoformat(lps_str)
            except Exception:
                pass
        if lps is None and cycle_day >= 1:
            lps = date.today() - timedelta(days=cycle_day - 1)

        fp = generate_plan(
            goal_type=goal_type,
            goal_time_str=goal_time,
            baseline_km=baseline,
            runs_per_week=rpw,
            race_date=race_date_obj,
            today=date.today(),
            last_period_start=lps,
        )

        cur_week = fp["weeks"][0] if fp.get("weeks") else {}
        days_map = cur_week.get("days", {})
        total_km = float(cur_week.get("total_km", baseline))

        WTYPE_EMOJI = {
            "easy": "🏃‍♀️", "tempo": "⚡", "intervals": "🔥",
            "long run": "🛤️", "rest": "🧘",
        }

        week = []
        monday = date.today() - timedelta(days=date.today().weekday())
        for i, day_key in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
            wo = days_map.get(day_key, {"type": "rest", "km": 0})
            wt = str(wo.get("type", "rest")).lower()
            day_date = monday + timedelta(days=i)
            day_cd = ((day_date - lps).days % cl + 1) if lps else cycle_day
            bounds = _phase_day_bounds(cl)
            day_phase = "Follicular"
            for ph, (ps, pe) in bounds.items():
                if ps <= day_cd <= pe:
                    day_phase = ph
                    break
            week.append({
                "date":       str(day_date),
                "dayName":    day_key,
                "type":       wt,
                "emoji":      WTYPE_EMOJI.get(wt, "🏃‍♀️"),
                "distanceKm": float(wo.get("km", 0) or 0),
                "phase":      day_phase,
                "phaseColor": PHASE_COLORS[day_phase],
                "isRest":     wt == "rest",
            })

        run_count = sum(1 for w in week if not w["isRest"])

        goal_card = {
            "hasRace":  has_race,
            "raceDist": race_dist,
            "raceDate": tr.get("race_date", ""),
            "goalTime": goal_time,
            "weeksAway": (
                max(0, (race_date_obj - date.today()).days // 7)
                if race_date_obj else 0
            ),
        }

        return {
            "goalCard": goal_card,
            "week":     week,
            "totals":   {"totalKm": round(total_km, 1), "runCount": run_count},
        }

    except Exception as e:
        return {
            "goalCard": {"hasRace": False, "raceDist": "10 km", "raceDate": "", "goalTime": "", "weeksAway": 0},
            "week": [],
            "totals": {"totalKm": 0, "runCount": 0},
            "error": str(e),
        }


# ─── /insights ────────────────────────────────────────────────────────────────
@app.get("/insights")
def get_insights():
    d = daily()
    r = runs()

    cl = _cycle_length(d)
    bounds = _phase_day_bounds(cl)

    performance_rhythm = []
    perf_col = None
    for c in ("performance_smoothed", "performance_score"):
        if c in d.columns and d[c].notna().any():
            perf_col = c
            break

    if perf_col and "cycleDay" in d.columns:
        sub = d.dropna(subset=["cycleDay", perf_col]).copy()
        sub["cd"] = sub["cycleDay"].astype(float).clip(1, 28).astype(int)
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
            z2 = r[(r["avgHr"].between(120, 155))].dropna(subset=["cyclePhase", pace_col])
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

    n_cycles = max(1, len(d) // _cycle_length(d)) if len(d) > 0 else 0
    n_runs   = len(r)

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
    phase = _resolve_phase(d)
    cycle_day = _resolve_cycle_day(d)
    cl = _cycle_length(d)
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
    phase = KEY_MAP.get(key.lower(), "Follicular")
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
        "key":          key,
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
@app.get("/workout/{workout_id}")
def get_workout(workout_id: str):
    d = daily()
    phase = _resolve_phase(d)

    PHASE_TIPS = {
        "Follicular": "Your body is primed for intensity. Push the intervals — this is your window.",
        "Ovulatory":  "Peak energy window. You might feel stronger than usual — trust it.",
        "Luteal":     "HR may feel 3–5 bpm higher today. Go by effort, not numbers.",
        "Menstrual":  "Energy may be lower. It's okay to cut short or swap for easy if needed.",
    }

    parts = workout_id.split("_")
    day_key = parts[0] if parts else "Mon"

    WTYPE_SEGMENTS = {
        "tempo": [
            {"kind": "warmup",   "title": "Warm up",        "durationMin": 10, "cue": "Easy pace, find your rhythm",    "color": "#D3D1C7"},
            {"kind": "work",     "title": "Tempo interval", "durationMin": 8,  "cue": "Comfortably hard",               "color": "#EF9F27"},
            {"kind": "recovery", "title": "Recovery jog",   "durationMin": 2,  "cue": "Slow down, let HR drop",         "color": "#5DCAA5"},
            {"kind": "work",     "title": "Tempo interval", "durationMin": 8,  "cue": "Same effort as first",           "color": "#EF9F27"},
            {"kind": "recovery", "title": "Recovery jog",   "durationMin": 2,  "cue": None,                             "color": "#5DCAA5"},
            {"kind": "work",     "title": "Tempo interval", "durationMin": 8,  "cue": "Last one — hold steady",         "color": "#EF9F27"},
            {"kind": "cooldown", "title": "Cool down",      "durationMin": 8,  "cue": "Easy jog, then walk",            "color": "#D3D1C7"},
        ],
        "intervals": [
            {"kind": "warmup",   "title": "Warm up",       "durationMin": 10, "cue": "Easy pace, loosen up",             "color": "#D3D1C7"},
            {"kind": "work",     "title": "Hard interval", "durationMin": 4,  "cue": "Hard effort — 85–90% max HR",     "color": "#EF9F27"},
            {"kind": "recovery", "title": "Recovery",      "durationMin": 2,  "cue": "Walk or jog slowly",               "color": "#5DCAA5"},
            {"kind": "work",     "title": "Hard interval", "durationMin": 4,  "cue": "Same effort — don't start fast",  "color": "#EF9F27"},
            {"kind": "recovery", "title": "Recovery",      "durationMin": 2,  "cue": None,                               "color": "#5DCAA5"},
            {"kind": "work",     "title": "Hard interval", "durationMin": 4,  "cue": "Dig deep",                         "color": "#EF9F27"},
            {"kind": "cooldown", "title": "Cool down",     "durationMin": 8,  "cue": "Easy jog, then walk",              "color": "#D3D1C7"},
        ],
        "long run": [
            {"kind": "warmup",   "title": "Warm up",    "durationMin": 10, "cue": "Start slow, let your body wake up",   "color": "#D3D1C7"},
            {"kind": "work",     "title": "Steady run", "durationMin": 60, "cue": "Find a sustainable rhythm",           "color": "#7F77DD"},
            {"kind": "cooldown", "title": "Cool down",  "durationMin": 5,  "cue": "Slow to a jog, then walk",            "color": "#D3D1C7"},
        ],
        "easy": [
            {"kind": "work", "title": "Easy run", "durationMin": 35, "cue": "Conversational pace — you should be able to chat", "color": "#5DCAA5"},
        ],
    }

    try:
        plan_data = get_plan()
        week = plan_data.get("week", [])
        wo = next((w for w in week if w["dayName"] == day_key), None)
        if wo:
            wt = wo["type"]
            dist = wo["distanceKm"]
            segs = WTYPE_SEGMENTS.get(wt, WTYPE_SEGMENTS["easy"])
            label_map = {"easy": "Easy run", "tempo": "Tempo run", "intervals": "Intervals", "long run": "Long run", "rest": "Rest"}
            return {
                "id":             workout_id,
                "dateLabel":      day_key,
                "type":           wt,
                "distanceKm":     dist,
                "structureLabel": label_map.get(wt, wt.capitalize()),
                "timeEstimate":   f"{int(dist*5.5)}–{int(dist*6.5)} min",
                "segments":       segs,
                "phaseTip":       PHASE_TIPS.get(phase, "Listen to your body and adapt as needed."),
                "override":       None,
            }
    except Exception:
        pass

    return {
        "id":             workout_id,
        "dateLabel":      day_key,
        "type":           "easy",
        "distanceKm":     5.0,
        "structureLabel": "Easy run",
        "timeEstimate":   "28–33 min",
        "segments":       WTYPE_SEGMENTS["easy"],
        "phaseTip":       PHASE_TIPS.get(phase, "Listen to your body."),
        "override":       None,
    }


# ─── /debrief (POST) ──────────────────────────────────────────────────────────
@app.post("/debrief")
def save_debrief(debrief: dict):
    """Save workout debrief to JSON file."""
    path = os.path.join(BASE_DIR, "workout_debriefs.json")
    existing = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    if not isinstance(existing, list):
        existing = []
    existing.append(debrief)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    return {"status": "saved"}


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
