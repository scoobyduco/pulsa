"""
plan_engine.py — Training plan generation for Pulsa.

Architecture:
  Layer 1  Input / goal params
  Layer 2  VDOT / fitness scoring (Jack Daniels)
  Layer 3  80/20 polarized distribution
  Layer 4  Periodization by goal
  Layer 5  Menstrual cycle integration
  Layer 6  Load management (ACWR)
  Layer 7  Weekly workout structure
  Layer 8  Workout type definitions
  Layer 9  Daily readiness (consumed by UI, not generated here)
  Layer 10 Output: full multi-week plan dict
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta
from typing import Optional


# ─── VDOT Engine ───────────────────────────────────────────────────────────────

def calculate_vdot(time_seconds: int, distance_km: float) -> float:
    """Calculate VDOT from a race result (Jack Daniels formula)."""
    t = max(1.0, time_seconds / 60.0)        # minutes
    v = (distance_km * 1000.0) / t           # m/min
    fr = (0.8
          + 0.1894393 * math.exp(-0.012778 * t)
          + 0.2989558 * math.exp(-0.1932605 * t))
    vo2 = -4.60 + 0.182258 * v + 0.000104 * v * v
    return round(vo2 / fr, 1)


def estimate_vdot_from_volume(baseline_km: float) -> float:
    """Rough VDOT estimate when no race time is available."""
    return round(28 + baseline_km * 0.4, 1)


# ─── Pace Zone Tables ──────────────────────────────────────────────────────────

# Derived from Jack Daniels' Running Formula pace tables.
# Format: vdot → (easy_slow, easy_fast, mp, tempo, interval, rep) all in sec/km
_VDOT_PACES: dict[int, tuple[int, ...]] = {
    30: (490, 456, 430, 404, 372, 345),
    35: (456, 424, 382, 358, 330, 306),
    40: (409, 378, 341, 320, 295, 273),
    45: (371, 344, 309, 290, 267, 247),
    50: (340, 316, 282, 265, 244, 226),
    55: (314, 292, 259, 243, 224, 208),
    60: (292, 272, 240, 225, 207, 192),
    65: (274, 254, 222, 209, 192, 178),
    70: (259, 239, 206, 195, 179, 166),
}


def _fmt_pace(sec: float) -> str:
    s = int(round(sec))
    return f"{s // 60}:{s % 60:02d}"


def vdot_to_paces(vdot: float, goal_cat: str = "42k") -> dict:
    """Return training pace zones from VDOT, labelled by goal type."""
    v = min(max(float(vdot), 30), 70)
    keys = sorted(_VDOT_PACES.keys())
    lo = max(k for k in keys if k <= v)
    hi = min(k for k in keys if k >= v)

    if lo == hi:
        vals = tuple(float(x) for x in _VDOT_PACES[lo])
    else:
        t = (v - lo) / (hi - lo)
        lv, hv = _VDOT_PACES[lo], _VDOT_PACES[hi]
        vals = tuple(lv[i] + t * (hv[i] - lv[i]) for i in range(6))

    es, ef, mp, tempo, iv, rep = vals

    # Tempo range: ±5s/km around center
    tempo_range = f"{_fmt_pace(tempo - 5)}–{_fmt_pace(tempo + 5)}"
    iv_range    = f"{_fmt_pace(iv - 5)}–{_fmt_pace(iv + 5)}"
    rep_range   = f"{_fmt_pace(rep - 5)}–{_fmt_pace(rep + 5)}"
    easy_range  = f"{_fmt_pace(ef)}–{_fmt_pace(es)}"

    if goal_cat in ("42k", "21k"):
        return {
            "easy": easy_range,
            "zone2": _fmt_pace(mp),
            "zone3": tempo_range,
            "zone4": iv_range,
            "labels": ("Easy", "Marathon", "Tempo", "Interval"),
            "_raw": {"easy_fast": ef, "easy_slow": es, "mp": mp,
                     "tempo": tempo, "interval": iv, "rep": rep},
        }
    elif goal_cat in ("10k", "5k"):
        return {
            "easy": easy_range,
            "zone2": tempo_range,
            "zone3": iv_range,
            "zone4": rep_range,
            "labels": ("Easy", "Tempo", "Interval", "Rep"),
            "_raw": {"easy_fast": ef, "easy_slow": es, "mp": mp,
                     "tempo": tempo, "interval": iv, "rep": rep},
        }
    else:  # maintenance
        return {
            "easy": easy_range,
            "zone2": tempo_range,
            "zone3": "Varied",
            "zone4": f"{_fmt_pace(es + 20)}–{_fmt_pace(es - 20)}",
            "labels": ("Easy", "Tempo", "Fartlek", "Long"),
            "_raw": {"easy_fast": ef, "easy_slow": es, "mp": mp,
                     "tempo": tempo, "interval": iv, "rep": rep},
        }


# ─── Menstrual Cycle Integration ───────────────────────────────────────────────

CYCLE_PHASE_INFO: dict[str, dict] = {
    "menstrual": {
        "emoji": "🩸",
        "name": "Menstrual",
        "days": "Days 1–5",
        "modifications": [
            ("mod-down",   "If cramps or fatigue hit, convert tempo to easy run"),
            ("mod-down",   "Cut long run by 2–4 km if energy is genuinely low"),
            ("mod-adjust", "Iron stores deplete — prioritize iron-rich foods with vitamin C"),
        ],
    },
    "follicular": {
        "emoji": "🌱",
        "name": "Follicular",
        "days": "Days 6–13",
        "modifications": [
            ("mod-up",   "Energy peaks — extend tempo by 1–2 km if feeling strong"),
            ("mod-up",   "Add a 4th easy run (5–6 km) if body feels good"),
            ("mod-up",   "Push long run pace slightly or add faster finish"),
        ],
    },
    "ovulatory": {
        "emoji": "🌸",
        "name": "Ovulatory",
        "days": "Days 14–16",
        "modifications": [
            ("mod-up",     "Peak power window — best time for key sessions and PRs"),
            ("mod-up",     "Can push harder than prescribed; trust the feeling"),
            ("mod-adjust", "Ligament laxity slightly elevated — warm up thoroughly"),
        ],
    },
    "luteal": {
        "emoji": "⚡",
        "name": "Luteal",
        "days": "Days 17–28",
        "modifications": [
            ("mod-adjust", "HR runs 3–4 bpm higher — go by RPE 7, not pace"),
            ("mod-adjust", "Higher calorie burn — increase fueling, extra gel on long runs"),
            ("mod-down",   "If energy crashes, swap quality session for easy run"),
        ],
    },
}

_MOD_COLORS: dict[str, tuple[str, str]] = {
    "mod-up":     ("#E1F5EE", "#0F6E56"),
    "mod-adjust": ("#FAEEDA", "#854F0B"),
    "mod-down":   ("#FAECE7", "#993C1D"),
}


def predict_cycle_phase(
    week_start: date,
    last_period_start: Optional[date] = None,
    cycle_length: int = 28,
) -> str:
    if last_period_start is None:
        return "follicular"
    days = (week_start - last_period_start).days % cycle_length
    cycle_day = days + 1
    if cycle_day <= 5:
        return "menstrual"
    if cycle_day <= 13:
        return "follicular"
    if cycle_day <= 16:
        return "ovulatory"
    return "luteal"


# ─── ACWR ──────────────────────────────────────────────────────────────────────

def calculate_acwr(acute_km: float, chronic_km: float) -> float:
    if chronic_km <= 0:
        return 1.0
    return round(acute_km / chronic_km, 2)


def _ewma(chronic: float, acute: float, n: int = 4) -> float:
    lam = 2 / (n + 1)
    return acute * lam + chronic * (1 - lam)


# ─── Workout Type Definitions ──────────────────────────────────────────────────

WORKOUT_COLORS: dict[str, str] = {
    "easy":      "#5DCAA5",
    "tempo":     "#EF9F27",
    "interval":  "#7F77DD",
    "long":      "#378ADD",
    "mp_long":   "#7F77DD",
    "fartlek":   "#7F77DD",
    "hills":     "#D85A30",
    "race":      "#D4537E",
    "recovery":  "#5DCAA5",
    "rest":      "#D3D1C7",
}

WORKOUT_DISPLAY: dict[str, str] = {
    "easy":     "Easy",
    "tempo":    "Tempo",
    "interval": "Intervals",
    "long":     "Long",
    "mp_long":  "MP Run",
    "fartlek":  "Fartlek",
    "hills":    "Hills",
    "race":     "Race",
    "recovery": "Recovery",
    "rest":     "Rest",
}


def _rest() -> dict:
    return {"type": "rest", "type_raw": "rest", "km": 0, "details": "", "is_key": False}


def _easy(km: float, pace: str) -> dict:
    return {"type": "easy", "type_raw": "easy", "km": round(km, 1),
            "details": f"{km:.0f} km @ {pace}", "is_key": False}


def _recovery(km: float, pace: str) -> dict:
    return {"type": "easy", "type_raw": "recovery", "km": round(km, 1),
            "details": f"{km:.0f} km recovery @ {pace}", "is_key": False}


def _tempo(total_km: float, tempo_km: float, easy_pace: str, tempo_pace: str) -> dict:
    tp = tempo_pace.split("–")[0] if "–" in tempo_pace else tempo_pace
    return {"type": "tempo", "type_raw": "tempo", "km": round(total_km, 1),
            "details": f"{total_km:.0f} km w/{tempo_km:.0f} km @{tp}", "is_key": True}


def _interval(total_km: float, reps: int, rep_km: float, iv_pace: str) -> dict:
    ip = iv_pace.split("–")[0] if "–" in iv_pace else iv_pace
    m = int(rep_km * 1000)
    return {"type": "interval", "type_raw": "interval", "km": round(total_km, 1),
            "details": f"{reps}×{m} m @{ip}", "is_key": True}


def _long(km: float, pace: str) -> dict:
    ep = pace.split("–")[-1] if "–" in pace else pace
    return {"type": "long", "type_raw": "long", "km": round(km, 1),
            "details": f"{km:.0f} km @ {ep}", "is_key": False}


def _mp_long(total_km: float, mp_km: float, mp_pace: str) -> dict:
    return {"type": "long", "type_raw": "mp_long", "km": round(total_km, 1),
            "details": f"{total_km:.0f} km w/{mp_km:.0f} km @{mp_pace}", "is_key": True}


def _fartlek(km: float) -> dict:
    return {"type": "fartlek", "type_raw": "fartlek", "km": round(km, 1),
            "details": f"{km:.0f} km fartlek w/ surges", "is_key": True}


def _hills(km: float) -> dict:
    return {"type": "hills", "type_raw": "hills", "km": round(km, 1),
            "details": f"{km:.0f} km w/ hill repeats", "is_key": True}


def _race_day(km: float) -> dict:
    return {"type": "race", "type_raw": "race", "km": round(km, 1),
            "details": "Race day!", "is_key": True}


# ─── Day Slot Templates ────────────────────────────────────────────────────────

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

_SLOTS: dict[int, list[str]] = {
    3: ["Tue", "Thu", "Sun"],
    4: ["Tue", "Thu", "Sat", "Sun"],
    5: ["Mon", "Tue", "Thu", "Sat", "Sun"],
    6: ["Mon", "Tue", "Wed", "Thu", "Sat", "Sun"],
}


def _assign_days(runs_per_week: int, workouts: list[dict]) -> dict[str, dict]:
    slots = _SLOTS.get(min(max(runs_per_week, 3), 6), _SLOTS[3])
    day_map = {d: _rest() for d in DAYS}
    for i, wo in enumerate(workouts):
        if i < len(slots):
            wo = dict(wo, day=slots[i])
            day_map[slots[i]] = wo
    return day_map


# ─── Periodization ────────────────────────────────────────────────────────────

_PHASE_BADGE: dict[str, dict] = {
    "Build":   {"bg": "#E6F1FB", "text": "#185FA5"},
    "Peak":    {"bg": "#FAEEDA", "text": "#854F0B"},
    "Taper":   {"bg": "#E1F5EE", "text": "#0F6E56"},
    "Race":    {"bg": "#FBEAF0", "text": "#993556"},
    "Tempo":   {"bg": "#E1F5EE", "text": "#0F6E56"},
    "Fartlek": {"bg": "#EDE9FB", "text": "#534AB7"},
    "Hills":   {"bg": "#FAECE7", "text": "#993C1D"},
    "Long":    {"bg": "#E6F1FB", "text": "#185FA5"},
}

_MAINT_CYCLE = ["Tempo", "Fartlek", "Hills", "Long"]


def _periodize(total_weeks: int, goal_cat: str) -> list[str]:
    if goal_cat == "maintenance":
        return [_MAINT_CYCLE[i % 4] for i in range(total_weeks)]

    # For all race goals: always end with Race week, at least 1 Taper week.
    race_w  = 1
    taper_w = max(1, min(3, total_weeks // 5))
    remain  = total_weeks - race_w - taper_w

    if goal_cat == "5k":
        peak_w  = max(1, remain // 3)
        build_w = remain - peak_w
    elif goal_cat == "10k":
        peak_w  = max(1, remain // 3)
        build_w = remain - peak_w
    elif goal_cat == "21k":
        peak_w  = max(1, remain * 2 // 5)
        build_w = remain - peak_w
    else:  # 42k
        peak_w  = max(1, remain * 2 // 5)
        build_w = remain - peak_w

    return (["Build"] * max(0, build_w)
            + ["Peak"] * max(0, peak_w)
            + ["Taper"] * taper_w
            + ["Race"] * race_w)[:total_weeks]


def _volume_schedule(
    baseline: float,
    total_weeks: int,
    goal_cat: str,
    phases: list[str],
) -> list[int]:
    if goal_cat == "maintenance":
        mults = {"Tempo": 1.0, "Fartlek": 1.05, "Hills": 0.90, "Long": 1.10}
        return [round(baseline * mults.get(phases[i % 4], 1.0)) for i in range(total_weeks)]

    peak_mult = {"42k": 1.4, "21k": 1.3, "10k": 1.25, "5k": 1.2}.get(goal_cat, 1.25)
    peak_km   = min(baseline * peak_mult, 85 if goal_cat == "42k" else 65)
    build_end = max(1, sum(1 for p in phases if p == "Build"))

    vols: list[int] = []
    for i, ph in enumerate(phases):
        if ph == "Race":
            vols.append(round(peak_km * 0.40))
        elif ph == "Taper":
            taper_idx = [j for j, p in enumerate(phases) if p == "Taper"].index(i)
            mult = [0.78, 0.58, 0.42][min(taper_idx, 2)]
            vols.append(round(peak_km * mult))
        elif ph == "Peak":
            t = max(0, i - build_end) / max(1, sum(1 for p in phases if p in ("Peak", "Taper", "Race")) - 1)
            km = baseline + (peak_km - baseline) * (0.7 + 0.3 * min(t, 1))
            vols.append(round(km))
        else:  # Build
            t = (i + 1) / max(1, build_end)
            km = baseline + (peak_km - baseline) * t * 0.75
            if i > 0 and (i + 1) % 4 == 0:  # recovery week
                km *= 0.85
            vols.append(round(max(baseline, km)))
    return vols


# ─── Week Workout Generation ───────────────────────────────────────────────────

def _quality_session(
    phase: str,
    goal_cat: str,
    km: float,
    paces: dict,
    week_idx: int,
    primary: bool = True,
) -> dict:
    # Use raw paces: zone2–4 labels mean different things per goal (and "Varied" for maintenance)
    raw = paces.get("_raw", {})
    easy_p  = _fmt_pace(raw.get("easy_fast", 378))
    tempo_p = _fmt_pace(raw.get("tempo", 320))
    iv_p    = _fmt_pace(raw.get("interval", 295))

    if phase == "Taper":
        return _tempo(round(km), round(km * 0.4), easy_p, tempo_p)

    if phase in ("Tempo",):
        return _tempo(round(km), round(km * 0.4), easy_p, tempo_p)
    if phase == "Fartlek":
        return _fartlek(round(km))
    if phase == "Hills":
        return _hills(round(km))

    if goal_cat in ("42k", "21k"):
        if primary and week_idx % 2 == 0:
            return _interval(round(km), 5, 1.0, iv_p)
        return _tempo(round(km), round(km * 0.42), easy_p, tempo_p)
    elif goal_cat in ("10k", "5k"):
        reps = 6 if goal_cat == "5k" else 5
        rep_km = 0.8 if goal_cat == "5k" else 1.0
        if primary:
            return _interval(round(km), reps, rep_km, iv_p)
        return _tempo(round(km), round(km * 0.4), easy_p, tempo_p)

    # Default quality
    return _tempo(round(km), round(km * 0.4), easy_p, tempo_p)


def _build_week_workouts(
    phase: str,
    total_km: int,
    long_km: int,
    rpw: int,
    goal_cat: str,
    paces: dict,
    week_idx: int,
) -> list[dict]:
    raw     = paces.get("_raw", {})
    easy_p  = f"{_fmt_pace(raw.get('easy_fast', 378))}–{_fmt_pace(raw.get('easy_slow', 409))}"
    mp_p    = paces.get("zone2", "5:06")
    rem     = total_km - long_km
    workouts: list[dict] = []

    if rpw <= 3:
        easy_km = max(4, round(rem * 0.45))
        qual_km = max(5, rem - easy_km)
        workouts += [_easy(easy_km, easy_p),
                     _quality_session(phase, goal_cat, qual_km, paces, week_idx)]
    elif rpw == 4:
        e1 = max(4, round(rem * 0.30))
        q  = max(5, round(rem * 0.46))
        e2 = max(4, rem - e1 - q)
        workouts += [_easy(e1, easy_p),
                     _quality_session(phase, goal_cat, q, paces, week_idx, primary=True),
                     _easy(e2, easy_p)]
    elif rpw == 5:
        e1 = max(4, round(rem * 0.20))
        q1 = max(5, round(rem * 0.30))
        q2 = max(5, round(rem * 0.25))
        e2 = max(4, rem - e1 - q1 - q2)
        workouts += [_easy(e1, easy_p),
                     _quality_session(phase, goal_cat, q1, paces, week_idx, primary=True),
                     _quality_session(phase, goal_cat, q2, paces, week_idx, primary=False),
                     _easy(e2, easy_p)]
    else:  # 6
        e1 = max(4, round(rem * 0.15))
        q1 = max(5, round(rem * 0.25))
        e2 = max(4, round(rem * 0.12))
        q2 = max(5, round(rem * 0.22))
        e3 = max(4, rem - e1 - q1 - e2 - q2)
        workouts += [_easy(e1, easy_p),
                     _quality_session(phase, goal_cat, q1, paces, week_idx, primary=True),
                     _easy(e2, easy_p),
                     _quality_session(phase, goal_cat, q2, paces, week_idx, primary=False),
                     _easy(e3, easy_p)]

    # Long run (always last)
    if phase == "Race":
        workouts.append(_race_day(long_km))
    elif goal_cat in ("42k",) and phase == "Peak" and long_km >= 16:
        mp_km = round(long_km * 0.5)
        workouts.append(_mp_long(long_km, mp_km, mp_p))
    else:
        workouts.append(_long(long_km, easy_p))

    return workouts


# ─── Full Plan Generator ───────────────────────────────────────────────────────

_GOAL_NORM: dict[str, str] = {
    "5 km": "5k", "5k": "5k", "5km": "5k",
    "10 km": "10k", "10k": "10k", "10km": "10k",
    "half marathon (21 km)": "21k", "21k": "21k", "21km": "21k", "half marathon": "21k",
    "marathon (42 km)": "42k", "42k": "42k", "42km": "42k", "marathon": "42k",
    "maintenance": "maintenance", "no race": "maintenance",
}

_GOAL_LABEL: dict[str, str] = {
    "5k": "5K", "10k": "10K", "21k": "Half Marathon",
    "42k": "Marathon", "maintenance": "Maintenance",
}

_GOAL_DIST: dict[str, float] = {
    "5k": 5, "10k": 10, "21k": 21.1, "42k": 42.2,
}


def generate_plan(
    goal_type: str,
    goal_time_str: str = "",
    baseline_km: float = 30,
    runs_per_week: int = 3,
    current_vdot: Optional[float] = None,
    race_date: Optional[date] = None,
    today: Optional[date] = None,
    last_period_start: Optional[date] = None,
    cycle_length: int = 28,
) -> dict:
    """
    Generate a complete multi-week training plan.

    Returns a dict with keys:
      header, profile, paces, weeks, volumes, phases, goal_cat, total_weeks
    """
    if today is None:
        today = date.today()

    goal_cat = _GOAL_NORM.get(goal_type.strip().lower(), "maintenance")

    if current_vdot is None or current_vdot <= 0:
        current_vdot = estimate_vdot_from_volume(baseline_km)

    # Target VDOT from goal time
    target_vdot = current_vdot
    if goal_time_str and goal_cat != "maintenance":
        dist_km = _GOAL_DIST.get(goal_cat, 10)
        try:
            parts = str(goal_time_str).strip().split(":")
            if len(parts) == 2:
                t_sec = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                t_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                t_sec = 0
            if t_sec > 0:
                target_vdot = round(calculate_vdot(t_sec, dist_km), 1)
        except Exception:
            target_vdot = current_vdot + 3

    # Working VDOT: don't overreach
    working_vdot = min(current_vdot + 10, max(current_vdot, target_vdot))

    paces = vdot_to_paces(working_vdot, goal_cat)

    # Plan length
    if goal_cat == "maintenance":
        total_weeks = 4
    elif race_date:
        days_to = (race_date - today).days
        total_weeks = max(4, min(20, days_to // 7))
    else:
        total_weeks = {
            "5k": 6, "10k": 8, "21k": 12, "42k": 16,
        }.get(goal_cat, 8)

    phases  = _periodize(total_weeks, goal_cat)
    volumes = _volume_schedule(baseline_km, total_weeks, goal_cat, phases)

    # ACWR
    chronic = float(baseline_km)
    acwrs: list[float] = []
    for v in volumes:
        acwrs.append(calculate_acwr(v, chronic))
        chronic = _ewma(chronic, v)

    # Week start = Monday of current week
    week_start = today - timedelta(days=today.weekday())

    long_pcts = {3: 0.40, 4: 0.40, 5: 0.35, 6: 0.32}
    rpw = min(max(runs_per_week, 3), 6)

    weeks: list[dict] = []
    for i in range(total_weeks):
        ws   = week_start + timedelta(weeks=i)
        ph   = phases[i]
        vol  = volumes[i]
        acwr = acwrs[i]
        badge = _PHASE_BADGE.get(ph, {"bg": "#F1EFE8", "text": "#5F5E5A"})

        long_km = max(6, round(vol * long_pcts.get(rpw, 0.40)))
        workouts = _build_week_workouts(ph, vol, long_km, rpw, goal_cat, paces, i)
        days = _assign_days(rpw, workouts)

        cycle_ph   = predict_cycle_phase(ws, last_period_start, cycle_length)
        cycle_info = CYCLE_PHASE_INFO[cycle_ph]

        # Stats row copy
        focus_label = ph if goal_cat == "maintenance" else None

        weeks.append({
            "week_num":    i + 1,
            "week_start":  ws.isoformat(),
            "phase":       ph,
            "phase_badge": badge,
            "total_km":    vol,
            "acwr":        acwr,
            "long_km":     long_km,
            "focus":       focus_label,
            "cycle_phase": cycle_ph,
            "cycle_emoji": cycle_info["emoji"],
            "cycle_name":  cycle_info["name"],
            "cycle_days":  cycle_info["days"],
            "modifications": cycle_info["modifications"],
            "days":        days,
            "is_current":  i == 0,
        })

    # Header
    goal_label = _GOAL_LABEL.get(goal_cat, goal_type)
    if goal_cat == "maintenance":
        header_title    = "Maintenance plan — no race"
        header_subtitle = (f"Rolling 4-week cycle · {rpw} runs/week · "
                           "Optional cycle-aware modifications")
        header_gradient = "linear-gradient(135deg, #0F6E56 0%, #085041 100%)"
    else:
        time_label      = f" — {goal_time_str} goal" if goal_time_str else ""
        header_title    = f"{goal_label} training plan{time_label}"
        header_subtitle = (f"{total_weeks} weeks · {rpw} runs/week · "
                           "Optional cycle-aware modifications")
        header_gradient = "linear-gradient(135deg, #534AB7 0%, #3C3489 100%)"

    # Profile grid
    mp_raw = paces["_raw"].get("mp", paces["_raw"].get("tempo", 300))
    goal_pace_str = _fmt_pace(mp_raw) + "/km"
    peak_km = max(volumes) if volumes else int(baseline_km)

    if goal_cat == "maintenance":
        profile = {
            "box1_label": "Current VDOT", "box1_value": str(round(current_vdot, 1)),
            "box2_label": "Goal",          "box2_value": "Maintain",
            "box3_label": "Weekly target", "box3_value": f"{int(baseline_km)} km",
            "box4_label": "Structure",     "box4_value": "4-week rotation",
        }
    else:
        profile = {
            "box1_label": "Current VDOT", "box1_value": str(round(current_vdot, 1)),
            "box2_label": "Target VDOT",  "box2_value": str(round(target_vdot, 1)),
            "box3_label": "Goal pace",    "box3_value": goal_pace_str,
            "box4_label": "Volume build", "box4_value": f"{int(baseline_km)} → {peak_km} km",
        }

    return {
        "header":      {"title": header_title, "subtitle": header_subtitle, "gradient": header_gradient},
        "profile":     profile,
        "paces":       paces,
        "weeks":       weeks,
        "volumes":     volumes,
        "phases":      phases,
        "goal_cat":    goal_cat,
        "total_weeks": total_weeks,
        "working_vdot": working_vdot,
    }


# ─── Legacy Compat: validate_goal & compute_weekly_km ─────────────────────────
# Moved from pulsa_engine.py so ALL plan logic lives here.

def validate_goal(goal: dict) -> tuple[bool, str]:
    """Check for unrealistic combinations (e.g. marathon in 50 min). Returns (valid, message)."""
    if not goal.get("has_race"):
        return True, ""
    race_km = goal.get("race_km", 10)
    goal_time = goal.get("goal_time", "")
    if not goal_time or not str(goal_time).strip():
        return True, ""
    s = str(goal_time).strip().lower()
    minutes = None
    if "h" in s or ":" in s:
        parts = re.split(r"h|:", s)
        if len(parts) >= 2:
            try:
                h = float(parts[0].strip())
                m = float(parts[1].strip()) if len(parts) > 1 else 0
                minutes = h * 60 + m
            except (ValueError, IndexError):
                pass
    else:
        m = re.search(r"(\d+)\s*(?:min|m|mins)?", s)
        if m:
            try:
                minutes = float(m.group(1))
            except (ValueError, AttributeError):
                pass
    if minutes is None or minutes <= 0:
        return True, ""
    pace_min_per_km = minutes / race_km
    if race_km >= 42 and pace_min_per_km < 3.5:
        return False, "Marathon in that time is very ambitious. Consider adjusting your goal time."
    if race_km >= 21 and pace_min_per_km < 3.0:
        return False, "Half marathon in that time is elite-level. Double-check your goal."
    return True, ""


def compute_weekly_km(goal: dict, current_phase: str = "Unknown") -> int:
    """
    Derive recommended weekly km from goal and context.
    Goal dict: has_race, race_km, experience, baseline_km, days_per_week
    """
    has_race = goal.get("has_race", False)
    race_km = goal.get("race_km", 10)
    experience = goal.get("experience", "Intermediate")
    baseline_km = goal.get("baseline_km")
    days_per_week = goal.get("days_per_week", 4)

    if has_race and race_km:
        if race_km <= 5:
            base = 25
        elif race_km <= 10:
            base = 30
        elif race_km <= 21:
            base = 40
        elif race_km <= 42:
            base = 50
        else:
            base = max(35, min(60, race_km))
    else:
        base = baseline_km if baseline_km and baseline_km > 0 else 25

    if experience == "Beginner":
        base *= 0.7
    elif experience == "Advanced":
        base *= 1.2

    if days_per_week <= 3:
        base *= 0.85
    elif days_per_week >= 5:
        base *= 1.1

    if current_phase == "Follicular" or current_phase == "Ovulatory":
        base *= 1.05
    elif current_phase in ("Luteal", "Menstrual"):
        base *= 0.95

    return int(max(5, min(120, round(base))))
