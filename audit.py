"""
Pipeline audit log — a JSON-serialisable snapshot of what the engine produced.

Called by engineering.py after run_pulsa_engine() and written to audit.json,
so a stale export or a broken column is visible without opening the Excel files.
"""

from datetime import date

import pandas as pd

KEY_COLUMNS = [
    "restingHeartRate", "total_sleep_hours", "readiness_score", "cycleDay", "cyclePhase",
    "load_3d", "acwr", "distance_km", "avgHr", "pace", "efficiency", "performance_smoothed",
]


def build_audit_log(df: pd.DataFrame, outputs: dict | None = None) -> dict:
    outputs = outputs or {}
    dates = pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.Series(dtype="datetime64[ns]")
    latest = dates.max().date() if len(dates) else None

    coverage = {
        c: round(float(df[c].notna().mean()), 3)
        for c in KEY_COLUMNS if c in df.columns
    }
    missing = [c for c in KEY_COLUMNS if c not in df.columns]

    run_days = int((pd.to_numeric(df["distance_km"], errors="coerce") > 0).sum()) if "distance_km" in df.columns else 0
    phases = (
        {str(k): int(v) for k, v in df["cyclePhase"].value_counts().items()}
        if "cyclePhase" in df.columns else {}
    )

    windows = outputs.get("training_windows") or {}
    return {
        "generated_on": date.today().isoformat(),
        "rows": int(len(df)),
        "date_range": {
            "start": dates.min().date().isoformat() if len(dates) else None,
            "end": latest.isoformat() if latest else None,
            "lag_days": (date.today() - latest).days if latest else None,
        },
        "run_days": run_days,
        "phase_counts": phases,
        "column_coverage": coverage,
        "missing_columns": missing,
        "training_windows_status": windows.get("status"),
        "insight_phases": sorted((outputs.get("insights") or {}).keys()),
    }
