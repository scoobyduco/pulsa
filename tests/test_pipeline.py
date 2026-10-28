"""End-to-end run of engineering.main() on a small synthetic Garmin export."""

import json
import random
from datetime import date, datetime, timedelta

import pandas as pd

import engineering

DAYS = 90


def _write(folder, name, data):
    (folder / name).write_text(json.dumps(data))


def _fake_export(folder):
    rng = random.Random(0)
    start = date.today() - timedelta(days=DAYS - 1)
    days = [start + timedelta(days=i) for i in range(DAYS)]

    _write(folder, "x_UDSFile_1.json", [{
        "calendarDate": d.isoformat(),
        "totalKilocalories": 2200, "activeKilocalories": rng.randint(300, 900),
        "bmrKilocalories": 1400, "totalSteps": rng.randint(5000, 15000),
        "totalDistanceMeters": rng.randint(4000, 14000),
        "restingHeartRate": rng.randint(46, 54), "currentDayRestingHeartRate": rng.randint(46, 56),
        "minHeartRate": 44, "maxHeartRate": 170,
        "bodyBattery": {"chargedValue": 60, "drainedValue": 55},
        "respiration": {"avgWakingRespirationValue": 14},
    } for d in days])

    _write(folder, "x_sleepData.json", [{
        "calendarDate": d.isoformat(),
        "deepSleepSeconds": 5400, "remSleepSeconds": 6000,
        "lightSleepSeconds": rng.randint(12000, 16000), "awakeSleepSeconds": 900,
        "averageRespiration": 14, "avgSleepStress": 20,
    } for d in days])

    # Only one period is logged, 90 days ago: everything past day 45 must lose its phase.
    _write(folder, "x_MenstrualCycles.json", [{
        "startDate": start.isoformat(),
        "actualCycleLength": 28, "actualPeriodLength": 5,
        "fertileWindowStart": 12, "fertileWindowLength": 6,
    }])

    runs = []
    for i, d in enumerate(days[::2]):
        km = rng.uniform(5, 15)
        secs = km * rng.uniform(300, 360)
        runs.append({
            "activityId": 1000 + i, "name": "Run", "sportType": "RUNNING",
            "startTimeLocal": int(datetime(d.year, d.month, d.day, 7).timestamp() * 1000),
            "duration": secs * 1000, "movingDuration": secs * 1000,
            "distance": km * 100000, "avgSpeed": 3.0, "elevationGain": 5000,
            "avgHr": rng.randint(135, 165), "maxHr": 180, "avgRunCadence": 170,
            "aerobicTrainingEffect": 3.0, "anaerobicTrainingEffect": 1.0,
        })
    _write(folder, "x_summarizedActivities.json", [{"summarizedActivitiesExport": runs}])


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    data = tmp_path / "garmin_data"
    data.mkdir()
    _fake_export(data)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engineering, "DATA_FOLDER", str(data))
    monkeypatch.delenv("GARMIN_EMAIL", raising=False)

    engineering.main()

    daily = pd.read_excel(tmp_path / "daily_engine_output.xlsx")
    running = pd.read_excel(tmp_path / "running_engine_output.xlsx")
    assert len(daily) >= DAYS - 1
    assert {"readiness_score", "load_3d", "cyclePhase"} <= set(daily.columns)
    assert {"performance_smoothed", "efficiency", "session_flag"} <= set(running.columns)
    assert (running["distance_km"] > 0).sum() == len(range(0, DAYS, 2))

    # Days beyond MAX_CYCLE_DAY after the last logged period have no phase
    assert daily["cycleDay"].max() == 45
    assert daily["cycleDay"].isna().sum() >= DAYS - 46
    assert daily.loc[daily["cycleDay"].isna(), "cyclePhase"].isna().all()

    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["rows"] == len(running)
    assert audit["date_range"]["lag_days"] <= 1
