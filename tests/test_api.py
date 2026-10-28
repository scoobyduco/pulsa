from datetime import date

import pytest
from fastapi.testclient import TestClient

import api

TODAY = date(2026, 10, 28)  # a Wednesday


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_today", lambda: TODAY)
    monkeypatch.setattr(api, "DEBRIEF_PATH", str(tmp_path / "debriefs.json"))
    return TestClient(api.app)


def test_all_endpoints_respond(client):
    for path in ["/health", "/today", "/plan", "/insights", "/learn",
                 "/learn/phase/luteal", "/workout/Thu", "/debriefs"]:
        assert client.get(path).status_code == 200, path


def test_unknown_ids_404(client):
    assert client.get("/learn/phase/bogus").status_code == 404
    assert client.get("/workout/zzz").status_code == 404


def test_cycle_state_consistent_across_endpoints(client):
    today = client.get("/today").json()
    plan = client.get("/plan").json()
    learn = client.get("/learn").json()

    todays_slot = next(w for w in plan["week"] if w["date"] == TODAY.isoformat())
    assert todays_slot["phase"] == today["cyclePhase"]
    assert learn["currentPhase"]["name"] == today["cyclePhase"]
    assert learn["currentPhase"]["cycleDay"] == today["cycleDay"]
    assert 1 <= today["cycleDay"] <= learn["currentPhase"]["cycleLength"]


def test_today_recommendation_follows_plan(client):
    today = client.get("/today").json()
    plan = client.get("/plan").json()
    slot = next(w for w in plan["week"] if w["date"] == TODAY.isoformat())
    rec = today["recommendation"]
    assert rec["workoutId"] == slot["workoutId"]
    assert rec["distanceKm"] == slot["distanceKm"]
    assert rec["type"] in (slot["type"], "easy")  # easy when readiness forces a downgrade


def test_today_metrics_use_latest_known_values(client):
    today = client.get("/today").json()
    # The newest data row is a partial day without resting HR
    assert today["metrics"]["restingHr"] is not None
    assert today["dataAsOf"] == str(api.daily()["Date"].max().date())
    assert today["dataStale"] is True


def test_plan_week_shape(client):
    plan = client.get("/plan").json()
    week = plan["week"]
    assert [w["dayName"] for w in week] == api.DAY_KEYS
    assert plan["totals"]["runCount"] == sum(not w["isRest"] for w in week)
    for w in week:
        assert w["emoji"] == api.WTYPE_EMOJI[w["type"]]
        assert "Varied" not in w["details"]


def test_workout_segments_match_plan(client):
    plan = client.get("/plan").json()
    for w in plan["week"]:
        wo = client.get(f"/workout/{w['workoutId']}").json()
        assert wo["type"] == w["type"]
        assert wo["distanceKm"] == w["distanceKm"]
        if w["isRest"]:
            assert wo["segments"] == []
        else:
            assert wo["segments"]
            assert all(s["durationMin"] >= 1 for s in wo["segments"])


def test_segment_builder_scales_with_distance():
    paces = api._paces_raw({})
    short = sum(s["durationMin"] for s in api._workout_segments("long", 10, "", paces))
    long_ = sum(s["durationMin"] for s in api._workout_segments("long", 20, "", paces))
    assert long_ > short
    reps = [s for s in api._workout_segments("interval", 10, "6×800 m @4:30", paces) if s["kind"] == "work"]
    assert len(reps) == 6


def test_insights_use_runs_and_performance(client):
    ins = client.get("/insights").json()
    assert ins["runsCount"] == len(api.runs())
    assert ins["runsCount"] < len(api.daily())  # not one row per day
    assert len(ins["performanceRhythm"]) == 28
    assert ins["cyclesCount"] > 0


def test_stale_cycle_days_are_dropped():
    for df in (api.daily(), api._running_df):
        assert not (df["cycleDay"] > api.MAX_CYCLE_DAY).any()


def test_debrief_roundtrip(client):
    r = client.post("/debrief", json={"workoutId": "Tue_2026-10-27", "rpe": 6})
    assert r.json()["status"] == "saved"
    saved = client.get("/debriefs").json()["debriefs"]
    assert saved[-1]["rpe"] == 6 and "savedAt" in saved[-1]
