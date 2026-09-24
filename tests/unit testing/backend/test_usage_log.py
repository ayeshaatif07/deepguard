"""Unit tests for modules/usage_log.py, the per-feature usage log behind the dashboard
statistics.
"""


def test_log_usage_increments_feature_and_verdict(isolated_usage_db):
    isolated_usage_db.log_usage("Video", "Real")
    isolated_usage_db.log_usage("Video", "Real")
    isolated_usage_db.log_usage("Video", "Deepfake")

    stats = {row["feature"]: row for row in isolated_usage_db.get_usage_stats()}
    video = stats["Video"]

    assert video["total"] == 3
    assert video["labels"]["Real"] == 2
    assert video["labels"]["Deepfake"] == 1
    # A different feature must be unaffected by Video's usage.
    assert stats["Image"]["total"] == 0


def test_get_usage_stats_shows_all_features_with_zero_data(isolated_usage_db):
    stats = isolated_usage_db.get_usage_stats()
    features = [row["feature"] for row in stats]

    assert features == isolated_usage_db.FEATURES
    for row in stats:
        assert row["total"] == 0
        assert row["labels"] == {}


def test_log_usage_ignores_missing_verdict(isolated_usage_db):
    isolated_usage_db.log_usage("Voice Clone", None)
    isolated_usage_db.log_usage("Voice Clone", "")

    stats = {row["feature"]: row for row in isolated_usage_db.get_usage_stats()}
    assert stats["Voice Clone"]["total"] == 0
