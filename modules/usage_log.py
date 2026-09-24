"""Per-feature usage counts in a small SQLite database.
Endpoints call log_usage() on each verdict; the dashboard reads get_usage_stats().
"""
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'usage_log.db')

# Fixed display order; every feature is shown even with 0 uses.
FEATURES = [
    'Video',
    'Image',
    'Voice Clone',
    'Voice Manipulation',
    'Caption Check',
    'URL',
    'Pipeline 1: Video -> Audio',
    'Social Media Pipeline',
]


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS analysis_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature TEXT NOT NULL,
        verdict TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )''')
    return conn


def log_usage(feature, verdict):
    """Records one use of `feature` with its verdict label. No-ops if verdict is missing."""
    if not verdict:
        return
    conn = _get_conn()
    try:
        conn.execute('INSERT INTO analysis_log (feature, verdict) VALUES (?, ?)', (feature, verdict))
        conn.commit()
    finally:
        conn.close()


def get_usage_stats():
    """Returns one {feature, total, labels} dict per FEATURES entry, labels sorted by count."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            'SELECT feature, verdict, COUNT(*) FROM analysis_log GROUP BY feature, verdict'
        ).fetchall()
    finally:
        conn.close()

    by_feature = {f: {} for f in FEATURES}
    for feature, verdict, count in rows:
        by_feature.setdefault(feature, {})[verdict] = count

    stats = []
    for feature in FEATURES:
        labels = dict(sorted(by_feature[feature].items(), key=lambda kv: kv[1], reverse=True))
        stats.append({
            'feature': feature,
            'total': sum(labels.values()),
            'labels': labels,
        })
    return stats
