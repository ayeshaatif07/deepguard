"""Times DeepGuard's actual usage_log.db operations (modules/usage_log.py) as the table
grows, on a throwaway copy of the database so the real usage_log.db is never touched.
"""
import os
import random
import sqlite3
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import write_report, fmt_row, timeit

import modules.usage_log as usage_log

TEST_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "_scaling_test.db")
ROW_COUNTS = [100, 1_000, 10_000, 100_000]
VERDICTS = ["Real", "Deepfake", "Cloned", "Manipulative", "Coherent", "Incoherent"]


def seed_rows(conn, count):
    rows = [
        (random.choice(usage_log.FEATURES), random.choice(VERDICTS))
        for _ in range(count)
    ]
    conn.executemany("INSERT INTO analysis_log (feature, verdict) VALUES (?, ?)", rows)
    conn.commit()


def main():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    original_db_path = usage_log.DB_PATH
    usage_log.DB_PATH = TEST_DB_PATH  # redirect the real module at a throwaway file

    lines = ["Database Scaling Benchmark (usage_log.db)", "=" * 60, ""]
    try:
        conn = usage_log._get_conn()
        conn.close()

        running_total = 0
        for target in ROW_COUNTS:
            to_add = target - running_total
            conn = sqlite3.connect(TEST_DB_PATH)
            seed_rows(conn, to_add)
            conn.close()
            running_total = target

            insert_stats = timeit(lambda: usage_log.log_usage("Video", "Real"), n=10, warmup=2)
            query_stats = timeit(usage_log.get_usage_stats, n=10, warmup=2)

            row_insert = fmt_row(f"  insert (1 row) @ {target:>7,} rows", insert_stats)
            row_query = fmt_row(f"  get_usage_stats() @ {target:>7,} rows", query_stats)
            print(row_insert)
            print(row_query)
            lines.append(row_insert)
            lines.append(row_query)

        lines.append("")
        lines.append("Notes:")
        lines.append("- Run against a throwaway copy of usage_log.db (results/_scaling_test.db),")
        lines.append("  never the real usage_log.db, using the real log_usage() and")
        lines.append("  get_usage_stats() functions from modules/usage_log.py unmodified.")
        lines.append("- get_usage_stats() runs one GROUP BY query over the whole table on")
        lines.append("  every dashboard page load (verdict.html), so its scaling with row")
        lines.append("  count is the more operationally relevant of the two numbers.")
    finally:
        usage_log.DB_PATH = original_db_path
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    write_report("database_scaling.txt", lines)


if __name__ == "__main__":
    main()
