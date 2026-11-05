"""Copy NFL player grades from the Sports_Analytics_Project database into this project's database.

Source: Sports_Analytics_Project's TiDB database `nfl_analytics` (read only: SELECT / SHOW).
        Its grading scripts (position_grades.py, offense_grades.py) rebuild these tables on each run.
Target: this project's TiDB database (TIDB_* settings in .env), tables with the same names and columns:
  player_grades           defensive players (ED, IDL, LB, CB, S), one row per player for the current season
  offense_player_grades   offensive skill players (QB, RB, WR, TE)

Each run replaces the target tables with a fresh copy, then checks that row counts and grade totals match.

Run from the repo root:  python scripts/copy_nfl_grades.py
Passwords are read from the two projects' .env files and never printed.
"""
import os
import re
import ssl
from pathlib import Path

import pymysql

REPO = Path(__file__).resolve().parents[1]
SOURCE_PROJECT = Path(os.environ.get(
    "SPORTS_ANALYTICS_PROJECT",
    Path.home() / "OneDrive - The Pennsylvania State University/Sports Analytics/Sports_Analytics_Project",
))
SOURCE = {
    "host": "gateway01.us-west-2.prod.aws.tidbcloud.com",
    "port": 4000,
    "user": "2sjjjYZZAfxKqVv.root",
    "database": "nfl_analytics",
}
TABLES = ["player_grades", "offense_player_grades"]
BATCH = 500


def read_env(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(.*?)\s*$", line)
        if m:
            values[m.group(1)] = m.group(2).strip("\"'")
    return values


def connect(host, port, user, password, database):
    return pymysql.connect(host=host, port=int(port), user=user, password=password, database=database,
                           ssl=ssl.create_default_context(), connect_timeout=20)


def totals(cur, table):
    cur.execute(f"SELECT COUNT(*), ROUND(SUM(grade), 6), ROUND(SUM(composite), 6) FROM `{table}`")
    return cur.fetchone()


def main():
    src_env = read_env(SOURCE_PROJECT / ".env")
    dst_env = read_env(REPO / ".env")
    src = connect(**SOURCE, password=src_env["DATABASE_PASSWORD"])
    dst = connect(dst_env["TIDB_HOST"], dst_env["TIDB_PORT"], dst_env["TIDB_USER"],
                  dst_env["TIDB_PASSWORD"], dst_env["TIDB_DATABASE"])
    try:
        with src.cursor() as s, dst.cursor() as d:
            for table in TABLES:
                s.execute(f"SHOW CREATE TABLE `{table}`")
                ddl = s.fetchone()[1]
                s.execute(f"SELECT * FROM `{table}`")
                cols = [c[0] for c in s.description]
                rows = s.fetchall()

                d.execute(f"DROP TABLE IF EXISTS `{table}`")
                d.execute(ddl)
                placeholders = ", ".join(["%s"] * len(cols))
                col_list = ", ".join(f"`{c}`" for c in cols)
                insert = f"INSERT INTO `{table}` ({col_list}) VALUES ({placeholders})"
                for i in range(0, len(rows), BATCH):
                    d.executemany(insert, rows[i:i + BATCH])
                dst.commit()

                src_totals, dst_totals = totals(s, table), totals(d, table)
                status = "OK" if src_totals == dst_totals else "MISMATCH"
                print(f"{table}: copied {len(rows)} rows, {len(cols)} columns "
                      f"[{status}: source {src_totals} vs copy {dst_totals}]")
                if status != "OK":
                    raise SystemExit(f"Verification failed for {table}")
    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
