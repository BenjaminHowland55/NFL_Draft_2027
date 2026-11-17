"""Age on draft day for each player on a year's board.

Birth dates come from nflverse's players table (nflreadpy). Board players are linked to
NFL ids through the post-draft value file first (same matching as the value model), then
by name / college against everyone whose draft year or rookie season is that year.
Players who never reached the NFL have no age.

Writes app/data/{year}_ages.json: {slug: {"birth_date": "YYYY-MM-DD", "age": 21.4}}

    python scripts/draft_ages.py 2024 2025 2026
"""
import datetime as dt
import json
import sys
from pathlib import Path

import nflreadpy as nfl
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from import_data import name_key  # noqa: E402
from post_draft_value import match_board  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "app" / "data"
# First night of each draft (round 1)
DRAFT_DAY = {2024: dt.date(2024, 4, 25), 2025: dt.date(2025, 4, 24), 2026: dt.date(2026, 4, 23)}


def main(year, people):
    board = json.loads((DATA / f"{year}.json").read_text(encoding="utf-8"))["players"]
    gsis = {}
    value_path = DATA / f"{year}_value.json"
    if value_path.exists():
        value = json.loads(value_path.read_text(encoding="utf-8"))
        for v in value["players"] + value.get("not_graded", []):
            if v.get("board_name") and v.get("gsis_id"):
                gsis[name_key(v["board_name"])] = v["gsis_id"]

    birth = {r["gsis_id"]: r["birth_date"] for r in people.iter_rows(named=True) if r["birth_date"]}
    pool = people.filter((pl.col("draft_year") == year) | (pl.col("rookie_season") == year))
    candidates = [{"name": r["display_name"], "college": (r["college_name"] or "").split(";")[0],
                   "gsis_id": r["gsis_id"]} for r in pool.iter_rows(named=True)]
    # This year's entrants first (name / near-spelling / last name at the same college);
    # the value file's id only for players not found there. Draft-pick ids are occasionally
    # wrong (Peter Woods 2026 carries a 1970s player's id), so an age outside 19-30 is rejected.
    direct = match_board(board, candidates)

    out, missing = {}, []
    for b in board:
        age = None
        for gid in ((direct.get(b["slug"]) or {}).get("gsis_id"), gsis.get(name_key(b["name"]))):
            bd = birth.get(gid)
            if bd:
                age = round((DRAFT_DAY[year] - dt.date.fromisoformat(bd)).days / 365.25, 1)
                if 19 <= age <= 30:
                    break
                age = None
        if age is None:
            missing.append(b["name"])
            continue
        out[b["slug"]] = {"birth_date": bd, "age": age}
    path = DATA / f"{year}_ages.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    ages = sorted(v["age"] for v in out.values())
    print(f"[{year}] {len(out)}/{len(board)} ages -> {path.name}; youngest {ages[0]}, oldest {ages[-1]}")
    print(f"[{year}] no age ({len(missing)}): {missing}")


if __name__ == "__main__":
    people = nfl.load_players()
    for y in [int(a) for a in sys.argv[1:]] or sorted(DRAFT_DAY):
        main(y, people)
