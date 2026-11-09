"""Build the fixed yardstick for single-season grades (scripts/grade_reference.json).

For each position room: the mean and spread of every graded stat among the room's regulars, pooled over
the 2018-2025 seasons, plus the spread of the combined score. season_grades.grade_rows measures every
season (including one in progress) against this, so a grade reflects how far a season is from normal,
not where it ranks that year.

Run from the repo root:  python scripts/grade_reference.py      (a few minutes)
Re-run it only to change the reference seasons; then rerun post_draft_value.py and expected_curve.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from post_draft_value import contract_rooms  # noqa: E402
from season_grades import (REFERENCE_PATH, REFERENCE_SEASONS, build_reference,  # noqa: E402
                           defense_table, offense_table, pfr_gsis_map)


def main():
    seasons = REFERENCE_SEASONS
    rows = offense_table(seasons)
    rows.update(defense_table(seasons, contract_rooms(), pfr_gsis_map(seasons)))
    ref = build_reference(rows)
    REFERENCE_PATH.write_text(json.dumps({"seasons": seasons, "rooms": ref}, indent=1), encoding="utf-8")
    print(f"-> {REFERENCE_PATH}")
    for room, r in ref.items():
        print(f"  {room}: {r['regulars']} regular seasons, composite sd {r['composite_sd']:.3f}")


if __name__ == "__main__":
    main()
