"""Expected value score by draft pick, from past draft classes.

Runs the post-draft value model (post_draft_value.build) on the 2018-2023 draft classes, each measured
over its first three seasons (the same stage of their careers as a current class), and fits a smooth
curve of value score by pick: what a typical player taken at each slot turns into.

  score ~ a + b*ln(pick) + c*ln(pick)^2, least squares over every graded drafted player,
  then forced to never rise as the pick number grows, and floored at 0.

Offensive linemen (not graded yet) and specialists are left out, as in the value model.

Run from the repo root:  python scripts/expected_curve.py        (~20-40 minutes: six classes)
Writes app/data/expected_by_pick.json, which post_draft_value.py uses for "vs. pick" and "vs. your rank".
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from post_draft_value import OUT_DIR, build, draft_entries  # noqa: E402

CLASSES = [2018, 2019, 2020, 2021, 2022, 2023]
SEASONS_MEASURED = 3
MAX_PICK = 262


def fit(points):
    """Least squares for score = a + b*x + c*x^2 with x = ln(pick)."""
    sums = [[0.0] * 3 for _ in range(3)]
    rhs = [0.0] * 3
    for pick, score in points:
        x = math.log(pick)
        feats = [1.0, x, x * x]
        for i in range(3):
            rhs[i] += feats[i] * score
            for j in range(3):
                sums[i][j] += feats[i] * feats[j]
    # Solve the 3x3 system (Gaussian elimination)
    m = [row[:] + [rhs[i]] for i, row in enumerate(sums)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    return [m[i][3] / m[i][i] for i in range(3)]


def main():
    points, per_class = [], {}
    for year in CLASSES:
        through = year + SEASONS_MEASURED - 1
        players, _not_graded, *_ = build(year, draft_entries(year), through=through)
        class_points = [(p["pick"], p["score"]) for p in players if p.get("pick")]
        points.extend(class_points)
        per_class[year] = len(class_points)
        print(f"[{year}] {len(class_points)} graded picks through {through}")

    a, b, c = fit(points)
    expected, running = {}, float("inf")
    for pick in range(1, MAX_PICK + 1):
        x = math.log(pick)
        value = max(0.0, a + b * x + c * x * x)
        running = min(running, value)  # never rises with later picks
        expected[pick] = round(running, 1)

    # Average actual score by round, for reference
    by_round = {}
    for pick, score in points:
        rnd = min(7, 1 + (pick - 1) // 32)
        by_round.setdefault(rnd, []).append(score)
    data = {
        "classes": CLASSES,
        "seasons_measured": SEASONS_MEASURED,
        "players": len(points),
        "players_by_class": per_class,
        "coefficients": [a, b, c],
        "round_average": {r: round(sum(v) / len(v), 1) for r, v in sorted(by_round.items())},
        "by_pick": expected,
    }
    out = OUT_DIR / "expected_by_pick.json"
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"{len(points)} players -> {out}")
    print("expected score at picks 1/10/32/64/100/150/200/250:",
          [expected[p] for p in (1, 10, 32, 64, 100, 150, 200, 250)])
    print("average score by round:", data["round_average"])


if __name__ == "__main__":
    main()
