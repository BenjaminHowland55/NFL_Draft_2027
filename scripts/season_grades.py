"""Single-season player grades (0-100) for the post-draft value model.

Each season is graded on that season's stats only, so a real jump or drop shows up the year it happens
(the Sports_Analytics_Project grades blend three seasons, which lags). For each position room:
  1. Compute the room's stats for every player that season (table STATS below).
  2. Among the season's regulars in the room (enough opportunities, prorated for a partial season),
     turn each stat into a z-score (sign flipped for lower-is-better stats) and take the weighted average.
  3. Grade = percentile of that composite among the regulars.
  4. Players below the regular threshold are placed on the same scale, then pulled toward 50 in
     proportion to how far short of the threshold they are, so a handful of snaps can't produce an extreme.

Offense uses nflreadpy player stats; defense uses Pro Football Reference's charted stats (pressures,
coverage, missed tackles) with defensive snaps from the snap counts.
"""
import math

import nflreadpy as nfl
import polars as pl

SEASON_GAMES = 17

# room -> list of (stat, weight, lower_is_better)
STATS = {
    "QB": [("epa_per_play", 0.7, False), ("cpoe", 0.3, False)],
    "RB": [("epa_per_opp", 0.5, False), ("first_down_rate", 0.3, False), ("yards_per_touch", 0.2, False)],
    # Target share: how much of the passing game runs through him (efficiency per target alone
    # punishes high-volume receivers on weak offenses)
    "WR": [("target_share", 0.35, False), ("rec_epa_per_target", 0.35, False),
           ("yards_per_target", 0.15, False), ("first_downs_per_target", 0.15, False)],
    "TE": [("target_share", 0.35, False), ("rec_epa_per_target", 0.35, False),
           ("yards_per_target", 0.15, False), ("first_downs_per_target", 0.15, False)],
    "EDGE": [("pressure_rate", 0.6, False), ("sack_rate", 0.2, False), ("missed_tackle_rate", 0.2, True)],
    "IDL": [("pressure_rate", 0.6, False), ("sack_rate", 0.2, False), ("missed_tackle_rate", 0.2, True)],
    "LB": [("missed_tackle_rate", 0.3, True), ("tackle_rate", 0.25, False),
           ("yards_allowed_per_snap", 0.25, True), ("pressure_rate", 0.2, False)],
    "CB": [("yards_allowed_per_snap", 0.4, True), ("rating_allowed", 0.25, True),
           ("missed_tackle_rate", 0.2, True), ("int_rate", 0.15, False)],
    "S": [("yards_allowed_per_snap", 0.3, True), ("rating_allowed", 0.2, True), ("int_rate", 0.2, False),
          ("missed_tackle_rate", 0.2, True), ("tackle_rate", 0.1, False)],
}
# Opportunities needed in a full season to count as a regular (prorated for a partial season)
REGULAR = {"QB": 250, "RB": 100, "WR": 50, "TE": 35, "EDGE": 300, "IDL": 300, "LB": 300, "CB": 300, "S": 300}
OFFENSE_POS = {"QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB", "WR": "WR", "TE": "TE"}
# Pro Football Reference defensive positions -> room (fallback when the contract position is unknown)
PFR_POS = {"DE": "EDGE", "OLB": "EDGE", "EDGE": "EDGE", "DT": "IDL", "NT": "IDL", "DL": "IDL",
           "LB": "LB", "ILB": "LB", "MLB": "LB", "CB": "CB", "DB": "CB", "S": "S", "FS": "S", "SS": "S"}


def _safe(num, den):
    return num / den if den else None


def offense_table(seasons):
    """One row per (gsis_id, season) with offensive stats and opportunities."""
    ps = nfl.load_player_stats(seasons).filter(pl.col("season_type") == "REG")
    ps = ps.with_columns((pl.col("passing_cpoe").fill_null(0) * pl.col("attempts")).alias("cpoe_x_att"))
    agg = ps.group_by(["player_id", "season"]).agg(
        pl.col("position").last(), pl.col("player_display_name").last().alias("name"),
        pl.col("target_share").fill_null(0).mean().alias("target_share"),
        *[pl.col(c).fill_null(0).sum() for c in [
            "attempts", "sacks_suffered", "carries", "targets", "receptions", "passing_epa", "rushing_epa",
            "receiving_epa", "rushing_yards", "receiving_yards", "rushing_first_downs", "receiving_first_downs",
            "cpoe_x_att"]])
    rows = {}
    for r in agg.iter_rows(named=True):
        room = OFFENSE_POS.get(r["position"])
        if not room:
            continue
        plays = r["attempts"] + r["sacks_suffered"] + r["carries"]
        opp = r["carries"] + r["targets"]
        rows[(r["player_id"], r["season"])] = {
            "room": room, "name": r["name"],
            "opps": {"QB": r["attempts"], "RB": opp, "WR": r["targets"], "TE": r["targets"]}[room],
            "epa_per_play": _safe(r["passing_epa"] + r["rushing_epa"], plays),
            "cpoe": _safe(r["cpoe_x_att"], r["attempts"]),
            "epa_per_opp": _safe(r["rushing_epa"] + r["receiving_epa"], opp),
            "first_down_rate": _safe(r["rushing_first_downs"] + r["receiving_first_downs"], opp),
            "yards_per_touch": _safe(r["rushing_yards"] + r["receiving_yards"], r["carries"] + r["receptions"]),
            "target_share": r["target_share"],
            "rec_epa_per_target": _safe(r["receiving_epa"], r["targets"]),
            "yards_per_target": _safe(r["receiving_yards"], r["targets"]),
            "first_downs_per_target": _safe(r["receiving_first_downs"], r["targets"]),
        }
    return rows


def defense_table(seasons, room_by_gsis, pfr_to_gsis):
    """One row per (gsis_id, season) with defensive stats; opportunities = defensive snaps."""
    snaps = nfl.load_snap_counts(seasons).filter(pl.col("game_type") == "REG").group_by(
        ["pfr_player_id", "season"]).agg(pl.col("defense_snaps").sum())
    snap_map = {(p, s): n for p, s, n in snaps.iter_rows()}
    pfr = nfl.load_pfr_advstats(seasons, stat_type="def", summary_level="week").filter(pl.col("game_type") == "REG")
    pfr = pfr.with_columns((pl.col("def_passer_rating_allowed").fill_null(0) * pl.col("def_targets").fill_null(0)).alias("rat_x_tgt"))
    agg = pfr.group_by(["pfr_player_id", "season"]).agg(
        pl.col("pfr_player_name").last().alias("name"),
        *[pl.col(c).fill_null(0).sum() for c in [
            "def_ints", "def_targets", "def_yards_allowed", "def_sacks", "def_pressures",
            "def_tackles_combined", "def_missed_tackles", "rat_x_tgt"]])
    rows = {}
    for r in agg.iter_rows(named=True):
        gsis = pfr_to_gsis.get(r["pfr_player_id"])
        room = room_by_gsis.get(gsis) if gsis else None
        if room not in ("EDGE", "IDL", "LB", "CB", "S"):
            continue
        n = snap_map.get((r["pfr_player_id"], r["season"]), 0)
        tackle_attempts = r["def_tackles_combined"] + r["def_missed_tackles"]
        rows[(gsis, r["season"])] = {
            "room": room, "name": r["name"], "opps": n,
            "pressure_rate": _safe(r["def_pressures"], n),
            "sack_rate": _safe(r["def_sacks"], n),
            "missed_tackle_rate": _safe(r["def_missed_tackles"], tackle_attempts),
            "tackle_rate": _safe(r["def_tackles_combined"], n),
            "yards_allowed_per_snap": _safe(r["def_yards_allowed"], n),
            "rating_allowed": _safe(r["rat_x_tgt"], r["def_targets"]),
            "int_rate": _safe(r["def_ints"], n),
        }
    return rows


def _percentile(sorted_vals, x):
    lo, hi = 0, len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return 100.0 * lo / max(1, len(sorted_vals) - 1) if len(sorted_vals) > 1 else 50.0


def grade_rows(rows, season_fraction):
    """Add a 0-100 'grade' to each row, graded within (room, season)."""
    groups = {}
    for key, r in rows.items():
        groups.setdefault((r["room"], key[1]), []).append(r)
    for (room, season), members in groups.items():
        need = REGULAR[room] * season_fraction.get(season, 1.0)
        regulars = [r for r in members if r["opps"] >= need]
        if len(regulars) < 5:
            for r in members:
                r["grade"] = None
            continue
        stats = STATS[room]
        moments = {}
        for stat, _w, _low in stats:
            vals = [r[stat] for r in regulars if r[stat] is not None]
            mean = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) or 1.0
            moments[stat] = (mean, sd)

        def composite(r):
            total = 0.0
            for stat, w, low in stats:
                mean, sd = moments[stat]
                v = r[stat] if r[stat] is not None else mean
                z = max(-3.0, min(3.0, (v - mean) / sd))
                total += w * (-z if low else z)
            return total

        reg_scores = sorted(composite(r) for r in regulars)
        for r in members:
            pct = _percentile(reg_scores, composite(r))
            trust = min(1.0, r["opps"] / need) if need else 1.0
            r["grade"] = round(50 + (pct - 50) * trust, 1)


def season_grades(seasons, room_by_gsis, season_fraction):
    """(gsis_id, season) -> grade for every offensive skill player and defender."""
    rosters = nfl.load_rosters_weekly(seasons).select(["gsis_id", "pfr_id"]).drop_nulls().unique()
    pfr_to_gsis = {p: g for g, p in rosters.iter_rows()}
    rows = offense_table(seasons)
    rows.update(defense_table(seasons, room_by_gsis, pfr_to_gsis))
    grade_rows(rows, season_fraction)
    return {k: r["grade"] for k, r in rows.items()}, rows
