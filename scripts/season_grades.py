"""Single-season player grades (0-100) for the post-draft value model.

Each season is graded on that season's stats only, so a real jump or drop shows up the year it happens
(the Sports_Analytics_Project grades blend three seasons, which lags). For each position room:
  1. Compute the room's stats for every player that season (table STATS below).
  2. Turn each stat into a z-score against a FIXED yardstick: the mean and spread of that stat among
     the room's regulars across the 2018-2025 seasons (scripts/grade_reference.json, built by
     scripts/grade_reference.py), sign flipped for lower-is-better stats, and take the weighted average.
  3. Grade = 100 x normal CDF(composite / the reference composite's spread). 50 = a typical regular's
     season. It's a distance, not a rank: players with nearly identical stats get nearly identical
     grades, and a dominant season sits well clear of a merely good one.
  4. Players below the regular threshold (enough opportunities, prorated for a partial season) are
     pulled toward 50 in proportion to how far short they are, so a handful of snaps can't produce an
     extreme.

Offense uses nflreadpy player stats; defense uses Pro Football Reference's charted stats (pressures,
coverage, missed tackles) with defensive snaps from the snap counts.
"""
import json
import math
from pathlib import Path

import nflreadpy as nfl
import polars as pl

SEASON_GAMES = 17
REFERENCE_PATH = Path(__file__).resolve().parent / "grade_reference.json"
REFERENCE_SEASONS = list(range(2018, 2026))  # full seasons that set the fixed yardstick

# room -> list of (stat, weight, lower_is_better)
# Each room mixes efficiency (rates) with raw production per game played (yards, TDs, pressures, tackles),
# so a productive workhorse isn't outgraded by an efficient part-timer. Production is per game PLAYED, so
# games missed to injury don't drag it down. Corners have no production stat: in coverage, a quiet game
# is usually a good one.
STATS = {
    "QB": [("epa_per_play", 0.45, False), ("cpoe", 0.2, False),
           ("yards_per_game", 0.2, False), ("tds_per_game", 0.15, False)],
    "RB": [("epa_per_opp", 0.25, False), ("first_down_rate", 0.15, False), ("yards_per_touch", 0.1, False),
           ("scrimmage_yards_per_game", 0.35, False), ("scrimmage_tds_per_game", 0.15, False)],
    # Target share: how much of the passing game runs through him
    "WR": [("target_share", 0.25, False), ("rec_epa_per_target", 0.2, False), ("yards_per_target", 0.1, False),
           ("first_downs_per_target", 0.1, False), ("rec_yards_per_game", 0.25, False), ("rec_tds_per_game", 0.1, False)],
    "TE": [("target_share", 0.25, False), ("rec_epa_per_target", 0.2, False), ("yards_per_target", 0.1, False),
           ("first_downs_per_target", 0.1, False), ("rec_yards_per_game", 0.25, False), ("rec_tds_per_game", 0.1, False)],
    "EDGE": [("pressure_rate", 0.4, False), ("sack_rate", 0.1, False), ("missed_tackle_rate", 0.15, True),
             ("pressures_per_game", 0.25, False), ("sacks_per_game", 0.1, False)],
    "IDL": [("pressure_rate", 0.4, False), ("sack_rate", 0.1, False), ("missed_tackle_rate", 0.15, True),
            ("pressures_per_game", 0.25, False), ("sacks_per_game", 0.1, False)],
    "LB": [("missed_tackle_rate", 0.2, True), ("tackle_rate", 0.15, False), ("yards_allowed_per_snap", 0.2, True),
           ("pressure_rate", 0.1, False), ("tackles_per_game", 0.2, False), ("sacks_per_game", 0.15, False)],
    "CB": [("yards_allowed_per_snap", 0.4, True), ("rating_allowed", 0.25, True),
           ("missed_tackle_rate", 0.2, True), ("int_rate", 0.15, False)],
    "S": [("yards_allowed_per_snap", 0.25, True), ("rating_allowed", 0.15, True), ("int_rate", 0.15, False),
          ("missed_tackle_rate", 0.15, True), ("tackle_rate", 0.1, False), ("tackles_per_game", 0.2, False)],
    # Offensive line: an ESTIMATE from free data (no individual pressures/sacks allowed exist in it).
    # Run blocking = runs through the gaps he's responsible for; pass protection = his team's sack and
    # QB-hit rates on dropbacks in his games; penalties are charged to him individually. Run and pass
    # results are shared with teammates, so this separates good units and sides more than individuals.
    "OT": [("gap_run_epa", 0.35, False), ("gap_run_success", 0.15, False), ("sack_rate_allowed", 0.15, True),
           ("qb_hit_rate_allowed", 0.1, True), ("penalty_rate", 0.25, True)],
    "IOL": [("gap_run_epa", 0.35, False), ("gap_run_success", 0.15, False), ("sack_rate_allowed", 0.15, True),
            ("qb_hit_rate_allowed", 0.1, True), ("penalty_rate", 0.25, True)],
}
# Opportunities needed in a full season to count as a regular (prorated for a partial season)
REGULAR = {"QB": 250, "RB": 100, "WR": 50, "TE": 35, "EDGE": 300, "IDL": 300, "LB": 300, "CB": 300, "S": 300,
           "OT": 400, "IOL": 400}
# Run gaps each offensive line spot is responsible for: (run_location, run_gap) -> weight
OL_GAPS = {
    "LT": {("left", "end"): 1.0, ("left", "tackle"): 1.0, ("left", "guard"): 0.3},
    "LG": {("left", "tackle"): 0.5, ("left", "guard"): 1.0, ("middle", None): 0.5},
    "C": {("middle", None): 1.0, ("left", "guard"): 0.5, ("right", "guard"): 0.5},
    "RG": {("right", "tackle"): 0.5, ("right", "guard"): 1.0, ("middle", None): 0.5},
    "RT": {("right", "end"): 1.0, ("right", "tackle"): 1.0, ("right", "guard"): 0.3},
}
OL_SPOT_ROOM = {"LT": "OT", "RT": "OT", "LG": "IOL", "C": "IOL", "RG": "IOL"}
OFFENSIVE_PENALTIES = {"Offensive Holding", "False Start", "Illegal Use of Hands", "Chop Block",
                       "Clipping", "Illegal Block Above the Waist", "Low Block", "Offensive Offside",
                       "Illegal Formation", "Tripping", "Unnecessary Roughness", "Personal Foul"}
OFFENSE_POS = {"QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB", "WR": "WR", "TE": "TE"}
# Pro Football Reference defensive positions -> room (fallback when the contract position is unknown)
PFR_POS = {"DE": "EDGE", "OLB": "EDGE", "EDGE": "EDGE", "DT": "IDL", "NT": "IDL", "DL": "IDL",
           "LB": "LB", "ILB": "LB", "MLB": "LB", "CB": "CB", "DB": "CB", "S": "S", "FS": "S", "SS": "S"}


def _safe(num, den):
    return num / den if den else None


def offense_table(seasons, room_by_gsis=None):
    """One row per (gsis_id, season) with offensive stats and opportunities."""
    ps = nfl.load_player_stats(seasons).filter(pl.col("season_type") == "REG")
    ps = ps.with_columns((pl.col("passing_cpoe").fill_null(0) * pl.col("attempts")).alias("cpoe_x_att"))
    agg = ps.group_by(["player_id", "season"]).agg(
        pl.col("position").last(), pl.col("player_display_name").last().alias("name"),
        pl.col("target_share").fill_null(0).mean().alias("target_share"),
        pl.col("week").n_unique().alias("games"),
        *[pl.col(c).fill_null(0).sum() for c in [
            "attempts", "sacks_suffered", "carries", "targets", "receptions", "passing_epa", "rushing_epa",
            "receiving_epa", "rushing_yards", "receiving_yards", "rushing_first_downs", "receiving_first_downs",
            "cpoe_x_att", "passing_yards", "passing_tds", "rushing_tds", "receiving_tds"]])
    rows = {}
    for r in agg.iter_rows(named=True):
        # The position room he's valued in, if known (e.g. a fullback listed as a running back here but
        # valued as a tight end), else his listed position
        room = (room_by_gsis or {}).get(r["player_id"])
        if room not in ("QB", "RB", "WR", "TE"):
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
            "yards_per_game": _safe(r["passing_yards"] + r["rushing_yards"], r["games"]),
            "tds_per_game": _safe(r["passing_tds"] + r["rushing_tds"], r["games"]),
            "scrimmage_yards_per_game": _safe(r["rushing_yards"] + r["receiving_yards"], r["games"]),
            "scrimmage_tds_per_game": _safe(r["rushing_tds"] + r["receiving_tds"], r["games"]),
            "rec_yards_per_game": _safe(r["receiving_yards"], r["games"]),
            "rec_tds_per_game": _safe(r["receiving_tds"], r["games"]),
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
        pl.col("pfr_player_name").last().alias("name"), pl.col("week").n_unique().alias("games"),
        *[pl.col(c).fill_null(0).sum() for c in [
            "def_ints", "def_targets", "def_yards_allowed", "def_sacks", "def_pressures",
            "def_tackles_combined", "def_missed_tackles", "rat_x_tgt"]])
    # Sacks from the NFL's own weekly stats: current through the latest week (the charted stats above can
    # trail by a week), and the official count
    ps = nfl.load_player_stats(seasons).filter(pl.col("season_type") == "REG").group_by(
        ["player_id", "season"]).agg(pl.col("def_sacks").fill_null(0).sum(), pl.col("week").n_unique().alias("g"))
    official_sacks = {(g, s): (sk, n) for g, s, sk, n in ps.iter_rows()}
    rows = {}
    for r in agg.iter_rows(named=True):
        gsis = pfr_to_gsis.get(r["pfr_player_id"])
        room = room_by_gsis.get(gsis) if gsis else None
        if room not in ("EDGE", "IDL", "LB", "CB", "S"):
            continue
        n = snap_map.get((r["pfr_player_id"], r["season"]), 0)
        tackle_attempts = r["def_tackles_combined"] + r["def_missed_tackles"]
        sacks, sack_games = official_sacks.get((gsis, r["season"]), (r["def_sacks"], r["games"]))
        rows[(gsis, r["season"])] = {
            "room": room, "name": r["name"], "opps": n,
            "pressure_rate": _safe(r["def_pressures"], n),
            "sack_rate": _safe(sacks, n),
            "missed_tackle_rate": _safe(r["def_missed_tackles"], tackle_attempts),
            "tackle_rate": _safe(r["def_tackles_combined"], n),
            "yards_allowed_per_snap": _safe(r["def_yards_allowed"], n),
            "rating_allowed": _safe(r["rat_x_tgt"], r["def_targets"]),
            "int_rate": _safe(r["def_ints"], n),
            "pressures_per_game": _safe(r["def_pressures"], r["games"]),
            "sacks_per_game": _safe(sacks, sack_games),
            "tackles_per_game": _safe(r["def_tackles_combined"], r["games"]),
        }
    return rows


def _regulars(rows, season_fraction):
    """(room, season) -> (all members, regulars, the regular threshold)."""
    groups = {}
    for key, r in rows.items():
        groups.setdefault((r["room"], key[1]), []).append(r)
    out = {}
    for (room, season), members in groups.items():
        need = REGULAR[room] * season_fraction.get(season, 1.0)
        out[(room, season)] = (members, [r for r in members if r["opps"] >= need], need)
    return out


def _composite(r, room, moments):
    total = 0.0
    for stat, w, low in STATS[room]:
        mean, sd = moments[stat]
        v = r[stat] if r[stat] is not None else mean
        z = max(-3.0, min(3.0, (v - mean) / sd))
        total += w * (-z if low else z)
    return total


def build_reference(rows):
    """Fixed yardstick from full seasons: per room, each stat's mean/sd among regulars (pooled across
    seasons) and the spread of the resulting composite."""
    pooled = {}
    for (room, _season), (_members, regulars, _need) in _regulars(rows, {}).items():
        pooled.setdefault(room, []).extend(regulars)
    ref = {}
    for room, regulars in pooled.items():
        moments = {}
        for stat, _w, _low in STATS[room]:
            vals = [r[stat] for r in regulars if r[stat] is not None]
            mean = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) or 1.0
            moments[stat] = [mean, sd]
        comps = [_composite(r, room, moments) for r in regulars]
        cm = sum(comps) / len(comps)
        csd = math.sqrt(sum((c - cm) ** 2 for c in comps) / max(1, len(comps) - 1)) or 1.0
        ref[room] = {"regulars": len(regulars), "stats": moments, "composite_mean": cm, "composite_sd": csd}
    return ref


def load_reference():
    if not REFERENCE_PATH.exists():
        raise SystemExit(f"{REFERENCE_PATH.name} missing: run python scripts/grade_reference.py first")
    data = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    for room in data["rooms"].values():
        room["stats"] = {k: tuple(v) for k, v in room["stats"].items()}
    return data["rooms"]


def grade_rows(rows, season_fraction, reference=None):
    """Add a 0-100 'grade' to each row on the fixed yardstick (see the module docstring)."""
    reference = reference or load_reference()
    for (room, season), (members, regulars, need) in _regulars(rows, season_fraction).items():
        ref = reference.get(room)
        if ref is None or len(regulars) < 5:
            for r in members:
                r["grade"] = None
            continue
        for r in members:
            z = (_composite(r, room, ref["stats"]) - ref["composite_mean"]) / ref["composite_sd"]
            grade = 50.0 * (1.0 + math.erf(z / math.sqrt(2.0)))  # normal CDF, 0-100
            trust = min(1.0, r["opps"] / need) if need else 1.0
            r["grade"] = round(50 + (grade - 50) * trust, 1)


def pfr_gsis_map(seasons):
    """Pro Football Reference id -> NFL gsis id. The players table covers nearly everyone (weekly rosters
    leave pfr_id blank for many players, e.g. most offensive linemen); rosters fill any gaps."""
    out = {p: g for g, p in nfl.load_players().select(["gsis_id", "pfr_id"]).drop_nulls().iter_rows()}
    for g, p in nfl.load_rosters_weekly(seasons).select(["gsis_id", "pfr_id"]).drop_nulls().unique().iter_rows():
        out.setdefault(p, g)
    return out


def ol_positions(seasons):
    """(gsis_id, season) -> his main offensive line spot (LT/LG/C/RG/RT) from the depth charts."""
    out = {}
    for season in seasons:
        dc = nfl.load_depth_charts([season])
        # Two formats: newer seasons have pos_abb / pos_rank; older ones (through 2024) have
        # depth_position / depth_team ("1" = starter). The spot names (LT, LG, C, RG, RT) match.
        if "pos_abb" not in dc.columns:
            dc = dc.filter(pl.col("game_type") == "REG").select(
                "gsis_id", pl.col("depth_position").str.strip_chars().alias("pos_abb"),
                pl.col("depth_team").cast(pl.Int64, strict=False).alias("pos_rank"))
        dc = dc.filter(pl.col("pos_abb").is_in(list(OL_GAPS)))
        starters = dc.filter(pl.col("pos_rank") == 1)
        counts = {}
        for g, spot in (starters if starters.height else dc).select(["gsis_id", "pos_abb"]).iter_rows():
            if g:
                counts.setdefault(g, {}).setdefault(spot, 0)
                counts[g][spot] += 1
        # Players who never started: use any depth chart spot
        for g, spot in dc.select(["gsis_id", "pos_abb"]).iter_rows():
            if g and g not in counts:
                counts.setdefault(g, {}).setdefault(spot, 0)
        for g, c in counts.items():
            out[(g, season)] = max(c, key=c.get)
    return out


def offensive_line_table(seasons, pfr_to_gsis):
    """One row per (gsis_id, season) for offensive linemen; opportunities = offensive snaps."""
    spots = ol_positions(seasons)
    sc = nfl.load_snap_counts(seasons).filter(
        (pl.col("game_type") == "REG") & pl.col("position").is_in(["T", "G", "C", "OL", "OT", "OG"]))
    pbp = nfl.load_pbp(seasons).filter(pl.col("season_type") == "REG")
    runs = pbp.filter(pl.col("play_type") == "run").select(
        ["game_id", "posteam", "run_location", "run_gap", "epa", "success"])
    drops = pbp.filter(pl.col("qb_dropback") == 1).select(["game_id", "posteam", "sack", "qb_hit"])
    pens = pbp.filter((pl.col("penalty") == 1) & pl.col("penalty_type").is_in(list(OFFENSIVE_PENALTIES))
                      & pl.col("penalty_player_id").is_not_null())

    # Per team-game: run results by gap, and dropback totals
    run_by = {}
    for gid, team, loc, gap, epa, succ in runs.iter_rows():
        if epa is None:
            continue
        key = (loc, gap if loc != "middle" else None)
        rec = run_by.setdefault((gid, team), {}).setdefault(key, [0, 0.0, 0.0])
        rec[0] += 1
        rec[1] += epa
        rec[2] += succ or 0
    drop_by = {}
    for gid, team, sack, hit in drops.iter_rows():
        rec = drop_by.setdefault((gid, team), [0, 0, 0])
        rec[0] += 1
        rec[1] += sack or 0
        rec[2] += hit or 0
    pen_by = {}
    for g, season in pens.select(["penalty_player_id", "season"]).iter_rows():
        pen_by[(g, season)] = pen_by.get((g, season), 0) + 1

    acc = {}
    for gid, season, team, pfr, snaps, pct, name in sc.select(
            ["game_id", "season", "team", "pfr_player_id", "offense_snaps", "offense_pct", "player"]).iter_rows():
        gsis = pfr_to_gsis.get(pfr)
        spot = spots.get((gsis, season))
        if not gsis or not spot or not snaps:
            continue
        a = acc.setdefault((gsis, season), {"name": name, "spot": spot, "snaps": 0.0, "games": 0,
                                            "runs": 0.0, "run_epa": 0.0, "run_succ": 0.0,
                                            "drops": 0.0, "sacks": 0.0, "hits": 0.0})
        w = pct or 0.0  # his share of the team's snaps that game
        a["snaps"] += snaps
        a["games"] += 1
        for key, gap_w in OL_GAPS[spot].items():
            r = run_by.get((gid, team), {}).get(key)
            if r:
                a["runs"] += w * gap_w * r[0]
                a["run_epa"] += w * gap_w * r[1]
                a["run_succ"] += w * gap_w * r[2]
        d = drop_by.get((gid, team))
        if d:
            a["drops"] += w * d[0]
            a["sacks"] += w * d[1]
            a["hits"] += w * d[2]

    rows = {}
    for (gsis, season), a in acc.items():
        rows[(gsis, season)] = {
            "room": OL_SPOT_ROOM[a["spot"]], "spot": a["spot"], "name": a["name"], "opps": a["snaps"],
            "gap_run_epa": _safe(a["run_epa"], a["runs"]),
            "gap_run_success": _safe(a["run_succ"], a["runs"]),
            "sack_rate_allowed": _safe(a["sacks"], a["drops"]),
            "qb_hit_rate_allowed": _safe(a["hits"], a["drops"]),
            "penalty_rate": _safe(pen_by.get((gsis, season), 0), a["snaps"]),
        }
    return rows


def season_grades(seasons, room_by_gsis, season_fraction):
    """(gsis_id, season) -> grade for every offensive skill player, offensive lineman and defender."""
    pfr_to_gsis = pfr_gsis_map(seasons)
    rows = offense_table(seasons, room_by_gsis)
    rows.update(defense_table(seasons, room_by_gsis, pfr_to_gsis))
    # Offensive line estimate (offensive_line_table) is not used: free data can't separate a lineman
    # from his teammates well enough (it graded whole lines, not players). Linemen stay ungraded.
    grade_rows(rows, season_fraction)
    return {k: r["grade"] for k, r in rows.items()}, rows
