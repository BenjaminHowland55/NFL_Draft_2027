"""Post-draft value rank for a draft class: how much each drafted player has been worth to his team so far.

Value per season = positional value x snap share x grade multiplier
  positional value  median cap share (APY / cap) of veteran starters at the player's position, from Over The
                    Cap contracts (rookie deals left out). This is what the NFL pays the position.
  snap share        the player's offensive or defensive snaps / his team's snaps in the games he was
                    available for, relative to a typical starter at his position (starter_shares: e.g. RB
                    ~62%, QB ~94%), capped at 1: a 62% RB is as full-time as a 94% QB. Games missed to
                    injury (IR, PUP, Out/Doubtful, or on the report and didn't play) are excused up to
                    INJURY_ALLOWANCE of his team's games across his career; past that, missed games count
                    as not playing. Benchings and healthy scratches count as zero.
  grade multiplier  GRADE_MIN_MULT (grade 0) to GRADE_MAX_MULT (grade 100), from a single-season grade
                    (season_grades.py: that season's stats only, a percentile among the season's regulars at
                    the position). Wide enough that quality separates starters, while a 90%-snap player graded
                    40 still outranks a 30%-snap backup graded 80. No grade for a season = neutral 50.

Player value = weighted average of his season values. Weights: recency (RECENCY_WEIGHTS: quarterbacks
5-2-1 for this season / last season / two ago, other positions 3-2-1, whose play depends more on
circumstances) x how much of the season he was available for. The current season counts in full
even while it's in progress. Built to look ahead: a player who has improved and is playing well now ranks
high, and one who has dropped off falls.

Undrafted players from your big board are found among that year's undrafted rookies and valued the same way.

Score = SCORE_AT_MARKET x player value / his position's market rate: 70 = playing like a full-time,
market-rate starter at his position (average grade, every snap); higher = outplaying it (best possible
~126); 35 = half that value. Comparable across positions.

Value rank = rank of player value within the draft class (1 = most valuable). Kickers, punters and long
snappers are left out. Offensive linemen are listed but not ranked until individual OL data is added.

Run from the repo root:  python scripts/post_draft_value.py 2024
Writes app/data/<year>_value.json.
"""
import difflib
import json
import re
import ssl
import sys
from pathlib import Path

import nflreadpy as nfl
import polars as pl
import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_data import name_key, near_match, split_name  # noqa: E402
from season_grades import pfr_gsis_map, season_grades  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "app" / "data"
SEASON_GAMES = 17
GRADE_MIN_MULT, GRADE_MAX_MULT = 0.2, 1.8
NEUTRAL_GRADE = 50
# Injured games are excused up to this share of his team's games across his career; beyond it, missed
# games count as not playing (one bad injury year is forgiven, being hurt every year isn't)
INJURY_ALLOWANCE = 0.25
# Starters per team at each position, for the typical starter snap share (starter_shares)
STARTERS_PER_TEAM = {"QB": 1, "RB": 1, "WR": 3, "TE": 1, "OT": 2, "IOL": 3, "EDGE": 2, "IDL": 2, "LB": 2, "CB": 3, "S": 2}
SCORE_AT_MARKET = 70  # score for a full-time, average-graded starter at his position's market rate
# Season weights by how long ago it was (this season, last season, two ago, ...): built for
# looking ahead, so recent play counts most.
RECENCY_WEIGHTS = {
    "QB": [5, 2, 1, 0.5],
    "default": [3, 2, 1, 0.5],  # other positions depend more on scheme, teammates and role
}
SPECIALISTS = {"K", "P", "LS"}
# Offensive linemen have no individual grades yet; they are listed but left out of the value rank for now.
# Offensive linemen have no individual grades in free data; they are listed but left out of the value rank.
NOT_GRADED = {"OT", "IOL", "OL"}

# Rooms used for value. A room's positional value is the median of its top MARKET_TOP veteran contracts:
# the going rate for a top starter at that position on the open market. (Many starters are still on rookie
# deals, so going deeper reaches backup and minimum contracts, e.g. the 30th-highest-paid veteran QB is a
# backup at ~1% of the cap, which would understate the position.)
OFFENSE = {"QB", "RB", "WR", "TE", "OT", "IOL"}
MARKET_TOP = 10
STARTERS = {room: MARKET_TOP for room in ["QB", "RB", "WR", "TE", "OT", "IOL", "IDL", "EDGE", "LB", "CB", "S"]}
# Over The Cap contract positions -> room
OTC_ROOM = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "LT": "OT", "RT": "OT",
            "LG": "IOL", "RG": "IOL", "C": "IOL", "IDL": "IDL", "ED": "EDGE", "LB": "LB",
            "CB": "CB", "S": "S", "K": "K", "P": "P", "LS": "LS"}
# Draft-pick positions -> room (fallback when there's no contract record)
DRAFT_ROOM = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "T": "OT", "OT": "OT",
              "G": "IOL", "C": "IOL", "OL": "OL", "DT": "IDL", "NT": "IDL", "DL": "IDL", "DE": "EDGE",
              "OLB": "EDGE", "LB": "LB", "ILB": "LB", "CB": "CB", "S": "S", "SAF": "S", "DB": "CB",
              "K": "K", "P": "P", "LS": "LS"}
# Grade tables' rooms -> room
GRADE_ROOM = {"ED": "EDGE", "IDL": "IDL", "LB": "LB", "CB": "CB", "S": "S",
              "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE"}


def read_env(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(.*?)\s*$", line)
        if m:
            values[m.group(1)] = m.group(2).strip("\"'")
    return values


# ---------------------------------------------------------------- inputs

def positional_values(current_season):
    """Median cap share of each room's veteran starters (top-N active non-rookie contracts)."""
    ct = nfl.load_contracts().filter(pl.col("is_active"))
    ct = ct.with_columns(pl.col("position").replace_strict(OTC_ROOM, default=None).alias("room"))
    vets = ct.filter(
        pl.col("room").is_in(list(STARTERS))
        & pl.col("apy_cap_pct").is_not_null()
        & (pl.col("draft_year").is_null() | (pl.col("year_signed") > pl.col("draft_year")))
    )
    values = {}
    for room, n in STARTERS.items():
        top = vets.filter(pl.col("room") == room).sort("apy_cap_pct", descending=True).head(n)
        values[room] = float(top["apy_cap_pct"].median())
    values["OL"] = (values["OT"] + values["IOL"]) / 2  # drafted as "OL" with no contract position yet
    return values


def contract_rooms():
    """gsis_id -> room from each player's most recent contract record."""
    ct = nfl.load_contracts().filter(pl.col("gsis_id").is_not_null())
    ct = ct.sort("year_signed", descending=True).unique("gsis_id", keep="first")
    return {g: OTC_ROOM.get(p) for g, p in zip(ct["gsis_id"], ct["position"]) if OTC_ROOM.get(p)}


# Weekly roster codes for injury lists: R01 injured reserve, R48 IR (designated to return),
# R04 physically unable to perform, R05 non-football injury. Suspensions etc. are not excused.
INJURY_RESERVE_CODES = {"R01", "R48", "R04", "R05"}
INJURY_REPORT_OUT = {"Out", "Doubtful"}


def weekly_snaps(seasons):
    """Snap counts by week.

    Returns:
      team_week    (season, team) -> {week: (team offense snaps, team defense snaps)}
      player_week  (pfr_id, season) -> {week: (team, offense snaps, defense snaps)}
    """
    sc = nfl.load_snap_counts(seasons).filter(pl.col("game_type") == "REG")
    team_week = {}
    agg = sc.group_by(["season", "team", "week"]).agg(
        pl.col("offense_snaps").max().alias("off"), pl.col("defense_snaps").max().alias("def"))
    for r in agg.iter_rows(named=True):
        team_week.setdefault((r["season"], r["team"]), {})[r["week"]] = (r["off"], r["def"])
    player_week = {}
    for r in sc.iter_rows(named=True):
        player_week.setdefault((r["pfr_player_id"], r["season"]), {})[r["week"]] = (
            r["team"], r["offense_snaps"] or 0, r["defense_snaps"] or 0)
    return team_week, player_week


def injured_weeks(seasons):
    """(gsis_id, season) -> {"out": weeks hurt for sure, "listed": weeks on the injury report at all}.

    "out": on an injury list (IR, PUP, NFI) or ruled Out/Doubtful.
    "listed": any injury-report status (e.g. Questionable). These weeks are excused only if he then
    didn't play (season_line checks), so a game missed while Questionable counts as an injury.
    """
    out = {}
    rw = nfl.load_rosters_weekly(seasons).filter(
        (pl.col("game_type") == "REG") & pl.col("status_description_abbr").is_in(list(INJURY_RESERVE_CODES)))
    for g, s, w in rw.select(["gsis_id", "season", "week"]).iter_rows():
        out.setdefault((g, s), {"out": set(), "listed": set()})["out"].add(w)
    inj = nfl.load_injuries(seasons).filter((pl.col("game_type") == "REG") & pl.col("report_status").is_not_null())
    for g, s, w, status in inj.select(["gsis_id", "season", "week", "report_status"]).iter_rows():
        rec = out.setdefault((g, s), {"out": set(), "listed": set()})
        (rec["out"] if status in INJURY_REPORT_OUT else rec["listed"]).add(w)
    return out


def roster_teams(seasons):
    """(gsis_id, season) -> the team he was on most weeks (for seasons with no snaps, e.g. a year on IR)."""
    rw = nfl.load_rosters_weekly(seasons).filter(
        (pl.col("game_type") == "REG") & (pl.col("status") != "CUT"))
    best = {}
    for g, s, team, n in rw.group_by(["gsis_id", "season", "team"]).len().iter_rows():
        if (g, s) not in best or n > best[(g, s)][1]:
            best[(g, s)] = (team, n)
    return {k: v[0] for k, v in best.items()}


def nfl_grades():
    """gsis_id -> {season: grade} from the copied Sports_Analytics_Project tables."""
    env = read_env(REPO / ".env")
    conn = pymysql.connect(host=env["TIDB_HOST"], port=int(env["TIDB_PORT"]), user=env["TIDB_USER"],
                           password=env["TIDB_PASSWORD"], database=env["TIDB_DATABASE"],
                           ssl=ssl.create_default_context(), connect_timeout=20)
    grades = {}
    try:
        with conn.cursor() as c:
            for table in ("player_grades", "offense_player_grades"):
                c.execute(f"SELECT gsis_id, room, season, grade, trend_seasons, trend_grades FROM `{table}`")
                for gsis, room, season, grade, t_seasons, t_grades in c.fetchall():
                    if not gsis:
                        continue
                    g = grades.setdefault(gsis, {"room": GRADE_ROOM.get(room), "by_season": {}})
                    if t_seasons and t_grades:
                        for s, v in zip(str(t_seasons).split(","), str(t_grades).split(",")):
                            if s.strip() and v.strip():
                                g["by_season"][int(s)] = float(v)
                    if grade is not None:
                        g["by_season"][int(season)] = float(grade)
    finally:
        conn.close()
    return grades


# ---------------------------------------------------------------- model

def grade_multiplier(grade):
    return GRADE_MIN_MULT + (GRADE_MAX_MULT - GRADE_MIN_MULT) * grade / 100


def hurt_weeks_for(side, weeks_played, team_weeks, injury):
    """Weeks he was hurt: ruled out / on an injury list, or on the injury report and then didn't play."""
    idx = 0 if side == "off" else 1
    hurt = set(injury.get("out", ())) | {
        w for w in injury.get("listed", ())
        if w not in weeks_played or not weeks_played[w][1 + idx]}
    return {w for w in hurt if w in team_weeks}


def season_line(side, weeks_played, team_weeks, hurt, counted_fraction):
    """Snap share and the share of the season that counts.

    Healthy weeks always count. Injured weeks are excused, except `counted_fraction` of each one, which
    counts as a game he didn't play (used once a player passes the career injury allowance).
    """
    if not team_weeks:
        return 0.0, 0, 0.0
    idx = 0 if side == "off" else 1
    available = [w for w in team_weeks if w not in hurt]
    team_snaps = sum(team_weeks[w][idx] for w in available)
    team_snaps += counted_fraction * sum(team_weeks[w][idx] for w in hurt)
    played = sum(weeks_played[w][1 + idx] for w in available if w in weeks_played)
    share = min(1.0, played / team_snaps) if team_snaps else 0.0
    games = sum(1 for w in available if w in weeks_played)
    counted = (len(available) + counted_fraction * len(hurt)) / len(team_weeks)
    return share, games, counted


def starter_shares(seasons, rooms_by_gsis):
    """Typical starter snap share by room: average of the top (32 x starters per team) players' snap
    share in the games they played (8+ games). A player at or above it is a full-time starter."""
    sc = nfl.load_snap_counts(seasons).filter(pl.col("game_type") == "REG")
    pfr_to_gsis = pfr_gsis_map(seasons)
    agg = sc.group_by(["pfr_player_id", "season"]).agg(
        pl.col("offense_pct").mean(), pl.col("defense_pct").mean(), pl.len().alias("g"))
    by_room = {}
    for pfr, season, off, dfn, g in agg.iter_rows():
        room = rooms_by_gsis.get(pfr_to_gsis.get(pfr))
        if room not in STARTERS_PER_TEAM or g < 8:
            continue
        share = off if room in OFFENSE else dfn
        by_room.setdefault((room, season), []).append(share or 0.0)
    out = {}
    for room, n in STARTERS_PER_TEAM.items():
        season_avgs = []
        for (r, _season), vals in by_room.items():
            if r == room:
                top = sorted(vals, reverse=True)[: 32 * n]
                season_avgs.append(sum(top) / len(top))
        out[room] = sum(season_avgs) / len(season_avgs) if season_avgs else 1.0
    return out


def school_key(name):
    s = re.sub(r"[^a-z ]", "", str(name or "").lower())
    s = re.sub(r"\b(university|of|the|college)\b", "", s).replace("state", "st")
    return re.sub(r"\s+", "", s)


def same_school(a, b):
    a, b = school_key(a), school_key(b)
    return bool(a and b) and (a in b or b in a or difflib.SequenceMatcher(None, a, b).ratio() >= 0.75)


def match_board(board_players, candidates):
    """Match your board players to NFL entries (draft picks or undrafted rookies).

    1. Same name, 2. near-spelling (import_data.near_match), 3. last-name overlap at the same college
       (catches Van Pran / Van Pran-Granger, and Johnny / Jer'Zhan Newton).
    Returns {board slug: candidate}.
    """
    by_key = {name_key(c["name"]): c for c in candidates}
    names = [c["name"] for c in candidates]
    out, taken = {}, set()
    for b in board_players:
        c = by_key.get(name_key(b["name"]))
        if c is None:
            near = near_match(b["name"], names)
            c = by_key.get(name_key(near)) if near else None
        if c is None:
            last = split_name(b["name"])[1]
            hits = [x for x in candidates
                    if last and (last in split_name(x["name"])[1] or split_name(x["name"])[1] in last)
                    and same_school(b.get("school"), x.get("college"))]
            c = hits[0] if len(hits) == 1 else None
        if c is not None and id(c) not in taken:
            out[b["slug"]] = c
            taken.add(id(c))
    return out


def undrafted_rookies(year):
    """Undrafted players who entered the league in `year` (from weekly rosters)."""
    rw = nfl.load_rosters_weekly(list(range(year, nfl.get_current_season() + 1)))
    rw = rw.filter((pl.col("entry_year") == year) & pl.col("draft_number").is_null() & pl.col("gsis_id").is_not_null())
    # Some weekly rows are missing the PFR id (needed to find snap counts): take it from any row that has it
    pfr_ids = rw.group_by("gsis_id").agg(pl.col("pfr_id").drop_nulls().first().alias("pfr_any"))
    rw = rw.sort(["season", "week"]).unique("gsis_id", keep="first").join(pfr_ids, on="gsis_id", how="left")
    rw = rw.with_columns(pl.coalesce("pfr_id", "pfr_any").alias("pfr_id"))
    return [{"name": r["full_name"], "pick": None, "round": None, "team": r["team"], "draft_position": r["position"],
             "college": r["college"], "gsis_id": r["gsis_id"], "pfr_id": r["pfr_id"]}
            for r in rw.iter_rows(named=True)]


def recency_weight(season, current, room=None):
    weights = RECENCY_WEIGHTS.get(room, RECENCY_WEIGHTS["default"])
    ago = current - season
    return weights[ago] if ago < len(weights) else weights[-1]


def build(year, entries, through=None):
    """Value every entry (draft picks plus undrafted players from your board).

    `through` stops at that season and treats it as the current one (used to measure past classes at the
    same stage of their careers, for the expected-score curve).
    """
    current = through or nfl.get_current_season()
    seasons = list(range(year, current + 1))
    pos_value = positional_values(nfl.get_current_season())
    rooms_by_contract = contract_rooms()
    team_week, player_week = weekly_snaps(seasons)
    hurt_weeks = injured_weeks(seasons)
    roster_team = roster_teams(seasons)
    grades = nfl_grades()  # used for each player's position room
    # Weeks played so far in each season (the current one is partial), for weighting it
    season_fraction = {}
    for (s, _team), weeks in team_week.items():
        season_fraction[s] = max(season_fraction.get(s, 0), min(1.0, len(weeks) / SEASON_GAMES))
    single_season, grade_rows = season_grades(seasons, rooms_by_contract, season_fraction)
    # Offensive linemen drafted as plain "OL": tackle or interior from their depth chart spot
    ol_room = {g: r["room"] for (g, _s), r in grade_rows.items() if r.get("room") in ("OT", "IOL")}
    starter_share = starter_shares(seasons, rooms_by_contract)

    players, not_graded = [], []
    for base in entries:
        gsis, pfr = base["gsis_id"], base["pfr_id"]
        room = (grades.get(gsis, {}).get("room") or rooms_by_contract.get(gsis)
                or DRAFT_ROOM.get(base["draft_position"]))
        if room == "OL":
            room = ol_room.get(gsis, "IOL")
        base = {**base, "room": room}
        if room in SPECIALISTS or room is None:
            continue
        if room in NOT_GRADED:
            not_graded.append(base)
            continue
        side = "off" if room in OFFENSE else "def"
        # Pass 1: each season's team, games and injured weeks
        lines = []
        for s in seasons:
            weeks_played = player_week.get((pfr, s), {})
            # His team that season: where he played the most snaps, else where he was rostered
            if weeks_played:
                by_team = {}
                for _w, (team, off, dfn) in weeks_played.items():
                    by_team[team] = by_team.get(team, 0) + off + dfn + 1
                team = max(by_team, key=by_team.get)
            else:
                team = roster_team.get((gsis, s))
            team_weeks = team_week.get((s, team), {}) if team else {}
            hurt = hurt_weeks_for(side, weeks_played, team_weeks, hurt_weeks.get((gsis, s), {}))
            lines.append((s, team, weeks_played, team_weeks, hurt))
        # Injury allowance: excuse injured games up to INJURY_ALLOWANCE of his team's games so far;
        # beyond that, the extra share of injured games counts as games he didn't play.
        total_games = sum(len(tw) for _, _, _, tw, _ in lines)
        total_hurt = sum(len(h) for *_, h in lines)
        allowed = INJURY_ALLOWANCE * total_games
        counted_fraction = (total_hurt - allowed) / total_hurt if total_hurt > allowed else 0.0

        # Pass 2: value per season
        season_rows, weighted, total_weight = [], 0.0, 0.0
        past_grades = []
        for s, team, weeks_played, team_weeks, hurt in lines:
            share, games, counted = season_line(side, weeks_played, team_weeks, hurt, counted_fraction)
            # Snap share relative to a typical starter at his position (a 62% RB is full-time)
            rel_share = min(1.0, share / starter_share[room]) if starter_share.get(room) else share
            grade = single_season.get((gsis, s))
            # A season still in progress (non-QBs): blend its grade with his previous seasons' average in
            # proportion to games played, so an 8-game hot or cold start moves him, but less than a full
            # season would. Quarterbacks use this season's grade as is.
            grade_used = grade
            frac = season_fraction.get(s, 1.0)
            if room != "QB" and frac < 1.0 and grade is not None and past_grades:
                grade_used = frac * grade + (1 - frac) * sum(past_grades) / len(past_grades)
            if grade is not None:
                past_grades.append(grade)
            mult = grade_multiplier(grade_used if grade_used is not None else NEUTRAL_GRADE)
            value = pos_value[room] * rel_share * mult
            # Recent seasons count more (QB 5-2-1, others 3-2-1; the current season in full even
            # mid-season), and each season counts for the share of it that counts (see above).
            weight = recency_weight(s, current, room) * (counted if team_weeks else 1.0)
            season_rows.append({"season": s, "team": team, "snap_share": round(share, 3),
                                "starter_share": round(rel_share, 3), "games": games,
                                "games_missed_injured": len(hurt),
                                "injured_games_counted": round(counted_fraction * len(hurt), 1),
                                "grade": grade,
                                "grade_used": round(grade_used, 1) if grade_used is not None else None,
                                "weight": round(weight, 3), "value": round(value * 100, 3)})
            weighted += value * weight
            total_weight += weight
        player_value = weighted / total_weight if total_weight else 0.0
        for row in season_rows:
            row["score"] = round(SCORE_AT_MARKET * row["value"] / (pos_value[room] * 100), 1)
        players.append({**base, "positional_value": round(pos_value[room] * 100, 2),
                        "value": round(player_value * 100, 3),
                        "score": round(SCORE_AT_MARKET * player_value / pos_value[room], 1),
                        "seasons": season_rows})

    players.sort(key=lambda r: -r["value"])
    for i, r in enumerate(players, start=1):
        r["value_rank"] = i  # overall draft value (positional market value included)
    # Position rank: within his position in this class. Positional value is the same for everyone in a
    # room, so this is decided by playing time, performance and recent form: how the pick compares with
    # the other players a team could have taken at that position.
    by_room = {}
    for r in players:
        by_room.setdefault(r["room"], []).append(r)
    for room, group in by_room.items():
        for i, r in enumerate(group, start=1):  # already sorted by value
            r["position_rank"] = i
            r["position_count"] = len(group)
    return players, not_graded, pos_value, seasons, season_fraction, current


# Labels for score minus expected score (vs. pick, vs. your rank)
EXPECTATION_LABELS = [(30, "Steal"), (10, "Above"), (-10, "As expected"), (-30, "Below"), (float("-inf"), "Miss")]


def expectation_label(diff):
    """Steal: +30 or more. Above: +10 to +30. As expected: within 10 either way. Below: -10 to -30.
    Miss: worse than -30."""
    if diff >= 30:
        return "Steal"
    if diff >= 10:
        return "Above"
    if diff > -10:
        return "As expected"
    if diff >= -30:
        return "Below"
    return "Miss"


def attach_expectations(players):
    """Compare each score with what's expected at his draft slot and at your board rank.

    Expected scores come from app/data/expected_by_pick.json (scripts/expected_curve.py: past classes
    measured over their first three seasons). Your board rank is treated like a pick number.
    """
    path = OUT_DIR / "expected_by_pick.json"
    if not path.exists():
        print("expected_by_pick.json not found: run scripts/expected_curve.py for vs. pick / vs. your rank")
        return None
    curve = json.loads(path.read_text(encoding="utf-8"))
    by_pick = {int(k): v for k, v in curve["by_pick"].items()}
    last = max(by_pick)
    for p in players:
        if p.get("pick"):
            exp = by_pick[min(p["pick"], last)]
            p["expected_at_pick"] = exp
            p["vs_pick"] = round(p["score"] - exp, 1)
            p["vs_pick_label"] = expectation_label(p["vs_pick"])
        if p.get("board_rank"):
            exp = by_pick[min(p["board_rank"], last)]
            p["expected_at_board"] = exp
            p["vs_board"] = round(p["score"] - exp, 1)
            p["vs_board_label"] = expectation_label(p["vs_board"])
    return {"classes": curve["classes"], "seasons_measured": curve["seasons_measured"],
            "players": curve["players"], "labels": [[f, l] for f, l in EXPECTATION_LABELS[:-1]]}


def draft_entries(year):
    picks = nfl.load_draft_picks([year]).sort("pick")
    return [{"name": p["pfr_player_name"], "pick": p["pick"], "round": p["round"], "team": p["team"],
             "draft_position": p["position"], "college": p["college"],
             "gsis_id": p["gsis_id"], "pfr_id": p["pfr_player_id"]} for p in picks.iter_rows(named=True)]


def main(year):
    entries = draft_entries(year)

    board_path = OUT_DIR / f"{year}.json"
    board = json.loads(board_path.read_text(encoding="utf-8"))["players"] if board_path.exists() else []
    matched = match_board(board, entries)
    # Board players who weren't drafted: look for them among the undrafted rookies
    unmatched = [b for b in board if b["slug"] not in matched]
    udfa_matches = match_board(unmatched, undrafted_rookies(year))
    for slug, entry in udfa_matches.items():
        entry["undrafted"] = True
        entries.append(entry)
    matched.update(udfa_matches)
    not_found = [b["name"] for b in board if b["slug"] not in matched]

    board_by_slug = {b["slug"]: b for b in board}
    entry_to_board = {id(e): board_by_slug[slug] for slug, e in matched.items()}
    for e in entries:
        b = entry_to_board.get(id(e))
        if b:
            e["board_rank"] = b.get("rank")
            e["board_tier"] = b.get("tier")
            e["board_name"] = b["name"]
            e["slug"] = b["slug"] if b.get("reports") else None

    players, not_graded, pos_value, seasons, season_fraction, current = build(year, entries)
    expectation = attach_expectations(players)
    data = {
        "year": year,
        "seasons": seasons,
        "season_weights": {"QB": {s: recency_weight(s, current, "QB") for s in seasons},
                           "other": {s: recency_weight(s, current) for s in seasons}},
        "positional_value": {k: round(v * 100, 2) for k, v in pos_value.items()},
        "grade_multiplier": [GRADE_MIN_MULT, GRADE_MAX_MULT],
        "players": players,
        "not_graded": not_graded,
        "board_not_found": not_found,
        "expectation": expectation,
    }
    out = OUT_DIR / f"{year}_value.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    n_udfa = sum(1 for p in players + not_graded if p.get("undrafted"))
    print(f"[{year}] {len(players)} players valued ({n_udfa} undrafted from your board), "
          f"{len(not_graded)} offensive linemen not graded yet -> {out.relative_to(REPO)}")
    print(f"[{year}] season weights (by recency): {data['season_weights']}")
    print(f"[{year}] board players matched by name/college fallback: "
          f"{[(e['board_name'], e['name']) for e in entries if e.get('board_name') and name_key(e['board_name']) != name_key(e['name'])]}")
    print(f"[{year}] on your board but not found in the NFL ({len(not_found)}): {', '.join(not_found)}")


if __name__ == "__main__":
    for y in [int(a) for a in sys.argv[1:]] or [2024]:
        main(y)
