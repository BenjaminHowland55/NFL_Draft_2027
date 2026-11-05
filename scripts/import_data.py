"""Import draft rankings (Excel) and scouting reports (Word) into app/data/<year>.json.

Run from the repo root:
    python scripts/import_data.py            # all years
    python scripts/import_data.py 2025       # one year

Rules this script follows:
- Scouting report text is copied word for word from the .docx. Nothing is
  written, summarized, or corrected. A verification step fails the import if
  any non-empty paragraph from a report is missing from the output.
- Header details (position, school, class, height, weight) and grades come
  from the Excel workbook, not the report header.
- The PFF column is not exported (third-party data).
- Images in the reports are not exported.

Requires: pip install python-docx openpyxl
"""
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote

import docx
import openpyxl

DESKTOP = Path.home() / "OneDrive" / "Desktop"
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "app" / "data"

SOURCES = {
    2024: {"workbook": DESKTOP / "DRAFT/2024/2024 NFL Draft Prospects.xlsx", "reports": None},
    2025: {"workbook": DESKTOP / "DRAFT/2025/2025 NFL Draft Prospects.xlsx", "reports": DESKTOP / "DRAFT/Scouting Reports"},
    2026: {"workbook": DESKTOP / "2026 Draft/2026 NFL Draft.xlsx", "reports": DESKTOP / "2026 Draft"},
}

POSITION_SHEETS = ["QB", "RB", "WR", "TE", "OT", "IOL", "IDL", "EDGE", "LB", "CB", "SAF"]

# Excel header -> output field. PFF is deliberately absent.
COLUMNS = {
    "name": "full_name",
    "first name": "first",
    "last name": "last",
    "position": "position",
    "school": "school",
    "class": "class",
    "height": "height",
    "weight": "weight",
    "size": "size_score",
    "size score": "size_score",
    "athleticism": "athleticism",
    "tape grade": "tape_grade",
    "grade": "grade",
    "report": "report_link",
}


# ---------------------------------------------------------------- helpers

def name_key(name):
    """Loose key for matching the same player across sheets/files."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", "", s)


def slugify(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"['’.]", "", s)
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def num(v):
    if isinstance(v, (int, float)):
        return round(float(v), 3)
    return None


def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = str(v).strip()
    return s or None


# ---------------------------------------------------------------- Excel

def read_position_sheets(wb):
    players = []
    for sheet in POSITION_SHEETS:
        if sheet not in wb.sheetnames:
            continue
        # Read the player table only: stop at the first blank row, which
        # separates it from the formula/weight rows below.
        rows = []
        for r in wb[sheet].iter_rows(values_only=True):
            if any(c is not None for c in r):
                rows.append(r)
            elif rows:
                break
        if not rows:
            continue
        header = [str(h).strip().lower() if h is not None else "" for h in rows[0]]
        for pos_rank, row in enumerate(rows[1:], start=1):
            rec = {}
            for i, h in enumerate(header):
                field = COLUMNS.get(h)
                if field and i < len(row):
                    rec[field] = row[i]
            first, last = clean(rec.get("first")), clean(rec.get("last"))
            full = clean(rec.get("full_name"))  # 2026 sheets use a single Name column
            if first and last:
                name = f"{first} {last}"
            elif full and " " in full:
                name = full
            else:
                continue
            if name.replace(".", "").replace(" ", "").isdigit():
                continue  # weight/formula rows at the bottom of the sheet
            players.append({
                "name": name,
                "group": sheet,
                "position": clean(rec.get("position")) or sheet,
                "school": clean(rec.get("school")),
                "class": clean(rec.get("class")),
                "height": clean(rec.get("height")),
                "weight": clean(rec.get("weight")),
                "size_score": num(rec.get("size_score")),
                "athleticism": num(rec.get("athleticism")),
                "tape_grade": num(rec.get("tape_grade")),
                "grade": num(rec.get("grade")),
                "report_link": clean(rec.get("report_link")),
            })
    # Position rank = order by grade within each group
    for g in POSITION_SHEETS:
        grp = sorted((p for p in players if p["group"] == g), key=lambda p: -(p["grade"] or 0))
        for i, p in enumerate(grp, start=1):
            p["position_rank"] = i
    return players


BOARD_ENTRY = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")


def read_big_board(wb):
    if "Big Board" not in wb.sheetnames:
        return [], []
    rows = [r for r in wb["Big Board"].iter_rows(values_only=True)]
    rows = [r for r in rows if any(c is not None for c in r)]
    header = rows[0]
    tier_cols = [(i, str(c).strip()) for i, c in enumerate(header) if c and str(c).strip().lower().startswith("tier")]
    tiers, entries = [], []
    for col, tier in tier_cols:
        tiers.append(tier)
        for r in rows[1:]:
            cell = r[col] if col < len(r) else None
            if not cell:
                continue
            m = BOARD_ENTRY.match(str(cell))
            if not m:
                continue
            rank, rest = int(m.group(1)), m.group(2)
            name, _, detail = rest.partition(",")
            grade = r[col + 2] if col + 2 < len(r) else None
            entries.append({"rank": rank, "tier": tier, "name": name.strip(), "detail": detail.strip(" ,"), "grade": num(grade)})
    entries.sort(key=lambda e: e["rank"])
    return tiers, entries


# ---------------------------------------------------------------- Word reports

def unique_cells(row):
    seen, out = set(), []
    for c in row.cells:
        if id(c._tc) not in seen:
            seen.add(id(c._tc))
            out.append(c)
    return out


def is_list(p):
    ppr = p._p.pPr
    return ppr is not None and ppr.numPr is not None


def texts(cell):
    return [p.text.strip() for p in cell.paragraphs if p.text.strip()]


def section(cell):
    """A labelled block: first paragraph is the heading, the rest is body text."""
    paras = [p for p in cell.paragraphs if p.text.strip()]
    if not paras:
        return None
    title = paras[0].text.strip()
    body = [{"text": p.text.strip(), "bullet": is_list(p)} for p in paras[1:]]
    return {"title": title.rstrip(":").strip(), "raw_title": title, "body": body}


SCORE = re.compile(r"^\s*([\d.]*)\s*/\s*(\d+)\s*$")


def score_cell(cell):
    t = texts(cell)
    label = t[0] if t else ""
    value = " ".join(t[1:])
    m = SCORE.match(value)
    if m:
        return {"name": label, "score": m.group(1) or None, "max": m.group(2), "raw": value}
    return {"name": label, "score": value or None, "max": None, "raw": value}


def parse_combine(table):
    labels = [" ".join(texts(c)) for c in unique_cells(table.rows[0])]
    values = [" ".join(texts(c)) for c in unique_cells(table.rows[1])] if len(table.rows) > 1 else []
    return [{"label": l, "value": values[i] if i < len(values) else ""} for i, l in enumerate(labels) if l]


def parse_report(path):
    d = docx.Document(str(path))
    tables = d.tables
    report = {"sections": [], "traits": [], "profiles": [], "combine": [], "grades": [], "notes": []}

    first_row = [" ".join(texts(c)) for c in unique_cells(tables[1].rows[0])]
    if first_row and first_row[0].startswith("Frame"):
        # 2026 format: Frame/Athleticism/Strength, then Strengths/Summary/Weaknesses
        report["format"] = "2026"
        report["traits"] = [s for s in (section(c) for c in unique_cells(tables[1].rows[0])) if s]
        report["sections"] = [s for s in (section(c) for c in unique_cells(tables[1].rows[1])) if s]
        report["combine"] = parse_combine(tables[2])
        for c in unique_cells(tables[3].rows[0]):
            t = texts(c)
            if t:
                report["grades"].append({"label": t[0], "value": t[1] if len(t) > 1 else None, "note": " ".join(t[2:]) or None})
    else:
        # 2025 format: Strengths/Summary/Weaknesses, profile scores, combine, grades
        report["format"] = "2025"
        report["sections"] = [s for s in (section(c) for c in unique_cells(tables[1].rows[0])) if s]
        prof = tables[2]
        heads = [" ".join(texts(c)) for c in unique_cells(prof.rows[0])]
        cols = [{"title": h, "traits": [], "total": None} for h in heads]
        body_rows = prof.rows[1:]
        for ri, row in enumerate(body_rows):
            for ci, c in enumerate(unique_cells(row)[: len(cols)]):
                sc = score_cell(c)
                if ri == len(body_rows) - 1:
                    cols[ci]["total"] = sc
                else:
                    cols[ci]["traits"].append(sc)
        report["profiles"] = cols
        report["combine"] = parse_combine(tables[3])
        g = tables[4]
        labels = [" ".join(texts(c)) for c in unique_cells(g.rows[0])]
        values = [texts(c) for c in unique_cells(g.rows[1])] if len(g.rows) > 1 else []
        for i, label in enumerate(labels):
            if not label:
                continue
            if label.startswith("*"):
                report["notes"].append(label)
                continue
            v = values[i] if i < len(values) else []
            report["grades"].append({"label": label, "value": v[0] if v else None, "note": " ".join(v[1:]) or None})

    verify_report(d, report, path)
    return report


def verify_report(d, report, path):
    """Every non-empty paragraph outside the header table must appear in the output."""
    out = []
    for s in report["traits"] + report["sections"]:
        out.append(s["raw_title"])
        out.extend(b["text"] for b in s["body"])
    for p in report["profiles"]:
        out.append(p["title"])
        for t in p["traits"] + ([p["total"]] if p["total"] else []):
            out.append(t["name"])
            out.append(t["raw"])
    for c in report["combine"]:
        out += [c["label"], c["value"]]
    for g in report["grades"]:
        out += [g["label"], g["value"] or "", g["note"] or ""]
    out += report["notes"]
    blob = "\n".join(x for x in out if x)
    missing = []
    for t in d.tables[1:]:
        for row in t.rows:
            for c in unique_cells(row):
                for p in c.paragraphs:
                    txt = p.text.strip()
                    if txt and txt not in blob:
                        missing.append(txt)
    if missing:
        raise SystemExit(f"VERIFY FAILED for {path}: {len(missing)} paragraph(s) not exported, e.g. {missing[0]!r}")


VARIANT = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")  # e.g. "Travis Hunter (WR)"


def index_reports(root):
    """name_key -> list of (label, path). A file named 'Name (WR)' is an extra report labelled 'WR'."""
    files = {}
    if not root or not Path(root).exists():
        return files
    for p in sorted(Path(root).glob("*/*.docx")):
        m = VARIANT.match(p.stem)
        name, label = (m.group(1), m.group(2)) if m else (p.stem, None)
        files.setdefault(name_key(name), []).append((label, p))
    for v in files.values():
        v.sort(key=lambda lp: lp[0] is not None)  # unlabelled (main) report first
    return files


def split_name(name):
    """(first, last) keys, ignoring suffixes like Jr./II."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    parts = [re.sub(r"[^a-z0-9-]", "", w) for w in s.split()]
    parts = [w for w in parts if w]
    if not parts:
        return "", ""
    return "".join(parts[:-1]), parts[-1]


def near_match(name, candidates):
    """Find the one candidate name that is the same person spelled differently.

    Requires the same last name, plus first names that are close
    (one contains the other, e.g. Vega/Olaivavega, or nearly the same
    spelling, e.g. Ruben/Rueben). If there is no single clear match,
    returns None rather than guessing.
    """
    first, last = split_name(name)
    hits = []
    for cand in candidates:
        cf, cl = split_name(cand)
        if not first or not cf:
            continue
        if cl == last:
            # Same last name, close first name (Vega/Olaivavega, Ruben/Rueben)
            close = first in cf or cf in first or difflib.SequenceMatcher(None, first, cf).ratio() >= 0.8
        elif cf == first:
            # Same first name, last name with a small typo (Kelin/Klein)
            close = difflib.SequenceMatcher(None, last, cl).ratio() >= 0.8
        else:
            close = False
        if close:
            hits.append(cand)
    if len(hits) == 1:
        return hits[0]
    # Last resort: a unique, distinctive (hyphenated) last name, e.g.
    # "Rabbit Taylor-Demerson" = "Dadrion Taylor-Demerson".
    if "-" in last:
        same_last = [c for c in candidates if split_name(c)[1] == last]
        if len(same_last) == 1:
            return same_last[0]
    return None


def report_files_for(player, files, file_names):
    # Match by player name first; the sheet's hyperlink is only a fallback,
    # because a few links point at the wrong file.
    found = files.get(name_key(player["name"]))
    if found:
        return found
    link = player.get("report_link")
    if link:
        stem = Path(unquote(link.replace("\\", "/"))).stem
        if name_key(stem) in files:
            return files[name_key(stem)]
    near = near_match(player["name"], file_names)
    if near:
        player.setdefault("_notes", []).append(f"report file '{near}' matched by near-spelling")
        return files[name_key(near)]
    return []


# ---------------------------------------------------------------- build

def build_year(year):
    src = SOURCES[year]
    try:
        wb = openpyxl.load_workbook(src["workbook"], data_only=True, read_only=True)
    except PermissionError:
        print(f"[{year}] SKIPPED: {src['workbook']} is locked. Close it in Excel and run again.")
        return False
    players = read_position_sheets(wb)
    tiers, board = read_big_board(wb)
    files = index_reports(src["reports"])

    by_key = {name_key(p["name"]): p for p in players}
    owner_of_file = {}  # report path -> player
    # Player name as written in each report's file name (without "(WR)"-style suffixes)
    file_names = []
    for lst in files.values():
        stem = lst[0][1].stem
        m = VARIANT.match(stem)
        file_names.append(m.group(1) if m else stem)
    match_notes = []

    def attach_reports(p):
        found = report_files_for(p, files, file_names)
        match_notes.extend(f"{p['name']}: {n}" for n in p.pop("_notes", []))
        reports = []
        for label, f in found:
            r = parse_report(f)
            # A player with several reports (e.g. EDGE/ and LB/ folders): label by folder.
            if label is None and len(found) > 1:
                label = f.parent.name
            r["label"] = label
            r["source_file"] = f"{f.parent.name}/{f.name}"
            reports.append(r)
            owner_of_file[f] = p
        if reports:
            p["reports"] = reports

    for p in players:
        attach_reports(p)

    # Match big board entries to players: by name, else by owning the same
    # report file (catches spelling differences between the board and sheets).
    board_out = []
    for e in board:
        k = name_key(e["name"])
        p = by_key.get(k)
        if p is None and k in files:
            p = next((owner_of_file[f] for _, f in files[k] if f in owner_of_file), None)
        if p is None:
            near = near_match(e["name"], [q["name"] for q in players])
            if near:
                p = by_key[name_key(near)]
                match_notes.append(f"big board '{e['name']}' matched to sheet '{near}' by near-spelling")
        if p is None and k in files:
            # On the big board with a report, but missing from the position sheets:
            # use the big board's own "POS, School" details.
            pos, _, school = e["detail"].partition(",")
            pos = pos.strip()
            folder = files[k][0][1].parent.name
            p = {"name": e["name"], "group": pos if pos in POSITION_SHEETS else (folder if folder in POSITION_SHEETS else None),
                 "position": pos or None, "school": school.strip() or None, "class": None, "height": None,
                 "weight": None, "size_score": None, "athleticism": None, "tape_grade": None,
                 "grade": e["grade"], "report_link": None, "board_only": True}
            players.append(p)
            attach_reports(p)
        if p:
            p["rank"], p["tier"] = e["rank"], e["tier"]
        board_out.append({**e, "player": p})
    used_files = set(owner_of_file)

    # Slugs (unique within the year)
    seen = {}
    for p in players:
        base = slugify(p["name"])
        seen[base] = seen.get(base, 0) + 1
        p["slug"] = base if seen[base] == 1 else f"{base}-{seen[base]}"
        p.pop("report_link", None)
    for e in board_out:
        p = e.pop("player")
        e["slug"] = p["slug"] if p else None

    all_files = {f for lst in files.values() for _, f in lst}
    orphans = sorted(f.name for f in all_files - used_files)
    data = {
        "year": year,
        "tiers": tiers,
        "big_board": board_out,
        "players": players,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{year}.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    n_reports = sum(len(p.get("reports", [])) for p in players)
    unmatched_board = [e["name"] for e in board_out if not e["slug"]]
    print(f"[{year}] {len(players)} players, {len(board_out)} big board entries, {n_reports} reports -> {out.relative_to(REPO)}")
    if unmatched_board:
        print(f"[{year}]   big board names not found in position sheets ({len(unmatched_board)}): {', '.join(unmatched_board)}")
    if orphans:
        print(f"[{year}]   report files not linked to any sheet row ({len(orphans)}): {', '.join(orphans)}")
    for n in match_notes:
        print(f"[{year}]   near-spelling match: {n}")
    return True


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or sorted(SOURCES)
    ok = all([build_year(y) for y in years])
    sys.exit(0 if ok else 1)
