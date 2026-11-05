from flask import Flask, abort, render_template

# Static files live in public/ so Vercel's CDN serves them; Flask serves the
# same folder at /static when running locally.
app = Flask(__name__, static_folder="../public/static", static_url_path="/static")

CURRENT_YEAR = 2027
YEARS = [2027, 2026, 2025, 2024]
POSITIONS = ["QB", "RB", "WR", "TE", "OT", "IOL", "IDL", "EDGE", "LB", "CB", "SAF"]


@app.context_processor
def inject_site_data():
    return {"years": YEARS, "current_year": CURRENT_YEAR}


@app.route("/")
def index():
    return render_template("index.html", year=CURRENT_YEAR, positions=POSITIONS)


@app.route("/<int:year>")
def year_page(year):
    if year not in YEARS:
        abort(404)
    return render_template("index.html", year=year, positions=POSITIONS)


@app.route("/about")
def about():
    return render_template("about.html", active="about")
