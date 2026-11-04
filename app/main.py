from flask import Flask, render_template

# Static files live in public/ so Vercel's CDN serves them; Flask serves the
# same folder at /static when running locally.
app = Flask(__name__, static_folder="../public/static", static_url_path="/static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")
