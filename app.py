import os
import pandas as pd
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, session
)
from ml.model import train_model, predict_book

app = Flask(__name__)
app.secret_key = "library_big_data_secret_2025"

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/plots", exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("file not selected", "error")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("file not selected", "error")
        return redirect(url_for("index"))

    if not file.filename.endswith(".csv"):
        flash("please upload a csv file", "error")
        return redirect(url_for("index"))

    filepath = os.path.join(UPLOAD_FOLDER, "uploaded_books.csv")
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        required_cols = {"genre", "author_popularity", "year_published",
                         "price", "page_count", "times_borrowed",
                         "rating", "is_popular"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            flash(f"missing columns: {', '.join(missing)}", "error")
            return redirect(url_for("index"))

        metrics = train_model(df)
        session["metrics"] = metrics
        flash("model trained successfully", "success")
        return redirect(url_for("results"))

    except Exception as e:
        flash(f"error processing file: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/results")
def results():
    metrics = session.get("metrics")
    if not metrics:
        flash("please upload data and train the model first", "error")
        return redirect(url_for("index"))
    return render_template("results.html", metrics=metrics)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not os.path.exists("ml/model.pkl"):
        flash("please upload data and train the model first", "error")
        return redirect(url_for("index"))

    result = None
    genres = ["fiction", "science", "history", "biography",
              "children", "fantasy", "detective", "romance",
              "psychology", "technology"]

    if request.method == "POST":
        try:
            result = predict_book(
                genre=request.form["genre"],
                author_popularity=request.form["author_popularity"],
                year_published=request.form["year_published"],
                price=request.form["price"],
                page_count=request.form["page_count"],
                times_borrowed=request.form["times_borrowed"],
                rating=request.form["rating"],
            )
        except Exception as e:
            flash(f"prediction error: {str(e)}", "error")

    return render_template("predict.html", result=result, genres=genres)


if __name__ == "__main__":
    app.run(debug=True)
