import os
import json
import threading
import logging
import time
from datetime import datetime
import pandas as pd
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, Response
)
from ml.model import train_model, predict_book
from data.generate_data import generate_library_data

app = Flask(__name__)
app.secret_key = "library_big_data_secret_2025"

UPLOAD_FOLDER = "data"
METRICS_PATH  = "data/metrics.json"
STATUS_PATH   = "data/status.json"
INFO_PATH     = "data/dataset_info.json"
LOG_PATH      = "data/app.log"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/plots", exist_ok=True)

# настраиваем логирование в файл
file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)


def _log(msg):
    app.logger.info(msg)


def _set_status(state, message=""):
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump({"state": state, "message": message}, f)


def _get_status():
    if not os.path.exists(STATUS_PATH):
        return {"state": "idle", "message": ""}
    with open(STATUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_dataset_info(n_rows, source="generated"):
    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump({"n_rows": n_rows, "source": source}, f)


def _get_dataset_info():
    if not os.path.exists(INFO_PATH):
        return {"n_rows": 0, "source": "unknown"}
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _train_in_background(csv_path):
    try:
        _set_status("training", "чтение данных...")
        _log(f"начало обучения, файл: {csv_path}")
        start = time.time()

        df = pd.read_csv(csv_path)
        _log(f"данные загружены: {len(df)} строк, {len(df.columns)} столбцов")
        _set_status("training", f"обучение на {len(df)} строках...")

        metrics = train_model(df)

        elapsed = round(time.time() - start, 1)
        _log(f"обучение завершено за {elapsed} сек, accuracy={metrics['accuracy']}")

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f)

        _set_status("done", f"модель обучена за {elapsed} сек")

    except Exception as e:
        _log(f"ошибка обучения: {str(e)}")
        _set_status("error", str(e))


@app.route("/")
def index():
    dataset_exists = os.path.exists("data/books.csv")
    model_exists   = os.path.exists("ml/model.pkl")
    kaggle_exists  = (
        os.path.exists("data/books_goodreads.csv") or
        os.path.exists("data/goodreads_raw/books.csv") or
        os.path.exists("goodreadsbooks.zip")
    )
    status       = _get_status()
    dataset_info = _get_dataset_info()
    return render_template("index.html",
                           dataset_exists=dataset_exists,
                           model_exists=model_exists,
                           kaggle_exists=kaggle_exists,
                           status=status,
                           dataset_info=dataset_info)


@app.route("/generate", methods=["POST"])
def generate():
    try:
        n_rows = int(request.form.get("n_rows", 100000))
        n_rows = max(1000, min(500000, n_rows))
        _log(f"генерация датасета: {n_rows} строк")
        generate_library_data(n_rows=n_rows, output_path="data/books.csv")
        _save_dataset_info(n_rows, source="сгенерирован")
        _set_status("idle", "")
        flash(f"датасет сгенерирован: {n_rows:,} строк".replace(",", " "), "success")
    except Exception as e:
        _log(f"ошибка генерации: {str(e)}")
        flash(f"ошибка при генерации: {str(e)}", "error")
    return redirect(url_for("index"))


@app.route("/use_kaggle", methods=["POST"])
def use_kaggle():
    try:
        _log("адаптация kaggle датасета goodreads")

        # распаковать zip если нужно
        if os.path.exists("goodreadsbooks.zip") and not os.path.exists("data/goodreads_raw/books.csv"):
            import zipfile
            _log("распаковка goodreadsbooks.zip")
            with zipfile.ZipFile("goodreadsbooks.zip", "r") as z:
                z.extractall("data/goodreads_raw")

        # если файл на месте — копируем
        if os.path.exists("data/goodreads_raw/books.csv") and not os.path.exists("data/books_goodreads.csv"):
            import shutil
            shutil.copy("data/goodreads_raw/books.csv", "data/books_goodreads.csv")
            _log("скопирован data/goodreads_raw/books.csv -> data/books_goodreads.csv")

        from data.adapt_goodreads import adapt_goodreads
        df = adapt_goodreads(output_path="data/books.csv")
        n_rows = len(df)
        _save_dataset_info(n_rows, source="kaggle goodreads")
        _set_status("idle", "")
        popular = int(df["is_popular"].sum())
        _log(f"kaggle датасет адаптирован: {n_rows} строк, популярных: {popular}")
        flash(f"kaggle датасет загружен: {n_rows:,} строк, популярных: {popular}".replace(",", " "), "success")
    except Exception as e:
        _log(f"ошибка kaggle адаптации: {str(e)}")
        flash(f"ошибка: {str(e)}", "error")
    return redirect(url_for("index"))


@app.route("/train", methods=["POST"])
def train():
    csv_path = "data/books.csv"
    if not os.path.exists(csv_path):
        flash("сначала подготовьте датасет", "error")
        return redirect(url_for("index"))

    status = _get_status()
    if status.get("state") == "training":
        flash("обучение уже запущено", "error")
        return redirect(url_for("index"))

    dataset_info = _get_dataset_info()
    n_rows = dataset_info.get("n_rows", 0)

    _log(f"запуск обучения, строк: {n_rows}")

    t = threading.Thread(
        target=_train_in_background,
        args=(csv_path,),
        daemon=True
    )
    t.start()

    return render_template("training.html", n_rows=n_rows)


@app.route("/status")
def status():
    s = _get_status()
    return json.dumps(s), 200, {"Content-Type": "application/json"}


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("файл не выбран", "error")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("файл не выбран", "error")
        return redirect(url_for("index"))

    if not file.filename.endswith(".csv"):
        flash("загрузите csv файл", "error")
        return redirect(url_for("index"))

    filepath = os.path.join(UPLOAD_FOLDER, "books.csv")
    file.save(filepath)
    _log(f"загружен файл: {file.filename}")

    try:
        df = pd.read_csv(filepath)
        required_cols = {
            "genre", "author_popularity", "year_published",
            "price", "page_count", "times_borrowed",
            "rating", "is_popular"
        }
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            flash(f"отсутствуют столбцы: {', '.join(missing)}", "error")
            return redirect(url_for("index"))

        n_rows = len(df)
        _save_dataset_info(n_rows, source=file.filename)
        _set_status("idle", "")
        _log(f"файл принят: {n_rows} строк")
        flash(f"файл загружен: {n_rows:,} строк".replace(",", " "), "success")
        return redirect(url_for("index"))

    except Exception as e:
        _log(f"ошибка загрузки: {str(e)}")
        flash(f"ошибка: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/results")
def results():
    if not os.path.exists(METRICS_PATH):
        flash("сначала обучите модель", "error")
        return redirect(url_for("index"))

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return render_template("results.html", metrics=metrics)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not os.path.exists("ml/model.pkl"):
        flash("сначала обучите модель", "error")
        return redirect(url_for("index"))

    result = None

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
            _log(f"предсказание: {result['label']}, вероятность: {result['probability']}%")
        except Exception as e:
            _log(f"ошибка предсказания: {str(e)}")
            flash(f"ошибка: {str(e)}", "error")

    return render_template("predict.html", result=result)


@app.route("/logs")
def logs():
    log_content = ""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            log_content = "".join(lines[-200:])  # последние 200 строк
    return render_template("logs.html", log_content=log_content)


@app.route("/logs/clear", methods=["POST"])
def clear_logs():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("")
    _log("логи очищены")
    flash("логи очищены", "success")
    return redirect(url_for("logs"))


@app.route("/logs/raw")
def logs_raw():
    if not os.path.exists(LOG_PATH):
        return "", 200
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        content = "".join(lines[-200:])
    return Response(content, mimetype="text/plain")


if __name__ == "__main__":
    _log("приложение запущено")
    app.run(debug=True, use_reloader=False)
