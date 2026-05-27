import pandas as pd
import numpy as np
import os


def adapt_goodreads(output_path="data/books.csv"):

    candidates = [
        "data/books_goodreads.csv",
        "data/goodreads_raw/books.csv",
    ]

    input_path = None
    for c in candidates:
        if os.path.exists(c):
            input_path = c
            break

    if input_path is None:
        raise FileNotFoundError(
            "не найден файл goodreads. "
            "распакуйте goodreadsbooks.zip в data/goodreads_raw/"
        )

    print(f"читаем файл: {input_path}")

    df = pd.read_csv(
        input_path,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace"
    )

    print(f"столбцы: {list(df.columns)}")
    print(f"строк до очистки: {len(df)}")

    df["average_rating"] = pd.to_numeric(
        df.get("average_rating", 3.0), errors="coerce"
    ).fillna(3.0)

    # в goodreads файле столбец num_pages иногда с пробелами
    num_pages_col = None
    for col in df.columns:
        if "num_pages" in col.strip():
            num_pages_col = col
            break

    if num_pages_col:
        df["num_pages_clean"] = pd.to_numeric(
            df[num_pages_col], errors="coerce"
        ).fillna(200)
    else:
        df["num_pages_clean"] = 200

    df["ratings_count"] = pd.to_numeric(
        df.get("ratings_count", 0), errors="coerce"
    ).fillna(0)

    df["text_reviews_count"] = pd.to_numeric(
        df.get("text_reviews_count", 0), errors="coerce"
    ).fillna(0)

    df = df[df["ratings_count"] > 0].copy()
    print(f"строк после очистки: {len(df)}")

    np.random.seed(42)

    genres = [
        "fiction", "science", "history", "biography",
        "children", "fantasy", "detective", "romance",
        "psychology", "technology"
    ]

    result = pd.DataFrame()
    result["book_id"] = range(1, len(df) + 1)
    result["genre"] = np.random.choice(genres, len(df))

    # популярность автора — через перцентили text_reviews_count
    reviews = df["text_reviews_count"].values.astype(float)
    reviews_rank = pd.Series(reviews).rank(pct=True).values
    result["author_popularity"] = np.round(reviews_rank, 2)

    # год издания
    pub_col = None
    for col in ["publication_date", "original_publication_year"]:
        if col in df.columns:
            pub_col = col
            break

    if pub_col:
        result["year_published"] = pd.to_datetime(
            df[pub_col], errors="coerce"
        ).dt.year.fillna(2000).astype(int).clip(1900, 2025).values
    else:
        result["year_published"] = np.random.randint(1970, 2024, len(df))

    result["price"] = np.round(
        np.random.uniform(150, 1500, len(df)), 2
    )

    result["page_count"] = df["num_pages_clean"].astype(int).clip(10, 1500).values

    # число выдач — перцентили ratings_count
    ratings = df["ratings_count"].values.astype(float)
    ratings_rank = pd.Series(ratings).rank(pct=True).values
    result["times_borrowed"] = (ratings_rank * 500).astype(int).clip(0, 500)

    result["rating"] = df["average_rating"].clip(1.0, 5.0).round(1).values

    # is_popular через медианы — примерно 30-40% будут популярными
    median_rating = result["rating"].median()
    median_borrowed = result["times_borrowed"].median()
    median_author = result["author_popularity"].median()

    result["is_popular"] = (
        (result["rating"] >= median_rating) &
        (result["times_borrowed"] >= median_borrowed) &
        (result["author_popularity"] >= median_author * 0.8)
    ).astype(int)

    # немного шума
    noise_idx = np.random.choice(len(result), size=int(len(result) * 0.03), replace=False)
    result.loc[noise_idx, "is_popular"] = 1 - result.loc[noise_idx, "is_popular"]

    result.to_csv(output_path, index=False)
    popular = result["is_popular"].sum()
    print(f"адаптировано {len(result)} строк -> {output_path}")
    print(f"популярных: {popular} ({popular / len(result) * 100:.1f}%)")
    print(f"непопулярных: {len(result) - popular}")
    return result


if __name__ == "__main__":
    adapt_goodreads()
