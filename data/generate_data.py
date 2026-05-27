import pandas as pd
import numpy as np


def generate_library_data(n_rows=100000, output_path="data/books.csv"):
    np.random.seed(42)

    genres = [
        "fiction", "science", "history", "biography",
        "children", "fantasy", "detective", "romance",
        "psychology", "technology"
    ]

    # базовая популярность жанра — реалистичные веса
    # детектив, фэнтези, художественная — популярнее
    # наука, история — менее популярны у широкой аудитории
    genre_popularity_base = {
        "fiction":    0.65,
        "fantasy":    0.70,
        "detective":  0.68,
        "romance":    0.62,
        "children":   0.60,
        "psychology": 0.55,
        "biography":  0.50,
        "technology": 0.45,
        "history":    0.42,
        "science":    0.38,
    }

    # жанр влияет на среднее число выдач
    genre_borrow_boost = {
        "fiction":    80,
        "fantasy":    90,
        "detective":  85,
        "romance":    75,
        "children":   70,
        "psychology": 55,
        "biography":  50,
        "technology": 40,
        "history":    35,
        "science":    30,
    }

    genre = np.random.choice(genres, n_rows)

    genre_pop_arr    = np.array([genre_popularity_base[g] for g in genre])
    genre_borrow_arr = np.array([genre_borrow_boost[g] for g in genre])

    # популярность автора
    author_popularity = np.round(np.random.uniform(0.0, 1.0, n_rows), 2)

    # год издания коррелирует с популярностью автора
    year_base = np.random.randint(1950, 2025, n_rows)
    year_published = np.clip(
        year_base + (author_popularity * 10).astype(int),
        1950, 2025
    ).astype(int)

    # цена коррелирует с годом и популярностью автора
    price_base = np.random.uniform(100, 1200, n_rows)
    price = np.round(
        price_base
        + (year_published - 1950) * 2.5
        + author_popularity * 250,
        2
    )
    price = np.clip(price, 100, 2500)

    # число страниц — слабо зависит от жанра
    page_base = np.random.randint(50, 900, n_rows)
    genre_pages = {
        "fiction":    50, "fantasy": 100, "detective": 30,
        "romance":    20, "children": -100, "psychology": 40,
        "biography":  60, "technology": 50, "history": 80, "science": 70,
    }
    page_boost = np.array([genre_pages[g] for g in genre])
    page_count = np.clip(page_base + page_boost, 30, 1200).astype(int)

    # рейтинг зависит от популярности автора и жанра
    rating_base = np.random.uniform(1.0, 4.5, n_rows)
    rating = np.round(
        np.clip(
            rating_base
            + author_popularity * 0.8
            + genre_pop_arr * 0.4,
            1.0, 5.0
        ),
        1
    )

    # число выдач зависит от рейтинга, популярности автора и жанра
    times_base = np.random.randint(0, 200, n_rows)
    times_borrowed = np.clip(
        times_base
        + (rating * 25).astype(int)
        + (author_popularity * 70).astype(int)
        + genre_borrow_arr,
        0, 500
    ).astype(int)

    # целевая переменная
    # жанр вносит прямой вклад через genre_pop_arr
    score = (
        (rating > 3.8).astype(int) * 2 +
        (times_borrowed > 180).astype(int) * 2 +
        (author_popularity > 0.5).astype(int) * 2 +
        (year_published > 2000).astype(int) +
        (genre_pop_arr > 0.55).astype(int)
    )
    is_popular = (score >= 5).astype(int)

    # шум
    noise_idx = np.random.choice(n_rows, size=int(n_rows * 0.04), replace=False)
    is_popular[noise_idx] = 1 - is_popular[noise_idx]

    df = pd.DataFrame({
        "book_id":           range(1, n_rows + 1),
        "genre":             genre,
        "author_popularity": author_popularity,
        "year_published":    year_published,
        "price":             price,
        "page_count":        page_count,
        "times_borrowed":    times_borrowed,
        "rating":            rating,
        "is_popular":        is_popular
    })

    df.to_csv(output_path, index=False)
    print(f"generated {n_rows} rows -> {output_path}")
    return df


if __name__ == "__main__":
    generate_library_data()
