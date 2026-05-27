import pandas as pd
import numpy as np

def generate_library_data(n_rows=100000, output_path="data/books.csv"):
    np.random.seed(42)

    genres = ["fiction", "science", "history", "biography",
              "children", "fantasy", "detective", "romance",
              "psychology", "technology"]

    genre = np.random.choice(genres, n_rows)

    author_popularity = np.round(np.random.uniform(0.0, 1.0, n_rows), 2)

    year_published = np.random.randint(1950, 2025, n_rows)
    price = np.round(np.random.uniform(100, 2000, n_rows), 2)
    page_count = np.random.randint(50, 1200, n_rows)

    times_borrowed = np.random.randint(0, 500, n_rows)
    rating = np.round(np.random.uniform(1.0, 5.0, n_rows), 1)

    score = (
        (rating > 3.5).astype(int) * 2 +
        (times_borrowed > 100).astype(int) * 2 +
        (author_popularity > 0.4).astype(int) +
        (year_published > 2000).astype(int)
    )
    is_popular = (score >= 3).astype(int)

    noise_idx = np.random.choice(n_rows, size=int(n_rows * 0.05), replace=False)
    is_popular[noise_idx] = 1 - is_popular[noise_idx]

    df = pd.DataFrame({
        "book_id": range(1, n_rows + 1),
        "genre": genre,
        "author_popularity": author_popularity,
        "year_published": year_published,
        "price": price,
        "page_count": page_count,
        "times_borrowed": times_borrowed,
        "rating": rating,
        "is_popular": is_popular
    })

    df.to_csv(output_path, index=False)
    print(f"generated {n_rows} rows -> {output_path}")
    return df

if __name__ == "__main__":
    generate_library_data()
