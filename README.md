# library big data — ml система предсказания популярности книг

лабораторная работа по курсу big data
итмо, 2026

## описание

веб-приложение для библиотек которое:
- загружает большие датасеты о книгах (csv, до 500 000+ строк)
- обучает модель random forest для предсказания популярности книги
- визуализирует аналитику фонда через 4 графика
- рекомендует количество экземпляров новой книги для закупки
- ведёт журнал всех событий в реальном времени

связь с тз библиотечной системы:
- статистика популярности книг (раздел 3.1.6)
- количество новых поступлений (раздел 3.1.6)
- формирование отчётов (раздел 3.1.6)

---

## быстрый старт

### 1. клонировать репозиторий

    git clone https://github.com/Sophie505-bit/Lab_3_Big_Data_Library.git
    cd Lab_3_Big_Data_Library

### 2. создать виртуальное окружение

    python -m venv venv

    windows (cmd):
    venv\Scripts\activate.bat

    windows (powershell):
    venv\Scripts\Activate.ps1

    macos / linux:
    source venv/bin/activate

### 3. установить зависимости

    pip install -r requirements.txt

### 4. запустить приложение

    python app.py

открыть в браузере: http://127.0.0.1:5000

---

## использование

после запуска на главной странице доступны три способа получить данные:

1. сгенерировать синтетический датасет (10 000 — 500 000 строк)
2. использовать реальный датасет kaggle goodreads (11 000 книг)
3. загрузить собственный csv файл

затем нажать кнопку запустить обучение модели.
результаты появятся автоматически после завершения обучения.

---

## kaggle датасет (опционально)

скачать:
    pip install kaggle
    kaggle datasets download -d jealousleopard/goodreadsbooks

положить goodreadsbooks.zip в корень проекта.
в интерфейсе появится кнопка использовать goodreads.

---

## структура проекта

    Lab_3_Big_Data_Library/
    ├── app.py                    главное flask-приложение
    ├── requirements.txt          зависимости python
    ├── README.md
    ├── data/
    │   ├── generate_data.py      генератор синтетического датасета
    │   └── adapt_goodreads.py    адаптер kaggle датасета
    ├── ml/
    │   └── model.py              обучение random forest + графики
    ├── static/
    │   ├── css/style.css
    │   └── plots/                сгенерированные графики
    └── templates/
        ├── index.html            главная страница
        ├── results.html          метрики и графики
        ├── predict.html          предсказание для новой книги
        ├── training.html         страница ожидания обучения
        └── logs.html             журнал событий

---

## формат csv

| столбец            | описание                        | тип    |
|--------------------|---------------------------------|--------|
| book_id            | идентификатор книги             | int    |
| genre              | жанр                            | string |
| author_popularity  | популярность автора (0.0-1.0)   | float  |
| year_published     | год издания                     | int    |
| price              | цена в рублях                   | float  |
| page_count         | количество страниц              | int    |
| times_borrowed     | число выдач                     | int    |
| rating             | рейтинг (1.0-5.0)              | float  |
| is_popular         | метка: популярная (1) / нет (0) | int    |

---

## ml модель

алгоритм: random forest classifier
признаки: genre, author_popularity, year_published, price, page_count, times_borrowed, rating
целевая переменная: is_popular
разбивка: 80% обучение / 20% тест
метрики: accuracy, precision, recall, f1, roc-auc

результаты на синтетических данных 500 000 строк:
- accuracy:  95.8%
- precision: 95.8%
- recall:    95.9%
- f1-score:  95.9%
- roc-auc:   0.9585

результаты на kaggle goodreads 11 000 строк:
- accuracy:  97.7%
- precision: 98.5%
- recall:    93.2%
- f1-score:  95.7%
- roc-auc:   0.9584

---

## страницы приложения

| страница      | адрес     | описание                          |
|---------------|-----------|-----------------------------------|
| главная       | /         | выбор датасета и запуск обучения  |
| результаты    | /results  | метрики модели и 4 графика        |
| предсказание  | /predict  | предсказание для новой книги      |
| логи          | /logs     | журнал событий в реальном времени |

---

## стек технологий

- python 3.10+
- flask 3.0
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

