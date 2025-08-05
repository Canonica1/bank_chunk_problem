## 🏦 Bank Churn Prediction

Набор файлов и ноутбуков, реализующий решение задачи Bank Churn Prediction, на котором я вошел в топ 5.
Notebook на kaggle https://www.kaggle.com/code/canonica1/top-3-private-score-0-93267

## 📁 Структура репозитория

```

BANK\_PROJECT/
│
├── notebooks/               # Jupyter ноутбуки с пайплайнами
├── src/                     # Исходный код (модули)
├── data/                    # Скрипты для загрузки/предобработки
├── models/                  # Сохранённые обученные модели (.pkl)
├── predicts/                # Генерация submission.csv и результаты
├── requirements.txt         # Список зависимостей
├── README.md                # Этот файл
├── .gitignore
└── LICENSE

````
Также был дообучен bert чтобы сделать фичу странны более крутой
вот ссылка на файл https://github.com/Canonica1/fine_tune_distilbert_surnames

---

## 🎯 Описание решения

Это решение включает следующие ключевые идеи:

- **Target encoding** по фамилии (`Surname`) для повышения информативности.
- **Признак страны (predicted country)**, полученный на основе нейронной модели, обученной на отдельном датасете Kaggle — по фамилии определяет страну.

---

## 🚀 Как воспроизвести

### 1. Установка окружения:

```bash
pip install -r requirements.txt
````
### 2. Обучение модели и Генерация предсказаний:

```bash
python __init__.py
```

---

## 📊 Результаты

| Public AUC | Private AUC  |
| -----------| ------------ |
| \~0.93929  | \~0.93267     |



