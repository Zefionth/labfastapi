FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости для PostgreSQL
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем приложение и скрипты для обучения
COPY ./app ./app
COPY ./scripts ./scripts
COPY ./data ./data

# Обучаем модель внутри образа (создаст ./models/bike_purchase_model.pkl)
RUN python scripts/train_model.py || true

# Устанавливаем PYTHONPATH
ENV PYTHONPATH=/app

# Открываем порт
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]