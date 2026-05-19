from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import logging

from app.schemas import CustomerFeatures, PredictionResponse
from app.database import get_db, init_db
from app.model import get_model
from app import crud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()
    logger.info("Loading model...")
    try:
        get_model()
    except Exception as e:
        logger.error(f"Model load failed at startup, continuing without model: {e}")
    logger.info("Application started successfully")
    yield

# Создаем приложение
app = FastAPI(
    title="Bike Purchase Prediction API",
    description="API для предсказания покупки велосипеда клиентом",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Bike Purchase Prediction API",
        "status": "running",
        "endpoints": ["/predict", "/health", "/history", "/stats"]
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    try:
        # Не вызываем get_model() если модель не загружена корректно при старте
        from app.model import model_instance
        model_loaded = model_instance is not None and getattr(model_instance, 'model', None) is not None
        return {
            "status": "healthy",
            "model_loaded": model_loaded,
            "timestamp": datetime.now(timezone.utc)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    features: CustomerFeatures,
    db: Session = Depends(get_db)
):
    """
    Предсказание покупки велосипеда
    
    - **marital_status**: Семейное положение (Married/Single)
    - **gender**: Пол (Male/Female)
    - **education**: Образование
    - **occupation**: Профессия
    - **home_owner**: Владелец дома (Yes/No)
    - **commute_distance**: Расстояние до работы
    - **region**: Регион
    - **income**: Доход
    - **children**: Количество детей
    - **cars**: Количество машин
    - **age**: Возраст
    """
    try:
        # Получаем модель
        model = get_model()
        
        # Делаем предсказание
        prediction, probability = model.predict(features.model_dump())
        
        # Сохраняем в БД
        crud.save_prediction(db, features.model_dump(), prediction, probability)
        
        # Формируем ответ
        message = "Customer is likely to purchase a bike" if prediction == 1 else "Customer is unlikely to purchase a bike"
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            message=message,
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/history")
async def get_history(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Получение истории предсказаний"""
    history = crud.get_prediction_history(db, skip, limit)
    return {
        "total": len(history),
        "predictions": [
            {
                "id": h.id,
                "prediction": h.prediction,
                "probability": h.probability,
                "created_at": h.created_at
            }
            for h in history
        ]
    }

@app.get("/stats")
async def get_statistics(db: Session = Depends(get_db)):
    """Получение статистики по предсказаниям"""
    stats = crud.get_stats(db)
    return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)