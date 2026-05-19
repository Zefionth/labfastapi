from sqlalchemy.orm import Session
from app.database import PredictionHistory
import json

def save_prediction(db: Session, input_data: dict, prediction: int, probability: float):
    """Сохранение предсказания в БД"""
    db_record = PredictionHistory(
        input_data=json.dumps(input_data, ensure_ascii=False),
        prediction=prediction,
        probability=probability
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record

def get_prediction_history(db: Session, skip: int = 0, limit: int = 100):
    """Получение истории предсказаний"""
    return db.query(PredictionHistory).order_by(
        PredictionHistory.created_at.desc()
    ).offset(skip).limit(limit).all()

def get_stats(db: Session):
    """Получение статистики по предсказаниям"""
    total = db.query(PredictionHistory).count()
    bought = db.query(PredictionHistory).filter(PredictionHistory.prediction == 1).count()
    
    return {
        "total_predictions": total,
        "bought_count": bought,
        "not_bought_count": total - bought,
        "buy_rate": bought / total if total > 0 else 0
    }