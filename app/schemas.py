from datetime import datetime
from pydantic import BaseModel, ConfigDict

class CustomerFeatures(BaseModel):
    """Входные данные для предсказания"""
    marital_status: str
    gender: str
    education: str
    occupation: str
    home_owner: str
    commute_distance: str
    region: str
    income: float
    children: int
    cars: int
    age: int
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "marital_status": "Married",
                "gender": "Male",
                "education": "Bachelors",
                "occupation": "Professional",
                "home_owner": "Yes",
                "commute_distance": "0-1 Miles",
                "region": "Europe",
                "income": 60000,
                "children": 2,
                "cars": 1,
                "age": 35
            }
        }
    )

class PredictionResponse(BaseModel):
    """Ответ модели"""
    prediction: int
    probability: float
    message: str
    timestamp: datetime

class PredictionLog(BaseModel):
    """Лог предсказания в БД"""
    id: int
    input_data: str
    prediction: int
    probability: float
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)