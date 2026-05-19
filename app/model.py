import joblib
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class BikePurchaseModel:
    """Класс-обёртка для модели предсказания покупки велосипеда"""
    
    def __init__(self, model_path: str = "./models/bike_purchase_model.pkl"):
        self.model = None
        self.model_path = model_path
        self._load_model()
        
        # Категориальные признаки
        self.categorical_cols = ['Marital Status', 'Gender', 'Education', 
                                  'Occupation', 'Home Owner', 'Commute Distance', 'Region']
        self.numerical_cols = ['Income', 'Children', 'Cars', 'Age']
        
    def _load_model(self):
        """Загрузка модели из файла"""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _prepare_input(self, features: dict) -> pd.DataFrame:
        """Подготовка входных данных для модели"""
        # Создаем DataFrame
        input_data = pd.DataFrame([features])
        
        # Переименовываем колонки в соответствии с обученной моделью
        column_mapping = {
            'marital_status': 'Marital Status',
            'gender': 'Gender',
            'education': 'Education',
            'occupation': 'Occupation',
            'home_owner': 'Home Owner',
            'commute_distance': 'Commute Distance',
            'region': 'Region',
            'income': 'Income',
            'children': 'Children',
            'cars': 'Cars',
            'age': 'Age'
        }
        
        input_data = input_data.rename(columns=column_mapping)
        
        # Убеждаемся, что все нужные колонки есть
        expected_cols = self.categorical_cols + self.numerical_cols
        for col in expected_cols:
            if col not in input_data.columns:
                raise ValueError(f"Missing column: {col}")
        
        return input_data[expected_cols]
    
    def predict(self, features: dict) -> Tuple[int, float]:
        """Предсказание"""
        try:
            X = self._prepare_input(features)
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][1]
            
            return int(prediction), float(probability)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

# Глобальный экземпляр модели
model_instance = None

def get_model():
    """Получение экземпляра модели (синглтон)"""
    global model_instance
    if model_instance is None:
        model_instance = BikePurchaseModel()
    return model_instance