import pytest
from fastapi.testclient import TestClient
from app.main import app

# Создаем тестовую БД
TEST_DATABASE_URL = "sqlite:///./test.db"

# Фикстура для тестового клиента
@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

# Тестовые данные
@pytest.fixture
def valid_customer():
    return {
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

def test_root_endpoint(client):
    """Тест корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["status"] == "running"

def test_health_check(client):
    """Тест health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid(client, valid_customer):
    """Тест предсказания с валидными данными"""
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "message" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1

def test_predict_invalid_data(client):
    """Тест предсказания с невалидными данными"""
    invalid_data = {
        "marital_status": "Married",
        "gender": "Male"
        # отсутствуют обязательные поля
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_predict_different_customers(client):
    """Тест предсказания для разных клиентов"""
    test_cases = [
        {  # Молодой, невысокий доход
            "marital_status": "Single",
            "gender": "Female",
            "education": "High School",
            "occupation": "Clerical",
            "home_owner": "No",
            "commute_distance": "More than 10 Miles",
            "region": "Europe",
            "income": 25000,
            "children": 0,
            "cars": 0,
            "age": 22
        },
        {  # Семейный, высокий доход
            "marital_status": "Married",
            "gender": "Male",
            "education": "Graduate Degree",
            "occupation": "Management",
            "home_owner": "Yes",
            "commute_distance": "0-1 Miles",
            "region": "North America",
            "income": 95000,
            "children": 2,
            "cars": 2,
            "age": 40
        }
    ]
    
    for customer in test_cases:
        response = client.post("/predict", json=customer)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data

def test_history_endpoint(client):
    """Тест получения истории"""
    response = client.get("/history")
    assert response.status_code == 200
    assert "predictions" in response.json()

def test_stats_endpoint(client):
    """Тест получения статистики"""
    response = client.get("/stats")
    assert response.status_code == 200
    assert "total_predictions" in response.json()