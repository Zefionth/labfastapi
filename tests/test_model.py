import pytest
from app.model import BikePurchaseModel

@pytest.fixture
def model():
    """Фикстура для загрузки модели"""
    return BikePurchaseModel()

@pytest.fixture
def sample_input():
    """Пример входных данных"""
    return {
        'marital_status': 'Married',
        'gender': 'Male',
        'education': 'Bachelors',
        'occupation': 'Professional',
        'home_owner': 'Yes',
        'commute_distance': '0-1 Miles',
        'region': 'Europe',
        'income': 60000,
        'children': 2,
        'cars': 1,
        'age': 35
    }

def test_model_loading(model):
    """Тест загрузки модели"""
    assert model.model is not None

def test_model_prediction(model, sample_input):
    """Тест предсказания"""
    prediction, probability = model.predict(sample_input)
    
    assert isinstance(prediction, int)
    assert prediction in [0, 1]
    assert isinstance(probability, float)
    assert 0 <= probability <= 1

def test_model_prediction_with_missing_field(model, sample_input):
    """Тест на отсутствующее поле"""
    del sample_input['income']
    
    with pytest.raises(ValueError):
        model.predict(sample_input)

def test_model_prediction_batch(model):
    """Тест пакетного предсказания"""
    test_data = [
        {
            'marital_status': 'Single',
            'gender': 'Female',
            'education': 'Master',
            'occupation': 'Manager',
            'home_owner': 'No',
            'commute_distance': '5-10 Miles',
            'region': 'North America',
            'income': 45000,
            'children': 0,
            'cars': 1,
            'age': 28
        },
        {
            'marital_status': 'Married',
            'gender': 'Male',
            'education': 'PhD',
            'occupation': 'Engineer',
            'home_owner': 'Yes',
            'commute_distance': '0-1 Miles',
            'region': 'Asia',
            'income': 85000,
            'children': 3,
            'cars': 2,
            'age': 42
        }
    ]
    
    for data in test_data:
        prediction, probability = model.predict(data)
        assert prediction in [0, 1]
        assert 0 <= probability <= 1