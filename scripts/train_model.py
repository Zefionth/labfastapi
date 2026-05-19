import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Константы
MODEL_PATH = "./models/bike_purchase_model.pkl"
RESULTS_PATH = "./training_results.txt"

def prepare_features(df):
    """Подготовка признаков для модели классификации"""
    data = df.copy()
    
    y = data['Purchased Bike'].map({'Yes': 1, 'No': 0})
    X = data.drop(['Purchased Bike'], axis=1)
    
    if 'ID' in X.columns:
        X = X.drop(['ID'], axis=1)
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Категориальные признаки: {categorical_cols}")
    print(f"Числовые признаки: {numerical_cols}")
    
    return X, y, categorical_cols, numerical_cols

def create_preprocessor(categorical_cols, numerical_cols):
    """Создание препроцессора"""
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    return preprocessor

def train():
    """Основная функция обучения модели"""
    # Создаём папку для модели
    os.makedirs('./models', exist_ok=True)
    
    # Загружаем очищенные данные
    df = pd.read_csv('./data/df_clear.csv')
    print(f"Загружено данных: {df.shape}")
    
    # Подготавливаем признаки
    X, y, cat_cols, num_cols = prepare_features(df)
    
    # Разделяем на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape}, Val size: {X_val.shape}")
    
    # Создаем препроцессор
    preprocessor = create_preprocessor(cat_cols, num_cols)
    
    # Параметры для GridSearch
    param_grid = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs']
    }
    
    # Пайплайн
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    print("Обучение модели...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\nМетрики на валидации:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Сохраняем модель
    joblib.dump(best_model, MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")
    
    # Сохраняем метрики
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"Best params: {best_params}\n")
    
    print(f"Метрики сохранены в {RESULTS_PATH}")
    return best_model

if __name__ == "__main__":
    train()