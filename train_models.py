"""
KrishiMitra ML Pipeline: Weather-to-Pest Outbreak Risk Predictor
Trains an XGBoost classifier using microclimate variables (temperature, humidity, rainfall).
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

def generate_synthetic_agro_dataset(n_samples=1500, random_state=42):
    np.random.seed(random_state)
    
    # Generate realistic microclimate distributions (Indian agro-climatic seasons)
    temp_max = np.random.uniform(18.0, 42.0, n_samples)
    temp_min = temp_max - np.random.uniform(5.0, 15.0, n_samples)
    humidity_morning = np.random.uniform(40.0, 98.0, n_samples)
    humidity_evening = humidity_morning - np.random.uniform(10.0, 30.0, n_samples)
    humidity_evening = np.clip(humidity_evening, 20.0, 95.0)
    
    rainfall_mm = np.random.exponential(scale=12.0, size=n_samples)
    rainfall_mm[np.random.rand(n_samples) > 0.4] = 0.0  # 60% dry days
    
    consecutive_wet_days = np.random.geometric(p=0.4, size=n_samples) - 1
    consecutive_wet_days = np.clip(consecutive_wet_days, 0, 7)
    
    wind_speed_kmh = np.random.uniform(3.0, 28.0, n_samples)
    crop_stage = np.random.choice([0, 1, 2, 3], size=n_samples) # 0:Seedling, 1:Veg, 2:Flowering, 3:Fruiting
    
    # Agronomic ground truth risk formula based on ICAR thresholds:
    # High humidity (>80%) + warm temp (22-30C) + consecutive wet days = highest fungal/pest risk
    risk_score = (
        0.35 * (humidity_morning / 100.0) +
        0.20 * (humidity_evening / 100.0) +
        0.25 * (consecutive_wet_days / 7.0) +
        0.15 * (1.0 - np.abs(temp_max - 28.0) / 20.0) +
        0.05 * (rainfall_mm > 5.0).astype(float)
    )
    # Add random biological variation noise
    risk_score += np.random.normal(0, 0.05, n_samples)
    
    # Label: 0: Low Risk (< 0.45), 1: Moderate Risk (0.45 - 0.65), 2: High/Severe Risk (> 0.65)
    labels = np.zeros(n_samples, dtype=int)
    labels[(risk_score >= 0.45) & (risk_score < 0.68)] = 1
    labels[risk_score >= 0.68] = 2

    df = pd.DataFrame({
        'temp_max': np.round(temp_max, 1),
        'temp_min': np.round(temp_min, 1),
        'humidity_morning': np.round(humidity_morning, 1),
        'humidity_evening': np.round(humidity_evening, 1),
        'rainfall_mm': np.round(rainfall_mm, 1),
        'consecutive_wet_days': consecutive_wet_days,
        'wind_speed_kmh': np.round(wind_speed_kmh, 1),
        'crop_stage': crop_stage,
        'pest_risk_level': labels
    })
    return df

def train_and_export():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    dataset_path = os.path.join(data_dir, 'weather_pest_dataset.csv')
    model_output_path = os.path.join(models_dir, 'weather_model.pkl')
    
    print("🌾 Generating synthetic agro-climatic dataset...")
    df = generate_synthetic_agro_dataset()
    df.to_csv(dataset_path, index=False)
    print(f"✅ Saved dataset to: {dataset_path} ({len(df)} records)")
    
    features = ['temp_max', 'temp_min', 'humidity_morning', 'humidity_evening', 
                'rainfall_mm', 'consecutive_wet_days', 'wind_speed_kmh', 'crop_stage']
    X = df[features]
    y = df['pest_risk_level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🤖 Training XGBoost Outbreak Predictor...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy on Test Set: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'Moderate Risk', 'High Risk']))
    
    # Save model and metadata
    package = {
        'model': model,
        'feature_names': features,
        'class_names': ['Low Risk', 'Moderate Risk', 'High Risk'],
        'accuracy': float(acc),
        'feature_importances': dict(zip(features, model.feature_importances_.tolist()))
    }
    with open(model_output_path, 'wb') as f:
        pickle.dump(package, f)
    print(f"📦 Successfully exported trained model to: {model_output_path}")

if __name__ == '__main__':
    train_and_export()
