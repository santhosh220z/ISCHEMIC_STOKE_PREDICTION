
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Configuration matching utils/preprocessing.py
NUMERICAL_FEATURES = ['age', 'avg_glucose_level', 'bmi']
CATEGORICAL_FEATURES = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
BINARY_FEATURES = ['hypertension', 'heart_disease']

# Mappings from utils/preprocessing.py
GENDER_MAP = {'Male': 0, 'Female': 1, 'Other': 2}
EVER_MARRIED_MAP = {'No': 0, 'Yes': 1}
WORK_TYPE_MAP = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
RESIDENCE_MAP = {'Urban': 0, 'Rural': 1}
SMOKING_MAP = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}

NORMALIZATION_PARAMS = {
    'age': {'min': 0, 'max': 100},
    'avg_glucose_level': {'min': 50, 'max': 300},
    'bmi': {'min': 10, 'max': 60}
}

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Drop ID
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        
    # Handle Missing BMI
    imputer = SimpleImputer(strategy='mean')
    df['bmi'] = imputer.fit_transform(df[['bmi']])
    
    # Feature Engineering / Preprocessing
    X = pd.DataFrame()
    
    # 1. Numerical Features (Normalized)
    for feature in NUMERICAL_FEATURES:
        if feature in df.columns:
            # Clip and Normalize
            min_val = NORMALIZATION_PARAMS[feature]['min']
            max_val = NORMALIZATION_PARAMS[feature]['max']
            X[feature] = (df[feature].clip(min_val, max_val) - min_val) / (max_val - min_val)
            X[feature] = X[feature].clip(0, 1)
            
    # 2. Categorical Features (Label Encoded)
    for feature in CATEGORICAL_FEATURES:
        if feature == 'gender':
            X[feature] = df[feature].map(GENDER_MAP).fillna(2).astype(int)
        elif feature == 'ever_married':
            X[feature] = df[feature].map(EVER_MARRIED_MAP).fillna(0).astype(int)
        elif feature == 'work_type':
            X[feature] = df[feature].map(WORK_TYPE_MAP).fillna(0).astype(int)
        elif feature == 'residence_type': # Note: CSV col is Residence_type usually, checking
             # The CSV column might be 'Residence_type' capitalized
             col_name = 'Residence_type' if 'Residence_type' in df.columns else 'residence_type'
             X['residence_type'] = df[col_name].map(RESIDENCE_MAP).fillna(0).astype(int)
        elif feature == 'smoking_status':
            X[feature] = df[feature].map(SMOKING_MAP).fillna(3).astype(int)
            
    # 3. Binary Features
    for feature in BINARY_FEATURES:
        X[feature] = df[feature]
        
    y = df['stroke']
    
    return X, y

def train_and_evaluate(X, y):
    print("\nSplitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Applying SMOTE to handle class imbalance in training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Training shapes: Original {X_train.shape}, Resampled {X_train_resampled.shape}")
    
    print("\nTraining Random Forest Model...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train_resampled, y_train_resampled)
    
    print("\nEvaluating on Test Set...")
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

if __name__ == "__main__":
    dataset_path = "datasets/healthcare-dataset-stroke-data.csv"
    try:
        X, y = load_and_preprocess_data(dataset_path)
        train_and_evaluate(X, y)
    except Exception as e:
        print(f"Error: {e}")
