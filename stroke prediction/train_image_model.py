"""
Training script for the MRI Stroke Detection Model (CNN).
Trains on the CT scan dataset with stroke detection labels.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from PIL import Image
import json

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Paths
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "datasets" / "dataset"
SUMMARY_CSV = DATASET_DIR / "summary.csv"
MODEL_OUTPUT_PATH = SCRIPT_DIR / "models" / "cnn_model.h5"

# Training parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0001
MAX_SLICES_PER_STUDY = 5  # Limit slices per study to reduce memory


def load_dataset_info():
    """Load dataset metadata and summary."""
    print("Loading dataset information...")
    
    df = pd.read_csv(SUMMARY_CSV)
    print(f"Total studies: {len(df)}")
    print(f"Splits: {df['part'].value_counts().to_dict()}")
    
    # DSA indicates stroke presence
    # Convert to binary labels
    df['label'] = df['dsa'].apply(lambda x: 1 if x == True or x == 'True' else 0)
    
    print(f"\nLabel distribution:")
    print(f"  Stroke (DSA=True): {(df['label'] == 1).sum()}")
    print(f"  No Stroke (DSA=False): {(df['label'] == 0).sum()}")
    
    return df


def load_npz_image(filepath, target_size=IMG_SIZE):
    """Load and preprocess an image from NPZ file."""
    try:
        data = np.load(str(filepath))
        # NPZ files contain arrays with keys
        if 'arr_0' in data:
            img = data['arr_0']
        else:
            # Get first array
            img = data[list(data.keys())[0]]
        
        # Handle different dimensions
        if len(img.shape) == 3:
            # Take middle slice if 3D
            if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
                img = img[img.shape[0] // 2]
            elif img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]:
                img = img[:, :, img.shape[2] // 2]
            else:
                img = img[:, :, 0]
        
        # Convert to float
        img = img.astype(np.float32)
        
        # Normalize to 0-1
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        
        # Resize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB
        img = np.stack([img, img, img], axis=-1)
        
        return img
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_images_from_split(df, split_name):
    """Load all images for a dataset split."""
    print(f"\nLoading {split_name} images...")
    
    split_df = df[df['part'] == split_name]
    split_dir = DATASET_DIR / split_name
    
    images = []
    labels = []
    
    for idx, row in split_df.iterrows():
        study_name = row['name']
        label = row['label']
        study_path = split_dir / study_name
        
        if not study_path.exists():
            continue
        
        # Find all slice directories
        slice_dirs = sorted([d for d in study_path.iterdir() if d.is_dir()])[:MAX_SLICES_PER_STUDY]
        
        for slice_dir in slice_dirs:
            # Look for image.npz
            npz_file = slice_dir / "image.npz"
            if npz_file.exists():
                img = load_npz_image(npz_file)
                if img is not None:
                    images.append(img)
                    labels.append(label)
    
    print(f"Loaded {len(images)} images from {split_name}")
    return np.array(images), np.array(labels)


def build_cnn_model(input_shape=(224, 224, 3)):
    """Build the CNN model for stroke detection."""
    inputs = keras.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 5
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name='stroke_detection_cnn')
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """Train the CNN model."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            str(MODEL_OUTPUT_PATH),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Class weights
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    class_weights = {0: 1.0, 1: n_neg / n_pos if n_pos > 0 else 1.0}
    print(f"\nClass weights: {class_weights}")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    print("\nStarting training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Metrics
    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test AUC:      {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nSensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    return {'accuracy': acc, 'auc': auc, 'sensitivity': sensitivity, 'specificity': specificity}


def main():
    print("="*60)
    print("MRI STROKE DETECTION - CNN MODEL TRAINING")
    print("="*60)
    
    print(f"\nGPU Available: {tf.config.list_physical_devices('GPU')}")
    
    # Load dataset
    df = load_dataset_info()
    
    # Load images
    X_train, y_train = load_images_from_split(df, 'train')
    X_val, y_val = load_images_from_split(df, 'val')
    X_test, y_test = load_images_from_split(df, 'test')
    
    if len(X_train) == 0:
        print("\n⚠️ No training images found!")
        return
    
    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape}, positive ratio: {y_train.mean():.3f}")
    print(f"  Val:   {X_val.shape}, positive ratio: {y_val.mean():.3f}")
    print(f"  Test:  {X_test.shape}, positive ratio: {y_test.mean():.3f}")
    
    # Build model
    model = build_cnn_model()
    
    # Train
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n✅ Model saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
