"""
Training script for the MRI Stroke Detection Model using transfer learning.
This script creates synthetic training data for demonstration and can be 
easily adapted to use real MRI data once downloaded.

For real data usage:
1. Download the OpenNeuro dataset using download_dataset.py
2. Update the DATA_DIR path to point to the downloaded data
3. Change USE_SYNTHETIC_DATA to False
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras.applications import ResNet50V2
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import json
from datetime import datetime

# Configuration
SCRIPT_DIR = Path(__file__).parent
MODEL_OUTPUT_DIR = SCRIPT_DIR / "models"
MODEL_OUTPUT_PATH = MODEL_OUTPUT_DIR / "mri_stroke_model.h5"

# Training parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0001
NUM_SYNTHETIC_SAMPLES = 500
USE_SYNTHETIC_DATA = True  # Set to False when real data is available


def generate_synthetic_mri_data(num_samples=100):
    """Generate synthetic MRI-like images for training demonstration.
    
    Creates two classes:
    - Class 0 (No Stroke): Normal brain patterns
    - Class 1 (Stroke): Brain patterns with bright lesion-like regions
    """
    print(f"Generating {num_samples} synthetic MRI images...")
    
    images = []
    labels = []
    
    for i in range(num_samples):
        # Create base brain-like image
        img = np.zeros((224, 224, 3), dtype=np.float32)
        
        # Add brain-like elliptical structure
        y, x = np.ogrid[:224, :224]
        center_x, center_y = 112, 112
        
        # Create brain outline
        brain_mask = ((x - center_x)**2 / 80**2 + (y - center_y)**2 / 90**2) < 1
        img[brain_mask] = np.random.uniform(0.3, 0.5, (brain_mask.sum(), 3))
        
        # Add brain texture
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        # Determine class
        label = np.random.randint(0, 2)
        
        if label == 1:  # Stroke - add bright lesion
            # Random lesion position (usually on one side)
            side = np.random.choice([-1, 1])
            lesion_x = center_x + side * np.random.randint(20, 50)
            lesion_y = center_y + np.random.randint(-30, 30)
            lesion_radius = np.random.randint(15, 35)
            
            # Create lesion mask
            lesion_mask = ((x - lesion_x)**2 + (y - lesion_y)**2) < lesion_radius**2
            lesion_mask = lesion_mask & brain_mask
            
            # Make lesion bright (hyperintense)
            img[lesion_mask] = np.random.uniform(0.7, 1.0, (lesion_mask.sum(), 3))
        
        images.append(img)
        labels.append(label)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} images")
    
    return np.array(images), np.array(labels)


def build_model(input_shape=(224, 224, 3)):
    """Build a transfer learning model using ResNet50V2.
    
    Uses ResNet50V2 pretrained on ImageNet as a feature extractor,
    with custom classification head for stroke detection.
    """
    print("Building model with ResNet50V2 backbone...")
    
    # Load pre-trained ResNet50V2 without top layers
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build custom classification head
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing for ResNet
    x = keras.applications.resnet_v2.preprocess_input(inputs)
    
    # Feature extraction
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model, base_model


def create_data_augmentation():
    """Create data augmentation pipeline for MRI images."""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )


def train_model(model, X_train, y_train, X_val, y_val, base_model=None):
    """Train the model with early stopping and learning rate reduction."""
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_OUTPUT_PATH),
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Data augmentation
    datagen = create_data_augmentation()
    
    # Phase 1: Train with frozen base model
    print("\n=== Phase 1: Training classification head (frozen backbone) ===")
    history1 = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=min(10, EPOCHS),
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        steps_per_epoch=len(X_train) // BATCH_SIZE
    )
    
    # Phase 2: Fine-tune last few layers of base model
    if base_model is not None and EPOCHS > 10:
        print("\n=== Phase 2: Fine-tuning last layers of backbone ===")
        
        # Unfreeze last 20 layers
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        history2 = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            epochs=EPOCHS - 10,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            steps_per_epoch=len(X_train) // BATCH_SIZE
        )
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data."""
    print("\n=== Evaluating Model ===")
    
    results = model.evaluate(X_test, y_test, verbose=1)
    
    print(f"\nTest Results:")
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Accuracy: {results[1]:.4f}")
    print(f"  AUC: {results[2]:.4f}")
    
    # Predictions for confusion matrix
    predictions = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Calculate metrics
    tp = np.sum((predictions == 1) & (y_test == 1))
    tn = np.sum((predictions == 0) & (y_test == 0))
    fp = np.sum((predictions == 1) & (y_test == 0))
    fn = np.sum((predictions == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {tp}")
    print(f"  True Negatives: {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': float(results[1]),
        'auc': float(results[2]),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("MRI Stroke Detection Model Training")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate or load data
    if USE_SYNTHETIC_DATA:
        print("\n>>> Using SYNTHETIC DATA for training <<<")
        print("To use real MRI data, set USE_SYNTHETIC_DATA = False")
        print("and ensure the OpenNeuro dataset is downloaded.\n")
        
        X, y = generate_synthetic_mri_data(NUM_SYNTHETIC_SAMPLES)
    else:
        # TODO: Implement loading from OpenNeuro dataset
        raise NotImplementedError(
            "Real data loading not yet implemented. "
            "Download the dataset first using download_dataset.py"
        )
    
    # Split data
    indices = np.random.permutation(len(X))
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Class distribution (train): {np.sum(y_train == 0)} normal, {np.sum(y_train == 1)} stroke")
    
    # Build model
    model, base_model = build_model()
    model.summary()
    
    # Train model
    model = train_model(model, X_train, y_train, X_val, y_val, base_model)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save final model and metrics
    model.save(MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")
    
    # Save training info
    info = {
        'training_date': datetime.now().isoformat(),
        'samples': len(X),
        'synthetic_data': USE_SYNTHETIC_DATA,
        'epochs': EPOCHS,
        'metrics': metrics
    }
    
    info_path = MODEL_OUTPUT_DIR / 'training_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Training info saved to: {info_path}")
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
