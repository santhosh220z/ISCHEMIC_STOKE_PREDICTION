"""
Preprocessing utilities for clinical data and MRI images.
"""

import numpy as np
import pandas as pd
from PIL import Image
import cv2


# Feature configuration - matches trained model from healthcare-dataset-stroke-data.csv
# The CSV has these 10 features: gender, age, hypertension, heart_disease, ever_married, 
# work_type, Residence_type, avg_glucose_level, bmi, smoking_status
NUMERICAL_FEATURES = ['age', 'avg_glucose_level', 'bmi']
CATEGORICAL_FEATURES = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
BINARY_FEATURES = ['hypertension', 'heart_disease']

# Encoding mappings
GENDER_MAP = {'Male': 0, 'Female': 1, 'Other': 2}
EVER_MARRIED_MAP = {'No': 0, 'Yes': 1}
WORK_TYPE_MAP = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
RESIDENCE_MAP = {'Urban': 0, 'Rural': 1}
SMOKING_MAP = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}

# Normalization ranges (typical clinical ranges)
NORMALIZATION_PARAMS = {
    'age': {'min': 0, 'max': 100},
    'avg_glucose_level': {'min': 50, 'max': 300},
    'bmi': {'min': 10, 'max': 60},
    'nihss_score': {'min': 0, 'max': 42},
    'systolic_bp': {'min': 70, 'max': 220},
    'diastolic_bp': {'min': 40, 'max': 140}
}


def normalize_value(value, feature_name):
    """Normalize a numerical value to 0-1 range."""
    params = NORMALIZATION_PARAMS.get(feature_name, {'min': 0, 'max': 1})
    min_val, max_val = params['min'], params['max']
    normalized = (value - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)


def encode_categorical(value, feature_name):
    """Encode categorical features to numerical values."""
    mappings = {
        'gender': GENDER_MAP,
        'ever_married': EVER_MARRIED_MAP,
        'work_type': WORK_TYPE_MAP,
        'residence_type': RESIDENCE_MAP,
        'smoking_status': SMOKING_MAP
    }
    
    mapping = mappings.get(feature_name, {})
    return mapping.get(value, 0)


def preprocess_clinical_data(data):
    """
    Preprocess clinical data for Random Forest model.
    
    Args:
        data: Dictionary containing patient clinical data
        
    Returns:
        numpy array of preprocessed features
    """
    features = []
    
    # Process numerical features
    for feature in NUMERICAL_FEATURES:
        value = data.get(feature, 0)
        normalized = normalize_value(float(value), feature)
        features.append(normalized)
    
    # Process categorical features
    for feature in CATEGORICAL_FEATURES:
        value = data.get(feature, '')
        encoded = encode_categorical(value, feature)
        features.append(encoded)
    
    # Process binary features
    for feature in BINARY_FEATURES:
        value = data.get(feature, 0)
        features.append(int(value))
    
    return np.array(features).reshape(1, -1)


def get_feature_names():
    """Get ordered list of feature names."""
    return NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def preprocess_mri_image(image, target_size=(224, 224)):
    """
    Preprocess MRI image for CNN model.
    
    Args:
        image: PIL Image or numpy array
        target_size: Tuple of (height, width)
        
    Returns:
        Preprocessed numpy array ready for model input
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize to target size
    img_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize pixel values to 0-1
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch


def preprocess_mri_for_segmentation(image, target_size=(224, 224)):
    """
    Preprocess MRI image for U-Net segmentation model.
    
    Args:
        image: PIL Image or numpy array
        target_size: Tuple of (height, width)
        
    Returns:
        Preprocessed numpy array ready for segmentation model
    """
    # Same preprocessing as CNN
    return preprocess_mri_image(image, target_size)


def postprocess_segmentation_mask(mask, original_size):
    """
    Postprocess segmentation mask to original image size.
    
    Args:
        mask: Predicted segmentation mask
        original_size: Tuple of (width, height)
        
    Returns:
        Resized binary mask
    """
    # Remove batch dimension if present
    if len(mask.shape) == 4:
        mask = mask[0]
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    # Threshold to binary mask
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Resize to original size
    resized_mask = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    return resized_mask


def create_lesion_overlay(original_image, mask, color=(255, 0, 0), alpha=0.5):
    """
    Create an overlay of lesion mask on original MRI image.
    
    Args:
        original_image: Original MRI image (numpy array)
        mask: Binary lesion mask
        color: RGB color for overlay
        alpha: Transparency of overlay
        
    Returns:
        Image with lesion overlay
    """
    # Ensure original image is RGB (3 channels)
    if len(original_image.shape) == 2:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 4:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
    elif original_image.shape[2] == 1:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original_image.copy()
    
    # Ensure uint8 format
    if original_rgb.dtype != np.uint8:
        if original_rgb.max() <= 1.0:
            original_rgb = (original_rgb * 255).astype(np.uint8)
        else:
            original_rgb = original_rgb.astype(np.uint8)
    
    # Resize mask to match image size
    if mask.shape[:2] != original_rgb.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (original_rgb.shape[1], original_rgb.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    # Create colored overlay with proper shape
    overlay = original_rgb.copy()
    mask_bool = mask > 0
    
    # Apply color only to RGB channels
    for i, c in enumerate(color[:3]):
        overlay[:, :, i][mask_bool] = c
    
    # Blend with original
    result = cv2.addWeighted(original_rgb, 1 - alpha, overlay, alpha, 0)
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  STROKE PREDICTION - PREPROCESSING DEMO")
    print("=" * 60)

    # --- Clinical Data Preprocessing ---
    sample_patient = {
        'age': 67,
        'avg_glucose_level': 228.69,
        'bmi': 36.6,
        'gender': 'Male',
        'ever_married': 'Yes',
        'work_type': 'Private',
        'residence_type': 'Urban',
        'smoking_status': 'formerly smoked',
        'hypertension': 1,
        'heart_disease': 1
    }

    print("\n--- Clinical Data Preprocessing ---")
    print(f"Input patient data: {sample_patient}")

    features = preprocess_clinical_data(sample_patient)
    feature_names = get_feature_names()

    print(f"\nFeature order: {feature_names}")
    print(f"Preprocessed features: {features}")
    print(f"Feature shape: {features.shape}")

    print("\nBreakdown:")
    for name, val in zip(feature_names, features[0]):
        print(f"  {name:25s} -> {val:.4f}")

    # --- MRI Image Preprocessing ---
    print("\n--- MRI Image Preprocessing ---")
    dummy_mri = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    print(f"Input MRI shape: {dummy_mri.shape}, dtype: {dummy_mri.dtype}")
    print(f"Input pixel range: [{dummy_mri.min()}, {dummy_mri.max()}]")

    processed = preprocess_mri_image(dummy_mri)
    print(f"Output shape: {processed.shape}")
    print(f"Output dtype: {processed.dtype}")
    print(f"Output pixel range: [{processed.min():.4f}, {processed.max():.4f}]")

    # --- Segmentation Postprocessing ---
    print("\n--- Segmentation Mask Postprocessing ---")
    dummy_mask = np.random.rand(1, 224, 224, 1).astype(np.float32)
    print(f"Raw mask shape: {dummy_mask.shape}")
    post_mask = postprocess_segmentation_mask(dummy_mask, (512, 512))
    print(f"Postprocessed mask shape: {post_mask.shape}")
    print(f"Unique values in binary mask: {np.unique(post_mask)}")

    print("\n" + "=" * 60)
    print("  Preprocessing demo complete!")
    print("=" * 60)
