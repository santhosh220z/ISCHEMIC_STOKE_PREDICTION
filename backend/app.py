"""
StrokeSense Flask API Backend
RESTful API for stroke prediction services.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import hashlib
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io
import base64

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import preprocess_clinical_data, preprocess_mri_image
from utils.prediction import predict_clinical_risk, predict_mri_stroke, segment_lesion
from utils.recommendations import get_recommendations

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])

# User storage
USERS_FILE = Path(__file__).parent.parent / "data" / "users.json"

def hash_password(password: str) -> str:
    """Hash password with SHA-256"""
    salt = "strokesense_secure_salt_2024"
    return hashlib.sha256((password + salt).encode()).hexdigest()

def load_users() -> dict:
    """Load users from JSON"""
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users: dict):
    """Save users to JSON"""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "StrokeSense API"})


# ============================================================================
# AUTHENTICATION
# ============================================================================

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email', '').lower()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({"success": False, "message": "Email and password required"}), 400
    
    users = load_users()
    user = users.get(email)
    
    if not user or user['password_hash'] != hash_password(password):
        return jsonify({"success": False, "message": "Invalid email or password"}), 401
    
    return jsonify({
        "success": True,
        "message": "Login successful",
        "user": {
            "email": email,
            "fullName": user['full_name']
        }
    })


@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email', '').lower()
    password = data.get('password', '')
    full_name = data.get('fullName', '')
    
    if not email or not password or not full_name:
        return jsonify({"success": False, "message": "All fields required"}), 400
    
    if '@' not in email:
        return jsonify({"success": False, "message": "Invalid email format"}), 400
    
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters"}), 400
    
    users = load_users()
    
    if email in users:
        return jsonify({"success": False, "message": "Email already registered"}), 409
    
    users[email] = {
        'password_hash': hash_password(password),
        'full_name': full_name,
        'email': email
    }
    save_users(users)
    
    return jsonify({
        "success": True,
        "message": "Account created successfully",
        "user": {
            "email": email,
            "fullName": full_name
        }
    })


# ============================================================================
# PATIENT RECORDS
# ============================================================================

RECORDS_FILE = Path(__file__).parent.parent / "data" / "records.json"

def load_records() -> dict:
    """Load patient records from JSON"""
    if RECORDS_FILE.exists():
        try:
            with open(RECORDS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_records(records: dict):
    """Save patient records to JSON"""
    RECORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RECORDS_FILE, 'w') as f:
        json.dump(records, f, indent=2)


@app.route('/api/records/save', methods=['POST'])
def save_record():
    """Save a new patient record (MRI or clinical analysis)"""
    import uuid
    from datetime import datetime
    
    data = request.json
    email = data.get('email')
    record_type = data.get('type')  # 'mri' or 'clinical'
    result = data.get('result')
    
    if not email or not record_type or not result:
        return jsonify({"success": False, "message": "Missing required fields"}), 400
    
    records = load_records()
    
    # Initialize user's records list if not exists
    if email not in records:
        records[email] = []
    
    # Create new record
    new_record = {
        "id": str(uuid.uuid4()),
        "type": record_type,
        "date": datetime.now().isoformat(),
        "result": result
    }
    
    records[email].append(new_record)
    save_records(records)
    
    return jsonify({
        "success": True,
        "message": "Record saved successfully",
        "record": new_record
    })


@app.route('/api/records/<email>', methods=['GET'])
def get_records(email):
    """Get all records for a patient"""
    records = load_records()
    
    patient_records = records.get(email, [])
    
    # Sort by date, newest first
    patient_records.sort(key=lambda x: x.get('date', ''), reverse=True)
    
    return jsonify({
        "success": True,
        "records": patient_records
    })


@app.route('/api/records/<email>/<record_id>', methods=['DELETE'])
def delete_record(email, record_id):
    """Delete a specific record"""
    records = load_records()
    
    if email not in records:
        return jsonify({"success": False, "message": "No records found"}), 404
    
    # Find and remove the record
    original_length = len(records[email])
    records[email] = [r for r in records[email] if r.get('id') != record_id]
    
    if len(records[email]) == original_length:
        return jsonify({"success": False, "message": "Record not found"}), 404
    
    save_records(records)
    
    return jsonify({
        "success": True,
        "message": "Record deleted successfully"
    })


# ============================================================================
# CLINICAL PREDICTION
# ============================================================================

@app.route('/api/predict/clinical', methods=['POST'])
def predict_clinical():
    data = request.json
    
    required = ['age', 'gender', 'everMarried', 'workType', 'residenceType', 
                'avgGlucose', 'bmi', 'hypertension', 'heartDisease', 'smokingStatus']
    
    for field in required:
        if field not in data:
            return jsonify({"success": False, "message": f"Missing field: {field}"}), 400
    
    # Map to model format
    patient_data = {
        'age': data['age'],
        'gender': data['gender'],
        'ever_married': data['everMarried'],
        'work_type': data['workType'],
        'residence_type': data['residenceType'],
        'avg_glucose_level': data['avgGlucose'],
        'bmi': data['bmi'],
        'hypertension': 1 if data['hypertension'] else 0,
        'heart_disease': 1 if data['heartDisease'] else 0,
        'smoking_status': data['smokingStatus']
    }
    
    try:
        features = preprocess_clinical_data(patient_data)
        result = predict_clinical_risk(features)
        recommendations = get_recommendations(result['probability'], patient_data)
        
        return jsonify({
            "success": True,
            "result": {
                "probability": result['probability'],
                "percentage": result['percentage'],
                "category": result['category'],
                "color": result['color'],
                "confidence": result['confidence']
            },
            "recommendations": {
                "lifestyle": recommendations.get('lifestyle', []),
                "medical": recommendations.get('medical', [])
            }
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ============================================================================
# MRI PREDICTION
# ============================================================================

@app.route('/api/predict/mri', methods=['POST'])
def predict_mri():
    print("Received MRI prediction request", flush=True)
    if 'image' not in request.files:
        print("No image in request files", flush=True)
        return jsonify({"success": False, "message": "No image provided"}), 400
    
    file = request.files['image']
    
    try:
        image = Image.open(file.stream)
        img_array = preprocess_mri_image(image)
        
        # Run detection
        detection_result = predict_mri_stroke(img_array)
        
        # Run segmentation
        segmentation_mask = segment_lesion(img_array)
        
        # Calculate affected area
        lesion_pixels = np.sum(segmentation_mask > 0.5)
        total_pixels = segmentation_mask.shape[0] * segmentation_mask.shape[1]
        affected_percentage = (lesion_pixels / total_pixels) * 100
        
        # Always generate lesion overlay image
        mask_2d = segmentation_mask[:, :, 0] if len(segmentation_mask.shape) == 3 else segmentation_mask
        
        # Create RGBA overlay image
        overlay = np.zeros((224, 224, 4), dtype=np.uint8)
        
        # Use red for stroke detected, yellow/orange for areas of interest when no stroke
        if detection_result['prediction'] == 'Stroke Detected':
            overlay[mask_2d > 0.5, 0] = 255  # Red channel
            overlay[mask_2d > 0.5, 3] = 150  # Alpha
        else:
            # Show lesion areas in yellow/orange as "areas of interest"
            overlay[mask_2d > 0.5, 0] = 255  # Red channel
            overlay[mask_2d > 0.5, 1] = 165  # Green channel (makes orange)
            overlay[mask_2d > 0.5, 3] = 120  # Alpha (slightly more transparent)
        
        # Convert to PIL and then to base64
        overlay_img = Image.fromarray(overlay, mode='RGBA')
        buffered = io.BytesIO()
        overlay_img.save(buffered, format="PNG")
        lesion_overlay_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            "success": True,
            "result": {
                "hasStroke": detection_result['prediction'] == 'Stroke Detected',
                "confidence": float(detection_result['confidence']),
                "prediction": detection_result['prediction'],
                "affectedArea": round(affected_percentage, 2),
                "lesionOverlay": lesion_overlay_base64
            }
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == '__main__':
    print("Starting StrokeSense API on http://localhost:5000")
    app.run(debug=True, port=5000, use_reloader=False)
