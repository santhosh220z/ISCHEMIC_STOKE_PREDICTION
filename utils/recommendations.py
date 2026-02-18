"""
Preventive recommendation system based on stroke risk level.
"""


# Risk thresholds
LOW_RISK_THRESHOLD = 0.30
MEDIUM_RISK_THRESHOLD = 0.70


def get_risk_category(probability):
    """
    Determine risk category based on stroke probability.
    
    Args:
        probability: Float between 0 and 1
        
    Returns:
        Tuple of (category_name, color_code)
    """
    if probability < LOW_RISK_THRESHOLD:
        return "Low Risk", "#28a745"  # Green
    elif probability < MEDIUM_RISK_THRESHOLD:
        return "Medium Risk", "#ffc107"  # Yellow/Orange
    else:
        return "High Risk", "#dc3545"  # Red


def get_recommendations(probability, patient_data=None):
    """
    Get personalized preventive recommendations based on risk level.
    
    Args:
        probability: Stroke risk probability (0-1)
        patient_data: Optional dict of patient clinical data for personalization
        
    Returns:
        Dictionary containing recommendations and lifestyle advice
    """
    category, _ = get_risk_category(probability)
    
    recommendations = {
        "Low Risk": {
            "title": "Low Risk - Maintain Your Health",
            "summary": "Your current health indicators suggest a low risk of ischemic stroke. Continue your healthy lifestyle!",
            "lifestyle": [
                "- Maintain a balanced diet rich in fruits, vegetables, and whole grains",
                "- Engage in regular physical activity (at least 150 minutes/week)",
                "- Keep a healthy sleep schedule (7-9 hours per night)",
                "- Stay hydrated and limit alcohol consumption",
                "- Practice stress management techniques"
            ],
            "medical": [
                "- Schedule routine health checkups annually",
                "- Monitor blood pressure periodically",
                "- Keep vaccinations up to date",
                "- Maintain awareness of family health history"
            ],
            "follow_up": "Annual routine checkup recommended",
            "urgency": "low"
        },
        "Medium Risk": {
            "title": "Medium Risk - Take Preventive Action",
            "summary": "Some of your health indicators suggest an elevated risk. Proactive management can significantly reduce your stroke risk.",
            "lifestyle": [
                "- Adopt a heart-healthy diet (DASH or Mediterranean diet)",
                "- Increase physical activity to at least 30 minutes daily",
                "- If you smoke, create a cessation plan immediately",
                "- Limit salt intake to less than 2,300mg per day",
                "- Reduce saturated fat and processed food consumption",
                "- Maintain a healthy weight (BMI 18.5-24.9)"
            ],
            "medical": [
                "- Monitor blood pressure weekly",
                "- Check blood glucose levels regularly",
                "- Schedule a cardiovascular health assessment",
                "- Discuss preventive medications with your doctor",
                "- Consider cholesterol screening"
            ],
            "follow_up": "Medical consultation within 1-2 weeks recommended",
            "urgency": "medium"
        },
        "High Risk": {
            "title": "High Risk - Immediate Action Required",
            "summary": "Your health indicators suggest a high risk of ischemic stroke. Please seek medical attention promptly.",
            "lifestyle": [
                "! Stop smoking immediately if applicable",
                "! Strictly follow a low-sodium, heart-healthy diet",
                "! Avoid alcohol consumption",
                "! Engage in light physical activity as approved by your doctor",
                "! Monitor for warning signs of stroke (FAST: Face, Arms, Speech, Time)"
            ],
            "medical": [
                "- Schedule immediate consultation with a neurologist",
                "- Ensure strict adherence to prescribed medications",
                "- Daily monitoring of blood pressure and glucose",
                "- Consider advanced imaging studies (MRI/CT angiography)",
                "- Discuss antiplatelet or anticoagulant therapy",
                "- Evaluate for carotid artery disease"
            ],
            "follow_up": "Immediate medical consultation required",
            "urgency": "high",
            "emergency_signs": [
                "! Sudden numbness or weakness in face, arm, or leg",
                "! Sudden confusion or trouble speaking",
                "! Sudden trouble seeing in one or both eyes",
                "! Sudden severe headache with no known cause",
                "! Sudden trouble walking, dizziness, or loss of balance"
            ]
        }
    }
    
    base_recommendation = recommendations.get(category, recommendations["Medium Risk"])
    
    # Add personalized recommendations based on patient data
    if patient_data:
        personalized = get_personalized_recommendations(patient_data)
        base_recommendation["personalized"] = personalized
    
    return base_recommendation


def get_personalized_recommendations(patient_data):
    """
    Generate personalized recommendations based on specific patient data.
    
    Args:
        patient_data: Dictionary of patient clinical data
        
    Returns:
        List of personalized recommendation strings
    """
    personalized = []
    
    # Check hypertension
    if patient_data.get('hypertension', 0) == 1:
        personalized.append("Your hypertension requires careful management. Consider home BP monitoring and medication adherence.")
    
    # Check heart disease
    if patient_data.get('heart_disease', 0) == 1:
        personalized.append("With heart disease present, regular cardiac monitoring is essential. Discuss aspirin therapy with your cardiologist.")
    
    # Check smoking status
    smoking = patient_data.get('smoking_status', '').lower()
    if smoking in ['smokes', 'formerly smoked']:
        if smoking == 'smokes':
            personalized.append("Smoking significantly increases stroke risk. Consider nicotine replacement therapy or cessation programs.")
        else:
            personalized.append("Great job quitting smoking! Continue avoiding tobacco products to maintain lower risk.")
    
    # Check glucose levels
    glucose = patient_data.get('avg_glucose_level', 100)
    if glucose > 126:
        personalized.append("Elevated glucose levels detected. Consider diabetic screening and dietary modifications.")
    elif glucose > 100:
        personalized.append("Pre-diabetic glucose range. Focus on diet and exercise to prevent diabetes development.")
    
    # Check BMI
    bmi = patient_data.get('bmi', 25)
    if bmi > 30:
        personalized.append("BMI indicates obesity. Weight management through diet and exercise is strongly recommended.")
    elif bmi > 25:
        personalized.append("BMI indicates overweight. Consider gradual weight loss through lifestyle changes.")
    
    # Check age
    age = patient_data.get('age', 50)
    if age > 65:
        personalized.append("Age is a non-modifiable risk factor. Focus on managing controllable risk factors more strictly.")
    
    # Check blood pressure
    systolic = patient_data.get('systolic_bp', 120)
    diastolic = patient_data.get('diastolic_bp', 80)
    if systolic >= 140 or diastolic >= 90:
        personalized.append("Blood pressure is elevated. Target BP below 130/80 mmHg with medication and lifestyle changes.")
    
    return personalized


def get_stroke_warning_signs():
    """
    Return information about stroke warning signs (FAST protocol).
    
    Returns:
        Dictionary with stroke warning information
    """
    return {
        "title": "Know the Signs of Stroke - Act FAST",
        "fast": {
            "F": {
                "letter": "Face",
                "description": "Ask the person to smile. Does one side of the face droop?"
            },
            "A": {
                "letter": "Arms", 
                "description": "Ask the person to raise both arms. Does one arm drift downward?"
            },
            "S": {
                "letter": "Speech",
                "description": "Ask the person to repeat a simple phrase. Is their speech slurred or strange?"
            },
            "T": {
                "letter": "Time",
                "description": "If you see any of these signs, call emergency services immediately. Time is critical!"
            }
        },
        "additional_signs": [
            "Sudden numbness or weakness, especially on one side of the body",
            "Sudden confusion or trouble understanding",
            "Sudden trouble seeing in one or both eyes",
            "Sudden trouble walking, dizziness, or loss of coordination",
            "Sudden severe headache with no known cause"
        ],
        "emergency_number": "Call 911 (US) or your local emergency number immediately"
    }
