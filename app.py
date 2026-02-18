"""
StrokeSense – AI-Powered Stroke Prediction System
A comprehensive healthcare AI application for stroke risk assessment.
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import sys
import hashlib
import json
import os
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.preprocessing import (
    preprocess_clinical_data, 
    preprocess_mri_image,
    create_lesion_overlay,
    postprocess_segmentation_mask
)
from utils.prediction import (
    predict_clinical_risk,
    predict_mri_stroke,
    segment_lesion,
    get_feature_importance,
    check_models_status
)
from utils.recommendations import (
    get_recommendations,
    get_risk_category,
    get_stroke_warning_signs
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="StrokeSense – AI Prediction",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM STYLING - StrokeSense Theme
# ============================================================================

def load_custom_css():
    st.markdown("""
    <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* ========== FORCE LIGHT THEME ========== */
        
        /* Main app and all text - FORCE dark text */
        .stApp, .stApp * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        .stApp {
            background-color: #FFFFFF !important;
        }
        
        /* Force ALL text to be dark */
        .stApp p, .stApp span, .stApp div, .stApp label, 
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stMarkdown, .stMarkdown p, .stMarkdown span,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] li {
            color: #0F172A !important;
        }
        
        /* Sidebar - light gray background */
        section[data-testid="stSidebar"] {
            background-color: #F8FBFB !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        
        section[data-testid="stSidebar"] * {
            color: #0F172A !important;
        }
        
        section[data-testid="stSidebar"] .stRadio label span {
            color: #475569 !important;
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* ========== CUSTOM COMPONENTS ========== */
        
        /* Brand section */
        .brand-container {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 0.5rem 0 1.5rem 0;
            border-bottom: 1px solid #E2E8F0;
            margin-bottom: 1rem;
        }
        
        .brand-icon {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #2CA58D 0%, #0F766E 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            color: white !important;
        }
        
        .brand-text h1 {
            font-size: 1.3rem;
            font-weight: 700;
            color: #0F172A !important;
            margin: 0;
            line-height: 1.2;
        }
        
        .brand-text p {
            font-size: 0.8rem;
            color: #64748B !important;
            margin: 0;
        }
        
        /* Top header */
        .top-header {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 1rem 0;
            font-size: 0.95rem;
            font-weight: 500;
            color: #475569 !important;
        }
        
        /* Hero section */
        .hero-section {
            text-align: center;
            padding: 2.5rem 1rem;
            max-width: 720px;
            margin: 0 auto;
        }
        
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 18px;
            border: 1px solid #E2E8F0;
            border-radius: 50px;
            font-size: 0.9rem;
            color: #475569 !important;
            background: #FFFFFF;
            margin-bottom: 1.5rem;
        }
        
        .hero-headline {
            font-size: 2.75rem;
            font-weight: 700;
            line-height: 1.15;
            color: #0F172A !important;
            margin-bottom: 1.25rem;
        }
        
        .hero-headline .highlight {
            color: #2CA58D !important;
        }
        
        .hero-description {
            font-size: 1.1rem;
            line-height: 1.7;
            color: #475569 !important;
            margin-bottom: 2rem;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }
        
        /* Stats grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.25rem;
            padding: 2rem 1rem;
            max-width: 950px;
            margin: 0 auto;
        }
        
        @media (max-width: 900px) {
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .hero-headline {
                font-size: 2rem;
            }
        }
        
        .stat-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 1.5rem 1rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06);
            border: 1px solid #E2E8F0;
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(44, 165, 141, 0.12);
            border-color: #2CA58D;
        }
        
        .stat-value {
            font-size: 2.25rem;
            font-weight: 700;
            color: #2CA58D !important;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            font-size: 0.85rem;
            color: #475569 !important;
            line-height: 1.4;
        }
        
        /* Section titles */
        .section-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #0F172A !important;
            margin-bottom: 1.5rem;
        }
        
        /* ========== FORM STYLING ========== */
        
        /* Input fields */
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {
            background: #F8FBFB !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            color: #0F172A !important;
            padding: 0.75rem 1rem !important;
        }
        
        .stSelectbox > div > div {
            background: #F8FBFB !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
        }
        
        .stSelectbox > div > div > div {
            color: #0F172A !important;
        }
        
        /* Labels */
        .stNumberInput label, .stSelectbox label, .stTextInput label,
        .stCheckbox label, .stRadio label {
            color: #0F172A !important;
            font-weight: 500 !important;
        }
        
        /* Checkbox */
        .stCheckbox > label > span {
            color: #0F172A !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #2CA58D 0%, #0F766E 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.875rem 2rem !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 12px rgba(44, 165, 141, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(44, 165, 141, 0.4) !important;
        }
        
        /* Result cards */
        .result-card {
            background: linear-gradient(135deg, rgba(44, 165, 141, 0.08) 0%, rgba(44, 165, 141, 0.02) 100%);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            border: 1px solid rgba(44, 165, 141, 0.2);
        }
        
        .result-card h2, .result-card h3 {
            margin: 0;
        }
        
        .result-card p {
            color: #475569 !important;
            margin: 0.5rem 0 0 0;
        }
        
        .result-card.risk-low {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.02) 100%);
            border-color: rgba(16, 185, 129, 0.3);
        }
        
        .result-card.risk-medium {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.02) 100%);
            border-color: rgba(245, 158, 11, 0.3);
        }
        
        .result-card.risk-high {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.02) 100%);
            border-color: rgba(239, 68, 68, 0.3);
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #2CA58D 0%, #0F766E 100%) !important;
            border-radius: 10px !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: #F8FBFB !important;
            border-radius: 10px !important;
            border: 1px solid #E2E8F0 !important;
            color: #0F172A !important;
        }
        
        .streamlit-expanderContent {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-top: none !important;
        }
        
        /* Info/Warning/Success/Error boxes */
        .stAlert {
            background: #F0FDF9 !important;
            border: 1px solid #2CA58D !important;
            color: #0F172A !important;
        }
        
        [data-testid="stAlert"] p {
            color: #0F172A !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #2CA58D !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #475569 !important;
        }
        
        /* Login button */
        .login-btn {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0.75rem 1rem;
            background: rgba(44, 165, 141, 0.1);
            border-radius: 10px;
            color: #2CA58D !important;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .login-btn:hover {
            background: rgba(44, 165, 141, 0.2);
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            background: #F8FBFB !important;
            border: 2px dashed #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 1rem !important;
        }
        
        [data-testid="stFileUploader"] label {
            color: #0F172A !important;
        }
        
        /* Form styling */
        [data-testid="stForm"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 1.5rem;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: #2CA58D !important;
        }
        
        /* Plotly charts background */
        .js-plotly-plot .plotly .bg {
            fill: transparent !important;
        }
        
        /* ========== ADDITIONAL TEXT FIXES ========== */
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            background: #F8FAFC !important;
            padding: 6px !important;
            border-radius: 12px !important;
            gap: 8px !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: #64748B !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            font-weight: 500 !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: #E2E8F0 !important;
            color: #0F172A !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: #2CA58D !important;
            color: white !important;
        }
        
        /* Dropdown/Selectbox fixes */
        [data-baseweb="select"] {
            background: #F8FBFB !important;
        }
        
        [data-baseweb="select"] > div {
            color: #0F172A !important;
            background: #F8FBFB !important;
            border-color: #E2E8F0 !important;
            border-radius: 10px !important;
        }
        
        [data-baseweb="popover"] {
            background: #FFFFFF !important;
        }
        
        [data-baseweb="menu"] {
            background: #FFFFFF !important;
        }
        
        [data-baseweb="menu"] li {
            color: #0F172A !important;
        }
        
        [data-baseweb="menu"] li:hover {
            background: #F0FDF9 !important;
        }
        
        /* Radio buttons */
        .stRadio > div {
            background: transparent !important;
        }
        
        .stRadio label {
            color: #0F172A !important;
        }
        
        .stRadio label span {
            color: #0F172A !important;
        }
        
        /* Number input fixes */
        [data-testid="stNumberInput"] input {
            color: #0F172A !important;
            background: #F8FBFB !important;
        }
        
        [data-testid="stNumberInput"] label {
            color: #0F172A !important;
        }
        
        /* Text input placeholder */
        input::placeholder {
            color: #94A3B8 !important;
        }
        
        /* Strong and bold text */
        strong, b {
            color: #0F172A !important;
        }
        
        /* Links */
        a {
            color: #2CA58D !important;
        }
        
        /* Info, Success, Warning, Error messages */
        .stSuccess {
            background: #F0FDF9 !important;
            color: #0F172A !important;
        }
        
        .stError {
            background: #FEF2F2 !important;
            color: #0F172A !important;
        }
        
        .stWarning {
            background: #FFFBEB !important;
            color: #0F172A !important;
        }
        
        .stInfo {
            background: #F0F9FF !important;
            color: #0F172A !important;
        }
        
        /* Image caption */
        [data-testid="stImage"] figcaption {
            color: #64748B !important;
        }
        
        /* Column headers */
        [data-testid="column"] h4,
        [data-testid="column"] h3 {
            color: #0F172A !important;
        }
        
        /* Form elements text */
        [data-testid="stForm"] p,
        [data-testid="stForm"] span,
        [data-testid="stForm"] label {
            color: #0F172A !important;
        }
        
        /* Horizontal rule */
        hr {
            border-color: #E2E8F0 !important;
        }
        
        /* ========== LOGIN PAGE ========== */
        
        .login-container {
            max-width: 420px;
            margin: 3rem auto;
            padding: 2.5rem;
            background: #FFFFFF;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(15, 23, 42, 0.1);
            border: 1px solid #E2E8F0;
        }
        
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .login-header .logo {
            width: 70px;
            height: 70px;
            background: linear-gradient(135deg, #2CA58D 0%, #0F766E 100%);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            margin: 0 auto 1rem auto;
            color: white !important;
        }
        
        .login-header h1 {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0F172A !important;
            margin: 0 0 0.5rem 0;
        }
        
        .login-header p {
            color: #64748B !important;
            font-size: 0.95rem;
        }
        
        .login-tabs {
            display: flex;
            gap: 0;
            margin-bottom: 1.5rem;
            background: #F1F5F9;
            border-radius: 10px;
            padding: 4px;
        }
        
        .login-tab {
            flex: 1;
            padding: 10px 20px;
            text-align: center;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            color: #64748B !important;
            transition: all 0.2s ease;
        }
        
        .login-tab.active {
            background: #FFFFFF;
            color: #0F172A !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }
        
        .divider {
            display: flex;
            align-items: center;
            text-align: center;
            margin: 1.5rem 0;
            color: #94A3B8 !important;
            font-size: 0.85rem;
        }
        
        .divider::before,
        .divider::after {
            content: '';
            flex: 1;
            border-bottom: 1px solid #E2E8F0;
        }
        
        .divider::before {
            margin-right: 1rem;
        }
        
        .divider::after {
            margin-left: 1rem;
        }
        
        .social-login {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .social-btn {
            flex: 1;
            padding: 12px;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            background: #FFFFFF;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            font-size: 0.9rem;
            color: #475569 !important;
            transition: all 0.2s ease;
        }
        
        .social-btn:hover {
            background: #F8FAFC;
            border-color: #CBD5E1;
        }
        
        .forgot-password {
            text-align: right;
            margin-bottom: 1rem;
        }
        
        .forgot-password a {
            color: #2CA58D !important;
            font-size: 0.875rem;
            text-decoration: none;
        }
        
        .login-footer {
            text-align: center;
            margin-top: 1.5rem;
            color: #64748B !important;
            font-size: 0.9rem;
        }
        
        .login-footer a {
            color: #2CA58D !important;
            font-weight: 500;
            text-decoration: none;
        }
        
        .user-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(44, 165, 141, 0.1);
            border-radius: 20px;
            font-size: 0.85rem;
            color: #2CA58D !important;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

# Path for storing user credentials securely
USERS_FILE = Path(__file__).parent / "data" / "users.json"

def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt"""
    salt = "strokesense_secure_salt_2024"
    return hashlib.sha256((password + salt).encode()).hexdigest()

def load_users() -> dict:
    """Load users from JSON file"""
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users: dict):
    """Save users to JSON file"""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def register_user(email: str, password: str, full_name: str) -> tuple[bool, str]:
    """Register a new user"""
    users = load_users()
    
    if email.lower() in users:
        return False, "Email already registered"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    if not email or '@' not in email:
        return False, "Please enter a valid email"
    
    users[email.lower()] = {
        'password_hash': hash_password(password),
        'full_name': full_name,
        'email': email.lower()
    }
    save_users(users)
    return True, "Account created successfully!"

def authenticate_user(email: str, password: str) -> tuple[bool, str, str]:
    """Authenticate a user"""
    users = load_users()
    
    if not email or not password:
        return False, "", "Please enter both email and password"
    
    user = users.get(email.lower())
    if not user:
        return False, "", "Invalid email or password"
    
    if user['password_hash'] != hash_password(password):
        return False, "", "Invalid email or password"
    
    return True, user['full_name'], "Login successful!"

def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'user_email' not in st.session_state:
        st.session_state.user_email = ""
    if 'show_login' not in st.session_state:
        st.session_state.show_login = False

def render_sidebar():
    with st.sidebar:
        # Branding
        st.markdown("""
        <div class="brand-container">
            <div class="brand-icon">SS</div>
            <div class="brand-text">
                <h1>StrokeSense</h1>
                <p>AI Prediction</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["Home", "Clinical Data", "MRI Images", "Help"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # System Status
        st.markdown("#### System Status")
        status = check_models_status()
        
        col1, col2 = st.columns(2)
        with col1:
            if status['random_forest']['exists']:
                st.success("Clinical ✓")
            else:
                st.info("Demo Mode")
        with col2:
            if status['cnn']['exists']:
                st.success("MRI ✓")
            else:
                st.info("Demo Mode")
        
        st.markdown("---")
        
        # User section
        if st.session_state.logged_in:
            st.markdown(f"""
            <div class="user-badge">
                <span></span>
                <span>{st.session_state.username}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.rerun()
        else:
            if st.button("Login / Sign Up", use_container_width=True):
                st.session_state.show_login = True
                st.rerun()
        
        return page


# ============================================================================
# LOGIN PAGE
# ============================================================================

def render_login_page():
    """Render login/signup page with secure authentication"""
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-header">
            <div class="logo">SS</div>
            <h1>Welcome to StrokeSense</h1>
            <p>Sign in to access AI-powered stroke prediction</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Security notice
        st.markdown("""
        <div style="background: #F0FDF9; border: 1px solid #2CA58D; border-radius: 10px; padding: 12px; margin-bottom: 1.5rem; text-align: center;">
            <span style="color: #0F766E;">🔒 Your data is securely encrypted</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Login/Signup tabs
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                st.markdown("**Email**")
                email = st.text_input("Email", placeholder="Enter your email", label_visibility="collapsed")
                
                st.markdown("**Password**")
                password = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed")
                
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    remember = st.checkbox("Remember me")
                with col_b:
                    st.markdown('<p style="text-align: right; margin: 0;"><a href="#" style="color: #2CA58D;">Forgot password?</a></p>', unsafe_allow_html=True)
                
                login_btn = st.form_submit_button("Sign In", use_container_width=True)
                
                if login_btn:
                    success, username, message = authenticate_user(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_email = email.lower()
                        st.session_state.show_login = False
                        st.success(message + " Redirecting...")
                        st.rerun()
                    else:
                        st.error(message)

        
        with tab2:
            with st.form("signup_form"):
                st.markdown("**Full Name**")
                full_name = st.text_input("Full Name", placeholder="Enter your full name", label_visibility="collapsed")
                
                st.markdown("**Email**")
                signup_email = st.text_input("Signup Email", placeholder="Enter your email", label_visibility="collapsed")
                
                st.markdown("**Password** (min 6 characters)")
                signup_password = st.text_input("Signup Password", type="password", placeholder="Create a strong password", label_visibility="collapsed")
                
                st.markdown("**Confirm Password**")
                confirm_password = st.text_input("Confirm", type="password", placeholder="Confirm your password", label_visibility="collapsed")
                
                agree = st.checkbox("I agree to the Terms of Service and Privacy Policy")
                
                signup_btn = st.form_submit_button("Create Account", use_container_width=True)
                
                if signup_btn:
                    if not full_name:
                        st.error("Please enter your full name")
                    elif not signup_email:
                        st.error("Please enter your email")
                    elif not signup_password:
                        st.error("Please create a password")
                    elif signup_password != confirm_password:
                        st.error("Passwords don't match")
                    elif not agree:
                        st.error("Please agree to the terms and conditions")
                    else:
                        success, message = register_user(signup_email, signup_password, full_name)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = full_name.split()[0].title()
                            st.session_state.user_email = signup_email.lower()
                            st.session_state.show_login = False
                            st.success(message + " Redirecting...")
                            st.rerun()
                        else:
                            st.error(message)
        
        # Back button
        st.markdown("---")
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.show_login = False
            st.rerun()


# ============================================================================
# HOME PAGE
# ============================================================================

def render_home_page():
    # Top header
    st.markdown("""
    <div class="top-header">
        <span></span>
        <span>AI-Powered Stroke Prediction System</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-badge">
            <span></span> Every Second Counts
        </div>
        <h1 class="hero-headline">
            Your Life is <span class="highlight">Precious</span><br>
            Protect It from Stroke
        </h1>
        <p class="hero-description">
            Stroke is a leading cause of death and disability worldwide. 
            But here's the hope: <strong>80% of strokes are preventable</strong>. 
            Take control of your health today with early risk assessment and awareness.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Get Started", use_container_width=True):
            st.session_state['page'] = "Clinical Data"
            st.rerun()
    
    # Stats cards
    st.markdown("""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">15M</div>
            <div class="stat-label">Strokes occur globally each year</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">80%</div>
            <div class="stat-label">Of strokes are preventable</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">5M</div>
            <div class="stat-label">Lives lost to stroke annually</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">4.5hrs</div>
            <div class="stat-label">Critical window for treatment</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # FAST Warning Signs
    st.markdown("---")
    st.markdown("### Know the Signs - Act FAST")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value" style="color: #EF4444;">F</div>
            <div class="stat-label"><strong>Face</strong><br>Does one side droop?</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value" style="color: #EF4444;">A</div>
            <div class="stat-label"><strong>Arms</strong><br>Can both arms be raised?</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value" style="color: #EF4444;">S</div>
            <div class="stat-label"><strong>Speech</strong><br>Is speech slurred?</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value" style="color: #EF4444;">T</div>
            <div class="stat-label"><strong>Time</strong><br>Call emergency now!</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# CLINICAL DATA PAGE
# ============================================================================

def render_clinical_page():
    st.markdown("""
    <div class="top-header">
        <span></span>
        <span>Clinical Risk Assessment</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
        
        with st.form("clinical_form"):
            # Demographics
            st.markdown("**Demographics**")
            demo_col1, demo_col2 = st.columns(2)
            with demo_col1:
                age = st.number_input("Age", min_value=1, max_value=120, value=None, placeholder="Enter age")
                gender = st.selectbox("Gender", ["-- Select --", "Male", "Female", "Other"], index=0)
            with demo_col2:
                ever_married = st.selectbox("Ever Married", ["-- Select --", "Yes", "No"], index=0)
                residence_type = st.selectbox("Residence Type", ["-- Select --", "Urban", "Rural"], index=0)
            
            st.markdown("**Health Measurements**")
            health_col1, health_col2 = st.columns(2)
            with health_col1:
                avg_glucose = st.number_input("Glucose Level (mg/dL)", min_value=50.0, max_value=400.0, value=None, placeholder="e.g. 100")
            with health_col2:
                bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=None, placeholder="e.g. 25")
            
            st.markdown("**Medical History**")
            history_col1, history_col2 = st.columns(2)
            with history_col1:
                hypertension = st.checkbox("Hypertension")
                heart_disease = st.checkbox("Heart Disease")
            with history_col2:
                work_type = st.selectbox("Work Type", ["-- Select --", "Private", "Self-employed", "Govt_job", "children", "Never_worked"], index=0)
                smoking_status = st.selectbox("Smoking Status", ["-- Select --", "never smoked", "formerly smoked", "smokes", "Unknown"], index=0)
            
            submitted = st.form_submit_button("Analyze Risk", use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)
        
        if submitted:
            # Validate all required fields
            validation_errors = []
            if age is None:
                validation_errors.append("Age")
            if gender == "-- Select --":
                validation_errors.append("Gender")
            if ever_married == "-- Select --":
                validation_errors.append("Ever Married")
            if residence_type == "-- Select --":
                validation_errors.append("Residence Type")
            if avg_glucose is None:
                validation_errors.append("Glucose Level")
            if bmi is None:
                validation_errors.append("BMI")
            if work_type == "-- Select --":
                validation_errors.append("Work Type")
            if smoking_status == "-- Select --":
                validation_errors.append("Smoking Status")
            
            if validation_errors:
                st.error(f"Please fill in: {', '.join(validation_errors)}")
            else:
                with st.spinner("Analyzing clinical data..."):
                    patient_data = {
                        'age': age,
                        'gender': gender,
                        'ever_married': ever_married,
                        'work_type': work_type,
                        'residence_type': residence_type,
                        'avg_glucose_level': avg_glucose,
                        'bmi': bmi,
                        'hypertension': 1 if hypertension else 0,
                        'heart_disease': 1 if heart_disease else 0,
                        'smoking_status': smoking_status
                    }
                    
                    features = preprocess_clinical_data(patient_data)
                    result = predict_clinical_risk(features)
                
                    # Result card
                    risk_class = "risk-low" if result['category'] == "Low Risk" else \
                                "risk-medium" if result['category'] == "Medium Risk" else "risk-high"
                    
                    st.markdown(f"""
                    <div class="result-card {risk_class}">
                        <h2 style="font-size: 3.5rem; font-weight: 700; color: {result['color']}; margin: 0;">
                            {result['percentage']:.1f}%
                        </h2>
                        <h3 style="color: {result['color']}; margin: 0.5rem 0;">{result['category']}</h3>
                        <p style="color: var(--text-secondary);">Confidence: {result['confidence']*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['percentage'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#E2E8F0"},
                            'bar': {'color': result['color']},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#E2E8F0",
                            'steps': [
                                {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                                {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.2)"},
                                {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                            ],
                        },
                        title={'text': "Risk Score", 'font': {'size': 16, 'color': '#475569'}}
                    ))
                    fig.update_layout(
                        height=220,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#475569'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    recommendations = get_recommendations(result['probability'], patient_data)
                    
                    with st.expander("Lifestyle Recommendations", expanded=True):
                        for rec in recommendations['lifestyle']:
                            st.markdown(rec)
                    
                    with st.expander("Medical Recommendations"):
                        for rec in recommendations['medical']:
                            st.markdown(rec)
        else:
            st.info("Fill in the patient information and click 'Analyze Risk' to see results.")
            
            with st.expander("Input Guide"):
                st.markdown("""
                - **Age**: Patient's age in years
                - **Gender**: Biological sex
                - **Glucose Level**: Average blood glucose (mg/dL)
                - **BMI**: Body Mass Index
                - **Hypertension/Heart Disease**: Medical conditions
                - **Smoking Status**: Smoking behavior
                """)


# ============================================================================
# MRI PAGE
# ============================================================================

def render_mri_page():
    st.markdown("""
    <div class="top-header">
        <span></span>
        <span>MRI Image Analysis</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-title">Upload MRI Scan</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a brain MRI scan for stroke detection analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI", use_container_width=True)
            
            st.markdown("**Analysis Options**")
            run_detection = st.checkbox("Stroke Detection (CNN)", value=True)
            run_segmentation = st.checkbox("Lesion Segmentation (U-Net)", value=True)
            
            if st.button("Analyze MRI", use_container_width=True):
                with col2:
                    st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)
                    
                    img_array = preprocess_mri_image(image)
                    original_size = image.size
                    
                    if run_detection:
                        with st.spinner("Running stroke detection..."):
                            detection_result = predict_mri_stroke(img_array)
                            
                            result_class = "risk-high" if detection_result['stroke_detected'] else "risk-low"
                            st.markdown(f"""
                            <div class="result-card {result_class}">
                                <h3 style="color: {detection_result['color']};">{detection_result['result']}</h3>
                                <p>Confidence: {detection_result['confidence']*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.progress(detection_result['confidence'])
                    
                    if run_segmentation:
                        with st.spinner("Running lesion segmentation..."):
                            mask = segment_lesion(img_array)
                            
                            original_array = np.array(image)
                            mask_resized = postprocess_segmentation_mask(mask, original_size)
                            overlay = create_lesion_overlay(original_array, mask_resized, color=(239, 68, 68), alpha=0.4)
                            
                            st.markdown("**Lesion Segmentation**")
                            seg_col1, seg_col2 = st.columns(2)
                            with seg_col1:
                                st.image(original_array, caption="Original", use_container_width=True)
                            with seg_col2:
                                st.image(overlay, caption="Detected Regions", use_container_width=True)
                            
                            lesion_pixels = np.sum(mask_resized > 0)
                            total_pixels = mask_resized.shape[0] * mask_resized.shape[1]
                            lesion_percentage = (lesion_pixels / total_pixels) * 100
                            
                            st.metric("Affected Area", f"{lesion_percentage:.2f}%")
    
    if uploaded_file is None:
        with col2:
            st.info("Upload an MRI image to begin analysis")
            
            with st.expander("Supported Formats"):
                st.markdown("""
                - **PNG** (.png)
                - **JPEG** (.jpg, .jpeg)
                - **BMP** (.bmp)
                - **TIFF** (.tiff)
                """)


# ============================================================================
# HELP PAGE
# ============================================================================

def render_help_page():
    st.markdown("""
    <div class="top-header">
        <span></span>
        <span>Help & Information</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">About StrokeSense</div>', unsafe_allow_html=True)
    
    st.markdown("""
    StrokeSense is an AI-powered healthcare application designed to help assess stroke risk
    and detect potential stroke patterns in brain MRI images.
    
    ### Features
    
    - **Clinical Risk Assessment**: Uses a Random Forest model trained on patient health data
      to predict stroke risk probability
    - **MRI Stroke Detection**: CNN-based deep learning model analyzes brain MRI scans
      for stroke patterns
    - **Lesion Segmentation**: U-Net architecture highlights affected brain regions
    
    ### Important Disclaimer
    
    **This tool is for educational and informational purposes only.**
    
    - Not a replacement for professional medical diagnosis
    - Always consult healthcare professionals for medical decisions
    - In case of stroke symptoms, call emergency services immediately
    
    ### Emergency Signs (FAST)
    
    - **F**ace: Ask the person to smile. Does one side droop?
    - **A**rms: Ask them to raise both arms. Does one drift downward?
    - **S**peech: Ask them to repeat a simple phrase. Is speech slurred?
    - **T**ime: If you observe any of these signs, call emergency immediately!
    """)
    
    st.markdown("---")
    
    # Feature importance visualization
    st.markdown('<div class="section-title">📊 Model Insights</div>', unsafe_allow_html=True)
    
    importance = get_feature_importance()
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    fig = go.Figure(go.Bar(
        x=list(sorted_importance.values()),
        y=list(sorted_importance.keys()),
        orientation='h',
        marker=dict(
            color='#2CA58D',
            line=dict(color='#0F766E', width=1)
        )
    ))
    
    fig.update_layout(
        title="Feature Importance in Clinical Model",
        xaxis_title="Importance Score",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#475569'},
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Initialize session state
    init_session_state()
    
    load_custom_css()
    
    # Check if login page should be shown
    if st.session_state.show_login:
        render_login_page()
    else:
        page = render_sidebar()
        
        if page == "🏠  Home":
            render_home_page()
        elif page == "📋  Clinical Data":
            render_clinical_page()
        elif page == "🖼️  MRI Images":
            render_mri_page()
        elif page == "❓  Help":
            render_help_page()


if __name__ == "__main__":
    main()
