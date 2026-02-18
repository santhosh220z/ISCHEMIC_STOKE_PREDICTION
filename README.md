# StrokeSense – AI-Powered Stroke Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/React-19.2-61DAFB?style=for-the-badge&logo=react" alt="React" />
  <img src="https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask" alt="Flask" />
  <img src="https://img.shields.io/badge/TensorFlow-ML-FF6F00?style=for-the-badge&logo=tensorflow" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python" alt="Python" />
</p>

StrokeSense is a comprehensive healthcare AI application designed for stroke risk assessment using both clinical data and MRI image analysis. It features a modern React frontend with a Flask API backend, leveraging machine learning models to predict stroke probability and identify risk factors.

## ✨ Features

### 🩺 Clinical Risk Assessment
- Predicts stroke probability based on patient demographics and health metrics
- Input parameters: Age, Gender, Glucose Level, BMI, Hypertension, Heart Disease, Smoking Status, etc.
- Real-time risk visualization with interactive gauges

### 🧠 MRI Image Analysis
- Upload and analyze brain MRI scans for stroke detection
- Deep learning-based stroke pattern detection (CNN model)
- Lesion segmentation using U-Net architecture
- Confidence scores and affected area percentage

### 📊 Risk Visualization
- Interactive charts and graphs powered by modern UI components
- Risk category classification (Low, Medium, High)
- Feature importance analysis

### 💡 Personalized Recommendations
- Tailored lifestyle advice based on risk factors
- Medical recommendations for high-risk patients

### 🔐 User Authentication
- Secure login and registration system
- User session management

## 📁 Project Structure

```
stroke_prediction/
├── frontend/                  # React + Vite frontend
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   │   ├── Layout.jsx     # App layout wrapper
│   │   │   ├── RiskGauge.jsx  # Risk visualization gauge
│   │   │   └── Sidebar.jsx    # Navigation sidebar
│   │   ├── pages/             # Application pages
│   │   │   ├── HomePage.jsx   # Landing page
│   │   │   ├── ClinicalPage.jsx   # Clinical risk assessment
│   │   │   ├── MRIPage.jsx    # MRI image analysis
│   │   │   ├── LoginPage.jsx  # Authentication
│   │   │   └── HelpPage.jsx   # Help & documentation
│   │   ├── App.jsx            # Main app with routing
│   │   └── main.jsx           # Entry point
│   ├── package.json           # Frontend dependencies
│   └── vite.config.js         # Vite configuration with API proxy
│
├── backend/                   # Flask API backend
│   └── app.py                 # RESTful API endpoints
│
├── utils/                     # Python utility modules
│   ├── preprocessing.py       # Data preprocessing functions
│   ├── prediction.py          # ML model inference logic
│   └── recommendations.py     # Recommendation engine
│
├── models/                    # Trained ML/DL models
│   ├── random_forest.pkl      # Clinical prediction model
│   ├── cnn_model.h5           # MRI stroke detection (CNN)
│   └── unet_model.h5          # Lesion segmentation (U-Net)
│
├── datasets/                  # Training datasets
├── data/                      # User data storage
├── assets/                    # Static assets
├── app.py                     # Streamlit app (legacy)
├── package.json               # Root package scripts
└── requirements.txt           # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.8+
- Git

### Installation

1. **Run the following command:**
   ```
   cd stroke_prediction
   ```

2. **Set up Python virtual environment:(optional)**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies:**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

**Start both frontend and backend with a single command:**

```bash
npm run dev
```

This will concurrently start:
- **Frontend**: React dev server at `http://localhost:5173`
- **Backend**: Flask API at `http://localhost:5000`

The frontend automatically proxies API requests to the backend.

### Alternative: Run Separately

**Frontend only:**
```bash
cd frontend
npm run dev
```

**Backend only:**
```bash
python backend/app.py
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/auth/login` | POST | User login |
| `/api/auth/register` | POST | User registration |
| `/api/predict/clinical` | POST | Clinical risk prediction |
| `/api/predict/mri` | POST | MRI stroke detection |

## 🛠️ Tech Stack

### Frontend
- **React 19** – UI framework
- **Vite** – Build tool
- **React Router** – Client-side routing
- **Tailwind CSS** – Styling
- **Framer Motion** – Animations
- **Lucide React** – Icons
- **Axios** – HTTP client

### Backend
- **Flask** – Python web framework
- **Flask-CORS** – Cross-origin support
- **TensorFlow/Keras** – Deep learning models
- **Scikit-learn** – Machine learning
- **Pillow** – Image processing
- **NumPy/Pandas** – Data manipulation

## 📦 Dependencies

### Python (requirements.txt)
```
flask
flask-cors
tensorflow
scikit-learn
pandas
numpy
pillow
joblib
scipy
```

### Node.js (package.json)
```json
{
  "dependencies": {
    "axios": "^1.13.4",
    "framer-motion": "^12.33.0",
    "lucide-react": "^0.563.0",
    "react": "^19.2.0",
    "react-dom": "^19.2.0",
    "react-router-dom": "^7.13.0"
  }
}
```

## 🧪 Model Information

| Model | Type | Purpose | Input |
|-------|------|---------|-------|
| Random Forest | Scikit-learn | Clinical risk prediction | 10 clinical features |
| CNN | TensorFlow/Keras | Stroke detection | 224×224×3 MRI images |
| U-Net | TensorFlow/Keras | Lesion segmentation | 224×224×3 MRI images |

## 📄 License

[MIT License](LICENSE)

---

<p align="center">
  Made with ❤️ for healthcare AI
</p>
