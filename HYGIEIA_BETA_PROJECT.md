# Hygieia Beta - AI-Powered Medical Diagnostic Platform

![Hygieia Logo](static/hygieia_color.svg)

## Project Overview

Hygieia Beta is an advanced medical diagnostic platform that utilizes artificial intelligence to provide early detection and risk assessment for various medical conditions. The current beta version focuses on two primary diagnostic capabilities:

1. **Dermatology Analysis**: AI-powered skin condition detection with 89.8% validation accuracy
2. **Diabetes Risk Assessment**: Advanced diabetes risk prediction with 88.4% accuracy

The platform provides an intuitive web interface for users to upload skin images or input health parameters, process this data through specialized ML models, and receive instant results with confidence scores, risk assessments, and medical recommendations.

## Technologies Used

### Backend
- **Python 3.8+**: Core programming language
- **Flask**: Web framework for the application
- **TensorFlow**: Deep learning framework for image analysis
- **Scikit-learn**: Machine learning for classification models
- **NumPy/Pandas**: Data manipulation and processing
- **Joblib**: Model serialization and persistence
- **OpenCV/PIL**: Image processing and manipulation
- **UUID**: Unique identifier generation for secure file handling

### Frontend
- **HTML5/CSS3**: Structure and styling
- **JavaScript**: Interactive elements
- **Font Awesome**: Icon library
- **Custom CSS**: Responsive design with gradients and animations

### ML Models
- **Google Derm Foundation v1.0**: Base model for dermatological analysis
- **LightGBM**: Primary model for diabetes prediction
- **Ensemble Classifiers**: Advanced voting algorithms for skin condition classification
- **Feature Engineering**: Custom engineered features for improved accuracy

## Project Structure

```
hygieia-beta/
├── app.py                      # Main Flask application
├── dermatology_model.py        # Dermatology analysis model
├── diabetes_model.py           # Diabetes risk assessment model
├── ml_models.py                # Common ML functionality
├── start_hygieia_beta.bat      # Windows startup script
├── static/                     # Static assets
│   ├── hygieia_color.svg       # Logo
│   ├── css/
│   │   └── style.css          # Main stylesheet
│   └── js/
│       └── main.js            # JavaScript functionality
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Home page
│   ├── dermatology.html       # Skin analysis page
│   ├── diabetes.html          # Diabetes assessment page
│   └── results.html           # Results display page
├── models/                     # ML model files
│   ├── Diabetes_Model/         # Diabetes model files
│   │   ├── diabetes_feature_info.joblib
│   │   ├── DIABETES_MODEL_FINAL_REPORT.md
│   │   ├── diabetes_model_lgbm_recreated.joblib
│   │   ├── diabetes_scaler.joblib
│   │   └── diabetes.csv
│   └── Skin_Disease_Model/     # Dermatology model files
│       ├── DERMATOLOGY_MODEL_COMPLETE_REPORT.md
│       ├── fingerprint.pb
│       ├── optimized_dermatology_model.joblib
│       ├── README.md
│       ├── saved_model.pb
│       ├── scin_dataset_precomputed_embeddings.npz
│       ├── test_derm_foundation_accuracy.py
│       └── variables/
└── uploads/                    # Uploaded images storage
    └── [user-uploaded files]
```

## Model Information

### 1. Dermatology Model

#### Architecture
- **Base Model**: Google's Derm Foundation v1.0 (BiT-M ResNet101x3)
- **Architecture Type**: Hybrid TensorFlow + Scikit-learn ensemble
- **Input**: 448×448 RGB images
- **Output**: Classification among 7 skin conditions with confidence scores

#### Performance
- **Overall Validation Accuracy**: 89.8% (44/49 correct)
- **Training Accuracy**: 72.4%
- **Perfect Classifications**: AKIEC (100%), VASC (100%)
- **Excellent Performance**: 5 classes at 85.7%+

#### Skin Conditions Detected
| Code | Full Name | Description | Risk Level |
|------|-----------|-------------|------------|
| **AKIEC** | Actinic Keratoses | Pre-cancerous skin lesions | Moderate |
| **BCC** | Basal Cell Carcinoma | Most common skin cancer | High |
| **BKL** | Benign Keratosis | Non-cancerous skin growths | Low |
| **DF** | Dermatofibroma | Benign skin nodules | Low |
| **MEL** | Melanoma | Dangerous skin cancer | High |
| **NV** | Melanocytic Nevi | Common moles | Low |
| **VASC** | Vascular Lesions | Blood vessel abnormalities | Low |

#### Technical Innovations
1. **Hybrid Architecture**: Combines TensorFlow foundation model with specialized ensemble classifiers
2. **Advanced Feature Engineering**: 6224-dimensional feature vectors derived from 6144-dimensional medical embeddings
3. **Optimized Ensemble**: Voting classifier combining Random Forest, Gradient Boosting, Logistic Regression, and SVM
4. **Risk Assessment**: Automatic risk categorization with confidence scores

### 2. Diabetes Model

#### Architecture
- **Primary Model**: LightGBM Gradient Boosting Classifier
- **Fallback**: Voting Classifier (LightGBM + KNN)
- **Input Features**: 8 medical parameters + 16 engineered features
- **Output**: Risk prediction with probability, confidence, and risk factors

#### Performance
- **Accuracy**: 88.41% ±1.60%
- **Precision**: 83.11% ±4.63%
- **Recall**: 84.35% ±3.13%
- **F1-Score**: 83.59% ±1.88%
- **ROC-AUC**: 94.55% ±1.13%

#### Input Parameters
- **Pregnancies**: Number of times pregnant (0-17)
- **Glucose**: Plasma glucose concentration (0-199 mg/dL)
- **BloodPressure**: Diastolic blood pressure (0-122 mm Hg)
- **SkinThickness**: Triceps skin fold thickness (0-99 mm)
- **Insulin**: 2-Hour serum insulin (0-846 mu U/ml)
- **BMI**: Body mass index (0-67.1 kg/m²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function (0.078-2.42)
- **Age**: Age in years (21-81)

#### Technical Innovations
1. **Advanced Feature Engineering**: 16 engineered features from 8 base features
2. **Robust Risk Assessment Framework**: Low/Medium/High risk categorization
3. **Confidence Calibration**: Enhanced confidence scoring based on probability distributions
4. **Risk Factor Identification**: Automatic detection of key contributing factors

## Workflow

### Dermatology Analysis Workflow
1. **Image Upload**: User uploads a skin condition image
2. **Image Processing**: System preprocesses the image to 448×448 RGB format
3. **Foundation Model**: Google Derm Foundation extracts 6144-dimensional medical embeddings
4. **Feature Engineering**: System generates 6224-dimensional enhanced features
5. **Ensemble Classification**: Optimized ensemble classifier predicts condition
6. **Risk Assessment**: System determines risk level and confidence
7. **Result Display**: User receives classification with confidence scores and recommendations

### Diabetes Risk Assessment Workflow
1. **Data Input**: User enters medical parameters (glucose, BMI, etc.)
2. **Data Validation**: System validates input ranges and formats
3. **Feature Engineering**: System creates 16 additional engineered features
4. **Model Prediction**: LightGBM model predicts diabetes risk
5. **Risk Analysis**: System determines risk level and identifies risk factors
6. **Result Display**: User receives risk assessment with confidence scores and recommendations

## Real-World Applications

### Healthcare Settings
1. **Primary Care Triage**: Assist healthcare providers in prioritizing patient cases
2. **Remote Healthcare**: Enable preliminary screening in underserved areas
3. **Medical Education**: Train healthcare students in recognizing conditions
4. **Clinical Decision Support**: Provide additional information for healthcare professionals
5. **Preventive Care**: Encourage early detection and regular monitoring

### Public Health
1. **Community Screening Programs**: Facilitate large-scale health screenings
2. **Health Education**: Increase awareness about skin conditions and diabetes risk
3. **Remote Populations**: Serve areas with limited access to specialists
4. **Emergency Response**: Support rapid assessment during health crises
5. **Health Monitoring**: Enable regular self-monitoring for early detection

### Research Applications
1. **Epidemiological Studies**: Support data collection for population health studies
2. **AI Model Development**: Serve as a foundation for more specialized medical AI
3. **Medical Protocol Evaluation**: Test effectiveness of screening protocols
4. **Health System Integration**: Explore integration with electronic health records
5. **Medical Device Development**: Inform development of specialized diagnostic devices

## Future Improvements

### Technical Enhancements
1. **Multi-modal Integration**: Combine image, text, and structured data inputs
2. **Explainable AI Features**: Provide visual explanations for predictions
3. **Real-time Performance**: Optimize for faster prediction speed
4. **Mobile Optimization**: Adapt for low-resource mobile devices
5. **Expanded Model Coverage**: Add more health conditions and risks

### User Experience
1. **Multilingual Support**: Add support for multiple languages
2. **Accessibility Features**: Enhance accessibility for users with disabilities
3. **Progressive Web App**: Enable offline functionality
4. **Personalized Reports**: Generate detailed PDF reports
5. **History Tracking**: Allow users to track changes over time

### Medical Capabilities
1. **Additional Conditions**: Expand to other skin conditions and health risks
2. **Integration with EHR**: Connect with electronic health record systems
3. **Telemedicine Features**: Add video consultation capabilities
4. **Personalized Recommendations**: Provide tailored health advice
5. **Medication Guidance**: Offer information about relevant treatments

## Installation & Usage

### Prerequisites
- Python 3.8+
- pip package manager
- Web browser

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hygieia-beta.git
   cd hygieia-beta
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Start the application:
   ```bash
   # On Windows
   start_hygieia_beta.bat
   # OR on any platform
   python app.py
   ```

2. Open your web browser and visit: `http://localhost:5001`

3. Navigate to either the Skin Analysis or Diabetes Risk Assessment section

4. Follow the on-screen instructions to upload an image or input health parameters

5. View your results with confidence scores and recommendations

## Important Medical Disclaimer

This AI-powered health analysis tool is for **educational and informational purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical concerns. Early detection saves lives, but professional medical evaluation is essential.

## Credits

- **Development**: Arkhins
- **Base Dermatology Model**: Google Derm Foundation v1.0
- **Diabetes Dataset**: Pima Indians Diabetes Dataset
- **Testing Dataset**: HAM10000

## License

All rights reserved. This codebase is proprietary and is not licensed for distribution or modification without explicit permission.

---

© Hygieia Beta Project, 2025.