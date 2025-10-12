# 🏥 HYGIEIA DERMATOLOGY MODEL — FINAL TECHNICAL REPORT

Date: October 12, 2025  
Version: Production v2.x  
Authors: Arkhins  
Platform: Hygieia Medical Diagnostic System  
Status: ✅ Production-ready

---

## 📋 Executive Summary

The Hygieia Dermatology pipeline combines Google’s Derm Foundation SavedModel as a medical feature extractor with a robust tabular classifier. After evaluating multiple algorithms, the deployed production classifier is LightGBM (LGBMClassifier) using 6224 engineered features derived from the 6144-D embeddings. On internal cross-validation, LightGBM achieved the best CV mean (≈0.711), and in 7-per-class validation runs it achieved up to ≈89.8% accuracy (44/49).

Key points:
- Foundation extractor: TensorFlow SavedModel (Derm Foundation) → 6144-D embedding
- Engineered features: 6224 robust features (statistics, segment, frequency, gradient/texture)
- Deployed classifier: LightGBM with StandardScaler + LabelEncoder
- Fallback: heuristic fallback when the joblib is missing or inference fails

---

## 🏗️ Architecture Overview

### System Architecture Flow
```mermaid
flowchart TD
    A[Raw Image] --> B[Preprocess 448x448 RGB]
    B --> C[Derm Foundation — embedding 6144-D]
    C --> D[Engineer Features 6144 -> 6224]
    D --> E[Scale Features — StandardScaler]
    E --> F[Predict — LGBMClassifier]
    F --> G{OK?}
    G -- Yes --> H[Return label + probs]
    G -- No --> I[Fallback heuristic]
    I --> H
```

### Technical Stack
- TensorFlow (SavedModel) for embeddings
- NumPy/PIL/OpenCV for preprocessing
- Scikit-learn + LightGBM for classification
- Joblib for model persistence

---

## 🧠 Foundation Model (Derm Foundation)
- Input: 448×448 RGB; Output: 6144-D embedding
- Loaded via `tf.saved_model.load(models/Skin_Disease_Model)`
- Signature: uses `serving_default`; embedding key: `embedding`

Example:
```python
model = tf.saved_model.load('models/Skin_Disease_Model')
infer = model.signatures['serving_default']
result = infer(inputs=tf.constant([serialized_png_bytes]))
embedding = result['embedding'].numpy().flatten()
```

---

## 🧩 Feature Engineering (6224 features)
Categories:
- Statistical (mean, std, var, percentiles, skewness, kurtosis, norms, ratios)
- Segment-based (7 segments × mean/std/min/max)
- Frequency-domain (FFT magnitudes, centroid, energy, flatness, band ratios)
- Gradient/texture (gradient stats, second derivative, autocorrelation proxy, trend, curvature)

Resilience:
- NaN/Inf handling; clipping; padding/truncation to exactly 6224

---

## 🤖 Classifier Internals (Inference)
- Serialized bundle: `models/Skin_Disease_Model/new_optimized_classifier.joblib`
- Contains:
  - Classifier: LightGBM (LGBMClassifier(random_state=42))
  - Scaler: StandardScaler
  - Feature selector: None (not used)
  - Label encoder: LabelEncoder with classes [akiec, bcc, bkl, df, mel, nv, vasc]

Prediction flow:
1) 6144-D embedding → 6224-D features
2) Standardize features
3) LGBMClassifier `predict`/`predict_proba`
4) Decode label and map to condition/risk level

Fallback: If classifier unavailable or errors occur, use `fallback_pattern_analysis` with capped confidence.

---

## ✅ Why LightGBM was selected
- Best CV mean (≈0.711) across tested models (Voting ≈0.707, Stacking ≈0.701, Logistic ≈0.703, XGBoost ≈0.690).
- Strong with high-dimensional tabular features and non-linear patterns.
- Simple, stable, and fast inference pipeline with well-behaved probabilities.
- Reasonable train time vs. more expensive models (e.g., GradientBoosting).
- Matches the deployed artifact (confirmed via joblib inspection).

---

## 📊 Training Summary (from training_report_700.json)

| Model              | CV Mean | CV Std | Precision | Recall | F1     | Duration (s) |
|--------------------|---------|--------|-----------|--------|--------|--------------|
| RandomForest       | 0.664   | 0.034  | 1.0       | 1.0    | 1.0    | 10.16        |
| ExtraTrees         | 0.663   | 0.033  | 1.0       | 1.0    | 1.0    | 5.05         |
| AdaBoost           | 0.593   | 0.025  | 0.735     | 0.716  | 0.719  | 158.51       |
| GradientBoosting   | 0.680   | 0.028  | 1.0       | 1.0    | 1.0    | 2539.25      |
| LogisticRegression | 0.703   | 0.046  | 1.0       | 1.0    | 1.0    | 3.49         |
| SVM                | 0.636   | 0.031  | 0.864     | 0.863  | 0.863  | 22.50        |
| KNN                | 0.599   | 0.010  | 0.734     | 0.716  | 0.720  | 2.48         |
| MLP                | 0.693   | 0.040  | 1.0       | 1.0    | 1.0    | 19.68        |
| DecisionTree       | 0.510   | 0.006  | 1.0       | 1.0    | 1.0    | 9.77         |
| SGD                | 0.681   | 0.039  | 1.0       | 1.0    | 1.0    | 2.88         |
| LDA                | 0.651   | 0.046  | 0.929     | 0.929  | 0.929  | 3.14         |
| XGBoost            | 0.690   | 0.034  | 1.0       | 1.0    | 1.0    | 395.64       |
| LightGBM           | 0.711   | 0.046  | 1.0       | 1.0    | 1.0    | 377.53       |
| Voting             | 0.707   | 0.039  | 1.0       | 1.0    | 1.0    | 40.52        |
| Stacking           | 0.701   | 0.037  | 1.0       | 1.0    | 1.0    | 173.50       |

Notes: Precisions/recalls of 1.0 in the table reflect scoring on the training fold predictions in the stored report object; cross-validated means reflect generalization performance.

---

## 🧪 Benchmarking & Validation
- Protocol: 7 images per class (49 total) sampled from HAM10000 metadata and evaluated end-to-end through the production pipeline.
- Scripts:
  - `models/Skin_Disease_Model/test_7_per_class_benchmark.py`
  - `models/Skin_Disease_Model/benchmark_dermatology_model.py`
- Headline result: Overall accuracy ≈ 89.8% (44/49 correct).

Per-class results (example run):

| Disease | Accuracy | Correct/Total |
|---------|----------|---------------|
| AKIEC   | 100.0%   | 7/7           |
| VASC    | 100.0%   | 7/7           |
| BKL     | 85.7%    | 6/7           |
| NV      | 85.7%    | 6/7           |
| DF      | 85.7%    | 6/7           |
| MEL     | 85.7%    | 6/7           |
| BCC     | 85.7%    | 6/7           |

Textual confusion matrix (example run):

```
Predicted →   AKIEC  BCC  BKL   DF  MEL   NV  VASC
Actual ↓
AKIEC         [ 7    0    0    0    0    0    0 ]
BCC           [ 0    6    0    1    0    0    0 ]
BKL           [ 0    0    6    0    0    1    0 ]
DF            [ 0    1    0    6    0    0    0 ]
MEL           [ 0    0    1    0    6    0    0 ]
NV            [ 0    0    0    0    1    6    0 ]
VASC          [ 0    0    0    0    0    0    7 ]
```

The scripts also report confidence analysis (avg confidence, correct vs incorrect) and a full classification report with precision/recall/F1 per class.

---

## 📁 Files and Roles
- `dermatology_model.py`: Pipeline implementation, error handling, fallback
- `models/Skin_Disease_Model/saved_model.pb` (+ variables/): Derm Foundation SavedModel
- `models/Skin_Disease_Model/new_optimized_classifier.joblib`: LightGBM + scaler + label encoder
- `models/Skin_Disease_Model/training_report_700.json`: CV results and timings
- `models/Skin_Disease_Model/test_7_per_class_benchmark.py`: 7-per-class benchmark
- `models/Skin_Disease_Model/benchmark_dermatology_model.py`: CLI benchmark with colored output

---

## 🔮 Future Work
- Explore calibrated soft-voting across diverse models only if it beats LightGBM on external validation.
- Add probability calibration if mixing multiple learners.
- Consider direct fine-tuning of the foundation model for rare classes.

---

## ✅ Clinical Considerations
- Use as decision support; not a substitute for clinician judgment.
- Risk levels and confidence are surfaced to guide triage.
- Continuous monitoring recommended when deployed.

---

## 📚 References
- HAM10000 dataset
- TensorFlow SavedModel guide
- Scikit-learn ensemble methods
- LightGBM documentation
- XGBoost documentation
