# APPENDIX A: CODE IMPLEMENTATION SNIPPETS

This appendix provides essential code excerpts from the HYGIEIA dual-framework implementation.

---

## A.1 Dermatology Pipeline

### A.1.1 Derm Foundation Embedding Generation

```python
def get_derm_foundation_embedding(image_input) -> Optional[np.ndarray]:
    """Generate 6144-d embedding from 448×448 RGB image"""
    model = get_derm_foundation_model()
    input_bytes = preprocess_image_for_derm_foundation(image_input)
    input_tensor = tf.constant([input_bytes])
    
    infer = model.signatures["serving_default"]
    result = infer(inputs=input_tensor)
    embedding = result['embedding'].numpy().flatten()
    return embedding
```

### A.1.2 Feature Engineering Pipeline

```python
def engineer_enhanced_features(embedding):
    """Create 6,224 features from 6,144-d embedding"""
    features = []
    
    # 1. Original embedding (6144)
    features.extend(embedding)
    
    # 2. Statistical features (25): mean, std, percentiles, skewness, kurtosis
    features.extend([np.mean(embedding), np.std(embedding), 
                    np.percentile(embedding, [10,25,75,90]), ...])
    
    # 3. Segment features (28): 7 segments × 4 statistics
    for segment in np.array_split(embedding, 7):
        features.extend([np.mean(segment), np.std(segment),
                        np.min(segment), np.max(segment)])
    
    # 4. FFT features (15): frequency domain analysis
    fft_magnitude = np.abs(np.fft.fft(embedding))
    features.extend([np.mean(fft_magnitude[:100]), 
                    np.mean(fft_magnitude[100:500]), ...])
    
    # 5. Gradient features (12): texture analysis
    gradient = np.gradient(embedding)
    features.extend([np.mean(gradient), np.std(gradient), ...])
    
    return np.array(features)  # Total: 6,224 features
```

### A.1.3 Ensemble Classifier

```python
def create_ensemble_classifier():
    """4-algorithm soft-voting ensemble (95.9% peak accuracy)"""
    ensemble = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=25)),
            ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=10)),
            ('lr', LogisticRegression(C=0.5, max_iter=1000)),
            ('svc', CalibratedClassifierCV(SVC(probability=True), cv=3))
        ],
        voting='soft'  # Aggregate probabilities: p̂ₖ = Σᵦ wᵦ pₖᵇ
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=500)),  # Top 500 features
        ('ensemble', ensemble)
    ])
    return pipeline
```

### A.1.4 Two-Stage Calibration

```python
def calibrate_prediction(raw_probs):
    """Temperature scaling (T=1.08) + prior adjustment (α=0.15)"""
    # Stage 1: Temperature scaling
    T = 1.08
    probs = raw_probs ** (1.0 / T)
    probs /= np.sum(probs)
    
    # Stage 2: Prior adjustment (downweight majority class nv=66.9%)
    α = 0.15
    for i, cls in enumerate(classes):
        prior = CLASS_PRIORS[cls]  # nv=0.669, mel=0.111, bkl=0.110, ...
        probs[i] *= (1.0 / prior ** α)
    
    return probs / np.sum(probs)
```

---

## A.2 Diabetes Pipeline

### A.2.1 LightGBM Configuration

```python
LGBM_CONFIG = {
    'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 250,
    'max_depth': 6, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'scale_pos_weight': 1.87  # Balance 268 pos / 252 neg
}

def create_lgbm_pipeline():
    """Single LightGBM (98.1% test accuracy, AUC-ROC 1.000)"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lgbm', LGBMClassifier(**LGBM_CONFIG))
    ])
```

### A.2.2 Feature Engineering

```python
def engineer_features(df):
    """UCI dataset: 18 base → 24 engineered features"""
    # Impute zeros with medians
    df['Glucose'].replace(0, 117.0, inplace=True)
    df['BMI'].replace(0, 32.0, inplace=True)
    
    # Interaction features (e.g., Glucose × Insulin, BMI / Glucose)
    df['N1'] = df['Glucose'] * df['Insulin']
    df['N2'] = df['BMI'] / df['Glucose']
    df['N3'] = df['Age'] * df['Glucose']
    # ... 11 more interaction terms
    
    return df
```

### A.2.3 Cross-Validation

```python
def train_and_validate(X, y):
    """Stratified 5-fold CV (96.9% ± 1.2 mean accuracy)"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = create_lgbm_pipeline()
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"CV: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    model.fit(X, y)
    return model
```

---

## A.3 Algorithm Pseudocode

**Algorithm 1: Dermatology Routine**
```
Input:  448×448 RGB image
Output: Calibrated 7-class probabilities

1. Generate 6,144-d embedding (Derm Foundation)
2. Engineer 6,224 features (statistical, FFT, gradient, texture)
3. SelectKBest → top 500 features; standardize
4. Ensemble soft-voting (RF, GB, LR, SVC) → raw probabilities
5. Temperature scaling (T=1.08) + prior adjustment (α=0.15)
6. Return calibrated probabilities
```

**Algorithm 2: Diabetes Routine**
```
Input:  Age, Gender, 16 binary symptoms
Output: Binary probability

1. Label-encode categorical features
2. Impute zeros; engineer interaction features
3. Standardize features
4. LightGBM prediction → probability
5. Return result
```

---

## A.4 Performance Summary

| **Pipeline**      | **Dataset**     | **Samples** | **Accuracy**    | **Latency** |
|-------------------|-----------------|-------------|-----------------|-------------|
| Dermatology (v4.0)| HAM10000        | 8,039       | 93.9% ± 2.1     | ~98 ms      |
| Diabetes (LightGBM)| UCI Early Stage| 520         | 98.1% (AUC 1.0) | ~0.06 ms    |

**Key Features:**
- **Dermatology:** Ensemble (RF, GB, LR, SVC), SelectKBest(500), calibration (T=1.08, α=0.15)
- **Diabetes:** LightGBM, 24 engineered features, top predictors (Polyuria, Polydipsia, Sudden weight loss)

---

## A.5 Reproducibility

- **Seeds:** `random_state=42` (training), seeds 123/456/789 (multi-seed validation)
- **Splits:** Stratified 80/20 train-test; dermatology split by `lesion_id`
- **Augmentations:** ±15° rotation, horizontal flip, ±10% brightness (training only)
- **CV:** Stratified K-fold (K=5) with shuffle
- **Software:** Python 3.8+, TensorFlow 2.x, scikit-learn 1.0+, LightGBM 3.3+

---

*End of Appendix A*
