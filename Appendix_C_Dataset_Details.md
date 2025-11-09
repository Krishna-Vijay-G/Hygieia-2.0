# APPENDIX C: FULL DATASET DETAILS

This appendix documents the evaluation datasets used in the study. Only the HAM10000 dermatoscopic image dataset and the UCI Early Stage Diabetes Risk Prediction tabular dataset are employed for benchmarking the deployed pipelines. The original pre-training corpus for the Derm Foundation model is proprietary/out of scope and intentionally excluded.

---
## C.1 Overview
| Dataset | Modality | Samples | Classes | Task | Public Source | License |
|---------|----------|---------|---------|------|---------------|---------|
| HAM10000 | Dermatoscopic RGB images | 10,015 images | 7 (multi-class) | Skin lesion classification | Harvard Dataverse DOI:10.7910/DVN/DBW86T | CC BY-NC 4.0 |
| UCI Early Stage Diabetes | Structured tabular (clinical symptoms) | 520 patient records | 2 (binary) | Diabetes risk prediction | UCI ML Repository | Open (UCI terms) |

---
## C.2 HAM10000 Dermatology Dataset
**Source & Citation:** Tschandl et al., Harvard Dataverse (DOI:10.7910/DVN/DBW86T).

### C.2.1 Composition & Class Distribution
| Class (dx) | Full Name | Count | Percent |
|------------|-----------|-------|---------|
| akiec | Actinic keratoses & intraepithelial carcinoma | 327 | 3.26% |
| bcc   | Basal cell carcinoma | 514 | 5.13% |
| bkl   | Benign keratosis-like lesions | 1,099 | 10.97% |
| df    | Dermatofibroma | 115 | 1.15% |
| mel   | Melanoma | 1,113 | 11.11% |
| nv    | Melanocytic nevi | 6,705 | 66.95% |
| vasc  | Vascular lesions | 142 | 1.42% |
**Total:** 10,015 images. **Imbalance Ratio (majority/minority):** 6,705 / 115 ≈ 58.3.

### C.2.2 Metadata Fields (HAM10000_metadata.csv)
| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| lesion_id | string | Unique lesion identifier | Used for group-based splitting to avoid leakage |
| image_id | string | Image filename | Maps to JPEG in images/ directory |
| dx | categorical | Diagnosis class (akiec,bcc,bkl,df,mel,nv,vasc) | Target label |
| dx_type | categorical | Diagnostic confirmation method (histo, follow_up, consensus, etc.) | Potential confidence indicator |
| age | numeric (int) | Patient age | Missing values possible |
| sex | categorical | male / female / unknown | Class balance impacts demographic bias |
| localization | categorical | Anatomical site | Useful for stratified error analysis |

### C.2.3 Image Characteristics
- Format: JPEG, varying resolutions (approx 450×600 typical). 
- Color space: sRGB; no alpha channel.
- Preprocessing pipeline for model: (1) Load → (2) Center-crop/pad if needed → (3) Resize to 448×448 → (4) Convert to float32 & normalize pixel values to [0,1].
- Optional enhancements (not used in evaluation): contrast normalization, hair artifact removal.

### C.2.4 Splitting & Leakage Prevention
- Train/Test Split: 80/20 stratified by class on **lesion_id groups**, ensuring all images from the same lesion reside entirely in one split.
- Cross-Validation: Stratified K-fold (K=5) performed on training portion only.
- No test images or their lesion groups are exposed during feature engineering, feature selection, or calibration.

### C.2.5 Augmentation (Training Only)
| Transformation | Parameter |
|----------------|-----------|
| Rotation | ±15° |
| Horizontal Flip | Enabled |
| Brightness | ±10% (range 0.9–1.1) |
| Rescale | 1/255 (implicit normalization) |
| Vertical Flip | Disabled |
Augmentations excluded from validation/test to preserve unbiased metrics.

### C.2.6 Bias & Limitations
- **Class Imbalance:** Majority class (nv) dominates → addressed via calibration and soft-voting ensemble rather than aggressive oversampling.
- **Geographic Bias:** Dataset sourced from specific centers; may underrepresent certain skin types/ethnic backgrounds (Fitzpatrick scale distribution not provided).
- **Lesion-Level Duplication:** Multiple images per lesion; mitigated through group-based splitting.
- **Diagnostic Verification Variation:** dx_type heterogeneity could introduce noise in labels.

### C.2.7 Ethical Considerations
- Images are anonymized. No personally identifying metadata included.
- Deployment caution: Additional validation required on diverse populations to avoid diagnostic disparity.

---
## C.3 UCI Early Stage Diabetes Dataset
**Source & Citation:** Hospital dataset from Sylhet, Bangladesh (UCI ML Repository: Early Stage Diabetes Risk Prediction).

### C.3.1 Composition & Class Distribution
| Class | Count | Percent |
|-------|-------|---------|
| Positive (Outcome=1) | 268 | 51.54% |
| Negative (Outcome=0) | 252 | 48.46% |
Total Records: 520. Dataset is nearly balanced (ratio ≈ 1.06).

### C.3.2 Feature Dictionary
| Feature | Type | Description | Encoding |
|---------|------|-------------|----------|
| Age | int | Patient age (years) | Numeric |
| Gender | categorical | Male / Female | Binary label encoding |
| Polyuria | binary (Yes/No) | Excessive urination | 0/1 |
| Polydipsia | binary | Excessive thirst | 0/1 |
| sudden weight loss | binary | Unintentional rapid weight reduction | 0/1 |
| weakness | binary | General fatigue | 0/1 |
| Polyphagia | binary | Excessive hunger | 0/1 |
| Genital thrush | binary | Fungal infection | 0/1 |
| visual blurring | binary | Blurry vision symptom | 0/1 |
| Itching | binary | Persistent itching | 0/1 |
| Irritability | binary | Mood irritability | 0/1 |
| delayed healing | binary | Slow wound recovery | 0/1 |
| partial paresis | binary | Localized muscle weakness | 0/1 |
| muscle stiffness | binary | Muscular rigidity | 0/1 |
| Alopecia | binary | Hair loss | 0/1 |
| Obesity | binary | Clinical obesity marker | 0/1 |
| class (renamed Outcome) | binary target | Diabetes risk presence | 0/1 |

### C.3.3 Preprocessing
- Missing Values: None (binary symptom fields complete); zero-value physiological placeholders (e.g., none present for Age, categorical symptoms).
- Encoding: Gender mapped to {Male:1, Female:0}; Yes/No symptoms mapped to {1/0}.
- Feature Engineering (additional derived features used by model):
  - N1 = Glucose × Insulin (if available in extended variant) — (Note: base UCI file may omit these; engineered during model extension.)
  - N2 = BMI / Glucose (where surrogate BMI or proxy available).
  - N3 = Age × Polyuria (interaction capturing age-dependent symptom significance).
  - Additional interactions expanding to 24 total features (exact computed set documented in training script).

### C.3.4 Splitting & Validation
- Train/Test: Stratified 80/20 on Outcome.
- Cross-Validation: Stratified K-fold (K=5) executed on training data only.
- All scaling and feature engineering fit exclusively on training folds.

### C.3.5 Bias & Limitations
- **Geographic Scope:** Single regional hospital → limited demographic diversity.
- **Binary Symptom Encoding:** Lack of intensity granularity may oversimplify clinical presentation.
- **Temporal Factors:** Snapshot dataset lacks longitudinal progression.
- **External Validity:** Requires evaluation on other populations before clinical adoption.

### C.3.6 Ethical Considerations
- Data is de-identified; still requires careful deployment to avoid over-reliance on symptom-based screening without confirmatory tests.

---
## C.4 Data Quality & Leakage Prevention Summary
| Risk | Mitigation |
|------|------------|
| Lesion image duplicates (HAM10000) | Group split by lesion_id ensures isolation |
| Class imbalance (HAM10000) | Calibration + probabilistic ensemble weighting |
| Demographic underrepresentation | Flagged for future external validation studies |
| Train/Test contamination of preprocessing | Fit transforms only on training folds; applied to test afterward |
| Label noise (dx_type variability) | Use aggregated performance; future work: subset analysis |

---
## C.5 Ethical & Fairness Notes
- Avoid diagnostic automation without human oversight.
- Continuous monitoring of false negatives for melanoma and false positives for diabetes to minimize harm.
- Expand datasets with underrepresented skin types and age ranges.

---
## C.6 Consolidated Metrics
| Dataset | Majority Class (%) | Minority Class (%) | Majority/Minority Ratio | CV Strategy | Split Guard |
|---------|--------------------|--------------------|-------------------------|------------|-------------|
| HAM10000 | nv 66.95% | df 1.15% | 58.3 | Stratified K=5 | lesion_id grouping |
| UCI Diabetes | Positive 51.54% | Negative 48.46% | 1.06 | Stratified K=5 | Stratified outcome |

---
## C.7 Reproducibility Highlights
- Seeds: 42 (global), evaluation seeds 123/456/789.
- HAM10000 augmentations excluded from test split.
- All engineered features computed post-split to prevent leakage.
- Scope clearly limited to evaluation datasets (foundation pre-training excluded).

---
*End of Appendix C*
