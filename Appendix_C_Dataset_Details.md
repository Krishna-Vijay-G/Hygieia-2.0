# APPENDIX C: FULL DATASET DETAILS

This appendix documents the evaluation datasets used in the study. Only the HAM10000 dermatoscopic image dataset and the UCI Early Stage Diabetes Risk Prediction tabular dataset are employed for benchmarking the deployed pipelines. The original pre-training corpus for the Derm Foundation model is proprietary/out of scope and intentionally excluded.

---

## C.1 Scope and Sources
- HAM10000: Dermatoscopic image dataset for multi-class skin lesion classification (7 classes), 10,015 images.
  - Source: Harvard Dataverse (DOI:10.7910/DVN/DBW86T). License: CC BY-NC 4.0.
- UCI Early Stage Diabetes Risk Prediction: Tabular clinical-symptom dataset, 520 records, binary target.
  - Source: UCI Machine Learning Repository. License: UCI terms (research/education use).

---

## C.2 HAM10000 (Dermatology)

### C.2.1 Snapshot
- Samples: 10,015 RGB dermatoscopic images (JPEG)
- Classes (dx): akiec, bcc, bkl, df, mel, nv, vasc (7-class classification)
- Class distribution (approx.): nv 6705 (66.95%), mel 1113 (11.11%), bkl 1099 (10.97%), bcc 514 (5.13%), akiec 327 (3.26%), vasc 142 (1.42%), df 115 (1.15%)
- Majority/minority ratio ≈ 58.3 (nv/df)

### C.2.2 Metadata schema (HAM10000_metadata.csv)
| Field | Type | Description |
|------|------|-------------|
| lesion_id | string | Unique lesion identifier (used for group-wise splits) |
| image_id | string | Image filename (maps to images/ directory) |
| dx | categorical | Target label (akiec, bcc, bkl, df, mel, nv, vasc) |
| dx_type | categorical | Diagnostic method (histo, follow_up, consensus, etc.) |
| age | int | Patient age (may include missing) |
| sex | categorical | male / female / unknown |
| localization | categorical | Anatomical site |

### C.2.3 Preprocessing and splits
- Preprocess: load → center-crop/pad as needed → resize to 448×448 → float32 normalize to [0,1].
- Split strategy: 80/20 train–test with stratification by dx and strict grouping by lesion_id to prevent leakage across splits.
- Cross-validation: Stratified K=5 on the training portion only; all transforms fitted only on training folds.

### C.2.4 Training-time augmentation (not applied to validation/test)
- Rotation ±15°, horizontal flip enabled, brightness ±10%; vertical flip disabled.

### C.2.5 Quality, leakage, and risks
- Leakage guard: lesion_id-based group split; no test images seen during feature selection/calibration.
- Imbalance: strong majority class (nv); mitigated downstream via calibration and ensemble probability handling.
- Potential biases: limited geographic/skin-type diversity; label-confidence heterogeneity via dx_type.
- Ethics: images are de-identified; additional external validation recommended before clinical use.

---

## C.3 UCI Early Stage Diabetes (Tabular)

### C.3.1 Snapshot
- Samples: 520 patient records; nearly balanced (Outcome=1: 268, Outcome=0: 252)
- Features: 18 total — Age, Gender, and 16 binary symptoms (Yes/No)
- Target: Outcome ∈ {0,1}

### C.3.2 Feature dictionary (concise)
| Feature | Type | Notes |
|---------|------|-------|
| Age | int | Age in years |
| Gender | categorical | Male/Female (binary encoded) |
| Polyuria, Polydipsia, sudden weight loss, weakness, Polyphagia, Genital thrush, visual blurring, Itching, Irritability, delayed healing, partial paresis, muscle stiffness, Alopecia, Obesity | binary | Yes/No → 1/0 |
| class (renamed Outcome) | binary target | Positive=1, Negative=0 |

### C.3.3 Preprocessing and validation
- Encoding: Gender → {Male:1, Female:0}; symptoms Yes/No → {1/0}.
- Split strategy: 80/20 stratified train–test on Outcome; transforms fitted on training only.
- Cross-validation: Stratified K=5 on training data; mean ± std reported across folds.

### C.3.4 Risks and ethics
- Geographic scope: single-region cohort; external validity must be established.
- Symptom-only inputs lack intensity/temporal granularity; should not replace clinical testing.
- Data are de-identified; maintain appropriate safeguards in deployment.

---

## C.4 Consolidated quick reference
| Item | HAM10000 | UCI Early Stage Diabetes |
|------|----------|--------------------------|
| Modality | Images (RGB JPEG) | Tabular (symptoms) |
| Size | 10,015 images | 520 records |
| Labels | 7 classes (dx) | Outcome ∈ {0,1} |
| Split | Stratified 80/20 + lesion_id grouping | Stratified 80/20 |
| CV | Stratified K=5 (train only) | Stratified K=5 (train only) |
| Leakage guard | Group split by lesion_id; fit transforms on train | Fit transforms on train only |
| License | CC BY-NC 4.0 | UCI terms |
| Source | Dataverse DOI:10.7910/DVN/DBW86T | UCI ML Repository |

---

## C.5 Reproducibility notes
- Seeds: global 42; additional evaluation seeds 123/456/789.
- All preprocessing/feature steps are fitted on training folds and applied to held-out data.
- Dataset placement:
  - HAM10000: `Derm upgrade/HAM10000/{HAM10000_metadata.csv, images/}`
  - UCI Diabetes: `Diab upgrade/Diabetes_Model/diabetes.csv`
- Scope reminder: Only evaluation datasets are listed; Derm Foundation pre-training data are excluded by design.

---

*End of Appendix C*


