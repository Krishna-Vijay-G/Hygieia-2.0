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

- Size/labels: 10,015 RGB images; 7 classes (akiec, bcc, bkl, df, mel, nv, vasc)
- Class balance (approx.): nv 66.95%, mel 11.11%, bkl 10.97%, bcc 5.13%, akiec 3.26%, vasc 1.42%, df 1.15% (majority/minority ≈ 58.3)
- Metadata fields: lesion_id, image_id, dx, dx_type, age, sex, localization (see dataset card)
- Preprocess: resize to 448×448; normalize to [0,1]
- Split: stratified 80/20 with lesion_id grouping (prevents leakage); CV: stratified K=5 on train only
- Augment (train only): rotation ±15°, horizontal flip, brightness ±10%
- Risks/ethics: severe class imbalance; geographic/skin-type bias possible; labels vary by dx_type; images de-identified; external validation advised

---

## C.3 UCI Early Stage Diabetes (Tabular)

- Size/labels: 520 records; Outcome ∈ {0,1}; nearly balanced (268/252)
- Features: 18 total — Age, Gender, 16 binary symptoms (Yes/No → 1/0); Gender encoded {Male:1, Female:0}
- Split/CV: stratified 80/20 on Outcome; stratified K=5 on train; transforms fitted on training only
- Risks/ethics: single-center cohort limits external validity; symptoms lack intensity/temporal detail; de-identified data; do not replace clinical tests

---

## C.4 Quick reference
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

## C.5 Reproducibility
- Seeds: global 42; evaluation seeds 123/456/789
- Fit all preprocessing/feature steps on training folds; apply to held-out only
- Paths: HAM10000 → `Derm upgrade/HAM10000/{HAM10000_metadata.csv, images/}`; UCI → `Diab upgrade/Diabetes_Model/diabetes.csv`
- Scope: only evaluation datasets; Derm Foundation pre-training excluded

---
