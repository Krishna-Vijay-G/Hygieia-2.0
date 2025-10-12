# Dermatology Model Complete Report

## Model Overview
- **Model:** Derm Foundation + Optimized Classifier
- **Embedding:** 6144-dimensional features from HAM10000 images
- **Classifier:** Trained ensemble/hybrid model (see training details)
- **Accuracy:** Reported via benchmark scripts (see below)

## Benchmark Results
- **Scripts Used:**
  - `test_7_per_class_benchmark.py` (7 images per class)
  - `benchmark_dermatology_model.py` (configurable samples per class)
- **Metrics:**
  - Overall accuracy
  - Per-class precision, recall, F1-score
  - Confusion matrix
  - Confidence analysis

## Example Output
```
Overall Accuracy: 0.78

Per-Class Performance:
Condition           Precision    Recall       F1-Score     Support  
akiec (Actinic K...) 0.80         0.71         0.75         7
bcc (Basal Cell ...) 0.85         0.85         0.85         7
...etc

Confusion Matrix:
       akiec  bcc  bkl  df  nv  vasc  mel
akiec    5     0    1   0   1    0     0
bcc      0     6    0   0   1    0     0
...etc
```

## Files Used
- `saved_model.pb`, `variables/` (Derm Foundation model)
- `new_optimized_classifier.joblib` (classifier)
- `dermatology_model.py` (main logic)
- `test_7_per_class_benchmark.py`, `benchmark_dermatology_model.py` (evaluation)

## Notes
- Model is validated on balanced samples from HAM10000.
- For retraining, see `train_classifier.py` and `features_700.npz`.
