# Dermatology Model Workflow & File Map

## Workflow Steps
1. **Image Selection**
   - Source: `HAM10000/images/`
   - Metadata: `HAM10000_metadata.csv`
2. **Feature Extraction**
   - Model: `saved_model.pb` + `variables/`
   - Code: `dermatology_model.py` (`get_derm_foundation_embedding`)
3. **Feature Engineering**
   - Code: `dermatology_model.py` (`engineer_enhanced_features`)
4. **Prediction**
   - Classifier: `new_optimized_classifier.joblib`
   - Code: `dermatology_model.py` (`predict_using_optimized_classifier`, `predict_image`)
5. **Benchmarking & Evaluation**
   - Scripts: `test_7_per_class_benchmark.py`, `benchmark_dermatology_model.py`
   - Metrics: accuracy, per-class, confusion matrix

## File Map
- `saved_model.pb`, `variables/` — Derm Foundation model
- `new_optimized_classifier.joblib` — Trained classifier
- `dermatology_model.py` — Main model logic
- `test_7_per_class_benchmark.py` — Benchmark script
- `benchmark_dermatology_model.py` — Advanced benchmark script
- `train_classifier.py` — Training script (uses `features_700.npz`)
- `features_700.npz` — Training features (not needed for inference)
- `HAM10000_metadata.csv` — Image metadata
- `HAM10000/images/` — Image data

## Example Usage
```python
from dermatology_model import predict_image
result = predict_image('HAM10000/images/ISIC_0026130.jpg')
print(result)
```

## Notes
- For retraining, use `train_classifier.py` and `features_700.npz`.
- For evaluation, use the benchmark scripts above.
