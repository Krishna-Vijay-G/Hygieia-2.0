# Dermatology Model Improvement History

## v1.0 - Initial Model
- Basic classifier trained on HAM10000 features
- Used simple feature engineering
- Accuracy: ~65%

## v2.0 - Derm Foundation Integration
- Added 6144-dimensional embeddings from Derm Foundation model
- Improved feature engineering pipeline
- Accuracy: ~72%

## v3.0 - Optimized Classifier
- Ensemble/hybrid models (RandomForest, SVM, Voting, Stacking)
- Cross-validation and robust metrics
- Saved as `new_optimized_classifier.joblib`
- Accuracy: ~78% (benchmark)

## v3.1 - Robust Benchmarking & Reporting
- Added scripts for balanced sampling and full pipeline tests
- Detailed logging, confidence analysis, confusion matrix
- Markdown reporting and workflow documentation

## Future Plans
- Add more clinical datasets
- Improve explainability and risk stratification
- Integrate with web UI and API endpoints
