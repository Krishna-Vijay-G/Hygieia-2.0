# APPENDIX B: FULL ENVIRONMENT SETUP (1 PAGE)

Essential details to reproduce the environment for both pipelines with minimal steps.

## B.1 Hardware & OS
- OS: Windows 10/11 (tested), Linux/macOS supported
- CPU/RAM: 4 cores, 16 GB RAM recommended (min. 2 cores, 8 GB)
- Storage: ~10 GB free (models + datasets)
- GPU: Optional; TensorFlow falls back to CPU

## B.2 Software (Core)
- Python 3.8–3.10
- Packages: TensorFlow 2.10+ (derm), scikit-learn 1.0+, LightGBM 3.3+ (diab), NumPy 1.20+, Pandas 1.3+, Pillow 8.0+, OpenCV 4.5+, joblib, tqdm

## B.3 Install (Windows PowerShell)
Option A — Conda (recommended)
```powershell
conda create -n hygieia python=3.9 -y; conda activate hygieia
conda install -c conda-forge tensorflow=2.10 scikit-learn=1.1 numpy=1.23 pandas=1.5 -y
conda install -c conda-forge lightgbm=3.3 -y
pip install pillow opencv-python joblib tqdm matplotlib seaborn scipy
```
Option B — venv + pip
```powershell
python -m venv .\.venv; .\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install tensorflow==2.10.0 scikit-learn==1.1.3 lightgbm==3.3.5 numpy==1.23.5 pandas==1.5.3 pillow==9.5.0 opencv-python==4.7.0.72 joblib==1.2.0 tqdm==4.65.0
```

## B.4 Project layout (key paths)
- Derm upgrade/: `derm_model.py`, `derm_calibration.py`, `derm_benchmarker.py`
- Derm upgrade/Dermatology_Model/: `derm_model.joblib`, `derm_scaler.joblib`, `derm_feature.joblib`, `saved_model.pb` (+ variables/)
- Derm upgrade/HAM10000/: `HAM10000_metadata.csv`, `images/`
- Diab upgrade/: `diab_model.py`, `diab_uci_benchmarker.py`
- Diab upgrade/Diabetes_Model/: `diab_model.joblib`, `diab_scaler.joblib`, `diabetes.csv`, `test_set_held_out.csv`

## B.5 Datasets (acquire & place)
- HAM10000 (Harvard Dataverse): extract images → `Derm upgrade/HAM10000/images/`; place `HAM10000_metadata.csv` in `Derm upgrade/HAM10000/`.
- UCI Early Stage Diabetes: download CSV → rename to `diabetes.csv` → `Diab upgrade/Diabetes_Model/`.

## B.6 Training & Reproducibility (summary)
- Dermatology: 6,144-d embedding → 6,224 engineered feats; SelectKBest k=500; soft-vote ensemble:
  - RF (n_estimators=300, max_depth=25), GB (n_estimators=200, max_depth=10), LR (C=0.5, max_iter=1000), Calibrated SVC (cv=3)
  - Calibration: temperature T=1.08, prior adjust α=0.15
- Diabetes: LightGBM — num_leaves=31, learning_rate=0.05, n_estimators=250, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=1.87
- Splits/Seeds: Stratified 80/20; CV=5 folds (shuffle, random_state=42); validation seeds: 123/456/789

## B.7 Quick checks
Environment
```powershell
python - << 'PY'
import sys, importlib
req = [
 ("tensorflow","2.10"), ("sklearn","1.0"), ("lightgbm","3.3"),
 ("numpy","1.20"), ("pandas","1.3"), ("PIL","8.0"), ("cv2","4.5")
]
print("Python:", sys.version.split()[0])
for m,v in req:
    try:
        mod = importlib.import_module(m if m!="PIL" else "PIL.Image"); print("✓", m)
    except Exception as e:
        print("✗", m, e)
PY
```
Models present
```powershell
if (Test-Path "Derm upgrade/Dermatology_Model/derm_model.joblib") {"Derm ✓"} else {"Derm ✗"}
if (Test-Path "Diab upgrade/Diabetes_Model/diab_model.joblib") {"Diab ✓"} else {"Diab ✗"}
```

— End —
