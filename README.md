# SOIL-RILAB

Soil organic carbon (OC) prediction using multi-source hyperspectral and satellite imagery. This project integrates VTT hyperspectral imaging, Kuva Space satellite data, and Sentinel-2 multispectral imagery with machine learning models to map soil OC across agricultural fields in the Flanders region of Belgium.

---

## Overview

The pipeline covers three complementary workflows:

1. **VTT HSI Analysis** — Radiometric calibration of ground/airborne hyperspectral images followed by CNN, Random Forest, and XGBoost model training and full-image OC prediction.
2. **Data Fusion (Kuva + Sentinel-2 + VTT)** — Combines three sensor sources into four feature configurations, trains RF and XGBoost models, performs sensitivity analysis, and generates fused OC prediction maps.
3. **Google Earth Engine ETL** — Extracts Sentinel-2 time-series spectra and LUCAS soil properties for Flanders sample points and exports TFRecord files for cloud-based model training.

---

## Directory Structure

```
SOIL-RILAB/
├── Googe_earth_engin_extract_sentinel2_data.py   # GEE Sentinel-2 extraction pipeline
├── VTT HSI sensor/
│   ├── VTT_analysis_2026_train.py                # Train CNN / RF / XGBoost on VTT data
│   ├── VTT_analysis_2026_predict.py              # Generate OC prediction maps (VTT)
│   └── python_functions.py                        # Calibration, feature engineering, model utilities
├── data fusion soillab/
│   ├── datafusion_train_kuva_sentinel.py         # Train multi-source fusion models
│   ├── datafusion_predict.py                     # Generate fusion prediction maps
│   └── python_functions.py                        # Shared utilities for data fusion
└── LICENSE                                        # MIT License (ScaleAGData 2025)
```

Model artefacts, metrics, and prediction outputs are written to an `artefacts/` directory created at runtime.

---

## Data Sources

| Source | Type | Bands | Role |
|---|---|---|---|
| VTT HSI Sensor | Ground/airborne hyperspectral | 6 bands | Primary reflectance input |
| Kuva Space L2A | Hyperspectral satellite | 22 bands | Fusion feature set |
| Sentinel-2 | Multispectral satellite | 10 bands (B2–B12) | Fusion feature set / GEE ETL |

Resources can be found at:  
- https://zenodo.org/records/19845300
- https://zenodo.org/records/19814229
- https://scaleagdata-soc-api-1002910116761.europe-west1.run.app

---

## Workflows

### 1. VTT HSI Analysis

```
Raw VTT GeoTIFF
    ↓
Radiometric Calibration (reference reflectance panels → DN-to-reflectance per band)
    ↓
Point Extraction + Spectral Index Computation
  6 bands + 5 first-order derivatives + 15 normalized band differences = 26 features
    ↓
5-fold Cross-Validation Training: CNN / Random Forest / XGBoost
    ↓
Full-image OC Prediction Maps (GeoTIFF + PNG, batch-processed)
```

**Run training:**
```bash
cd "VTT HSI sensor"
python VTT_analysis_2026_train.py
```

**Run prediction:**
```bash
cd "VTT HSI sensor"
python VTT_analysis_2026_predict.py
```

---

### 2. Data Fusion (Kuva + Sentinel-2 + VTT)

Four feature configurations are trained and evaluated:

| Config | Sources | Features |
|---|---|---|
| 1 | VTT only | 26 |
| 2 | VTT + Kuva | 48 |
| 3 | VTT + Sentinel-2 | 36 |
| 4 | VTT + Kuva + Sentinel-2 | 58 |

Bare-soil filtering is applied via NDVI thresholding (< 0.35) before training to reduce noise from vegetated pixels.

**Outputs:** metrics CSV, model comparison plots, permutation importance CSVs, feature rank heatmaps, OC prediction GeoTIFFs.

**Run training:**
```bash
cd "data fusion soillab"
python datafusion_train_kuva_sentinel.py
```

**Run prediction:**
```bash
cd "data fusion soillab"
python datafusion_predict.py
```

---

### 3. Google Earth Engine ETL

Extracts Sentinel-2 surface reflectance time-series (2018–present) at known soil sample points in Flanders, merges with LUCAS soil properties (bulk density, clay, coarse fragments, sand, silt), and exports 70/30 train/test TFRecord files to Google Cloud Storage.

**Spectral indices computed:** NDVI, NBR2, VNSIR, BSI, NDWI, NSMI

```bash
python Googe_earth_engin_extract_sentinel2_data.py
```

Requires an authenticated Google Earth Engine account and Google Cloud project with Storage access.

---

## Model Specifications

### CNN (VTT only)
- Architecture: Conv1D [16, 8] filters, kernel size 3, L2 regularization 0.0004, Dense [32, 16]
- Optimizer: Adam (lr=0.002), early stopping (patience=50), max 200 epochs

### Random Forest
- 200 estimators, unlimited depth

### XGBoost
- 200 estimators, learning rate 0.1, max depth 4, subsample 0.8, colsample_bytree 0.8

All models use 5-fold cross-validation. Fusion training additionally records out-of-fold predictions for unbiased evaluation.

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- XGBoost
- rasterio (with GDAL)
- pyproj
- numpy, pandas, matplotlib
- joblib
- Google Earth Engine Python API (`earthengine-api`) — for the GEE ETL only
- Google Cloud SDK + service account credentials — for GEE and Cloud Storage export

Install dependencies:
```bash
pip install tensorflow scikit-learn xgboost rasterio pyproj numpy pandas matplotlib joblib earthengine-api
```

Authenticate Earth Engine:
```bash
earthengine authenticate
```

---
