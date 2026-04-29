"""
predict_datafusion.py

For each VTT field in config_fields.py, predict organic carbon content
using the four data-fusion model configurations trained in
train_datafusion_kuva_sentinel.py:

  1. vtt_only    – VTT HSI bands + spectral indices
  2. vtt_kuva    – VTT + Kuva L2A hyperspectral bands
  3. vtt_s2      – VTT + Sentinel-2 bands
  4. vtt_kuva_s2 – VTT + Kuva L2A + Sentinel-2 bands

For each configuration, predictions are made with both RF and XGBoost.

Strategy:
  - Read and calibrate the VTT image (using saved calibration models)
  - Warp the Kuva L2A and Sentinel-2 images to the VTT pixel grid using
    rasterio.warp.reproject — this aligns all three sources without needing
    per-pixel coordinate lookups
  - Build feature matrices per pixel and run model inference in batches
  - Save one GeoTIFF + one visualisation PNG per model per field
    in 2025_analysis/prediction_results_datafusion/

Usage (from the project root):
    cd 2025_analysis && python predict_datafusion.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from config_fields import (
    B2_config, Caritas_config, L1_config, L3_config,
    M5B_config, R6_config, RvK1AB_config, S24_config,
)
from python_functions import add_spectral_indices


# -------------------------------------------------------
# CONFIG  — must match train_datafusion_kuva_sentinel.py
# -------------------------------------------------------
KUVA_PATHS = [
    './2025_analysis/images/L2A20250405.tif',
    './2025_analysis/images/L2A20250405b.tif',
]
S2_PATHS = [
    './2025_analysis/images/S2_20250405.tif',
]

N_KUVA_BANDS = 22
N_S2_BANDS   = 10   # B2 B3 B4 B5 B6 B7 B8 B8A B11 B12

ARTEFACTS_DIR  = '2025_analysis/artefacts_datafusion'
OUTPUT_DIR     = '2025_analysis/prediction_results_datafusion'
CALIB_DIR      = '2025_analysis/artefacts'

BATCH_SIZE = 50_000
VIS_VMIN, VIS_VMAX = 0.5, 2.5   # OC colour scale for maps

ALL_CONFIGS = [
    # L3_config, L1_config, B2_config, Caritas_config, 
    # M5B_config, R6_config, RvK1AB_config,
     S24_config
]

ALL_CONFIGS = [
    S24_config
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------
# Feature column names (must match training order)
# -------------------------------------------------------
VTT_BAND_COLS  = [f'band_{i + 1}' for i in range(6)]
KUVA_BAND_COLS = [f'kuva_band_{i + 1}' for i in range(N_KUVA_BANDS)]
S2_BAND_COLS   = [f's2_band_{i + 1}'   for i in range(N_S2_BANDS)]

# Build VTT spectral index column names (same logic as add_spectral_indices)
_n = len(VTT_BAND_COLS)
DERIV_COLS = [f'd_{VTT_BAND_COLS[i]}_{VTT_BAND_COLS[i+1]}' for i in range(_n - 1)]
ND_COLS    = [f'nd_{VTT_BAND_COLS[i]}_{VTT_BAND_COLS[j]}'
              for i in range(_n) for j in range(i + 1, _n)]
VTT_FEATURE_COLS = VTT_BAND_COLS + DERIV_COLS + ND_COLS


# -------------------------------------------------------
# Model configurations: (prefix, rf_path, xgb_path, extra_cols)
# -------------------------------------------------------
MODEL_CONFIGS = [
    # ('vtt_only',    VTT_FEATURE_COLS,                                        None),
    # ('vtt_kuva',    VTT_FEATURE_COLS + KUVA_BAND_COLS,                       'kuva'),
    # ('vtt_s2',      VTT_FEATURE_COLS + S2_BAND_COLS,                         's2'),
    ('vtt_kuva_s2', VTT_FEATURE_COLS + KUVA_BAND_COLS + S2_BAND_COLS,        'both'),
]


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def load_model_safe(path):
    """Load a joblib model; return None if file doesn't exist."""
    if not os.path.exists(path):
        print(f"  WARNING: model not found, skipping: {path}")
        return None
    return joblib.load(path)


def warp_to_grid(src_path, dst_crs, dst_transform, dst_shape):
    """
    Read all bands from src_path and warp them to match
    (dst_crs, dst_transform, dst_shape[H x W]).
    Returns float32 array [n_bands, H, W] with np.nan where no data.
    Returns None if the file doesn't exist.
    """
    if not os.path.exists(src_path):
        return None

    H, W = dst_shape
    with rasterio.open(src_path) as src:
        n_bands = src.count
        warped  = np.full((n_bands, H, W), np.nan, dtype=np.float32)
        for b in range(1, n_bands + 1):
            band_data = np.zeros((H, W), dtype=np.float32)
            reproject(
                source=rasterio.band(src, b),
                destination=band_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
            # rasterio fills out-of-extent pixels with 0 — mark those as NaN
            band_data[band_data == 0] = np.nan
            warped[b - 1] = band_data
    return warped   # shape: [n_bands, H, W]


def build_vtt_features(reflectance_data, valid_mask):
    """
    From a calibrated [6, H, W] array, build the VTT feature matrix
    for all valid pixels.
    Returns (pixel_features [N, n_vtt_feats], valid pixel indices).
    """
    valid_rows, valid_cols = np.where(valid_mask)
    N = len(valid_rows)
    if N == 0:
        return None, valid_rows, valid_cols

    # Extract 6 bands per valid pixel → DataFrame → add spectral indices
    pixel_bands = reflectance_data[:, valid_rows, valid_cols].T  # [N, 6]
    df = pd.DataFrame(pixel_bands, columns=VTT_BAND_COLS)
    df = add_spectral_indices(df, VTT_BAND_COLS)
    feature_cols = VTT_FEATURE_COLS
    feats = df[feature_cols].values.astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0)
    return feats, valid_rows, valid_cols


def predict_and_save(config, vtt_feats, valid_rows, valid_cols,
                     kuva_warped, s2_warped, img_shape, vtt_meta):
    """
    Run all model configurations for one VTT field and save outputs.
    """
    image_name = config['image_name']
    H, W = img_shape
    N = len(valid_rows)
    points = config.get('points', [])

    # Pre-extract Kuva and S2 values at valid pixel positions
    kuva_feats = None
    if kuva_warped is not None:
        kuva_feats = kuva_warped[:, valid_rows, valid_cols].T   # [N, 22]
        kuva_valid = ~np.any(np.isnan(kuva_feats), axis=1)     # [N] bool
    else:
        kuva_valid = np.zeros(N, dtype=bool)

    s2_feats = None
    if s2_warped is not None:
        s2_feats = s2_warped[:, valid_rows, valid_cols].T       # [N, 10]
        s2_valid = ~np.any(np.isnan(s2_feats), axis=1)
    else:
        s2_valid = np.zeros(N, dtype=bool)

    for prefix, feature_cols, requires in MODEL_CONFIGS:
        for algo in ('rf', 'xgb'):
            model_path = f'{ARTEFACTS_DIR}/{prefix}_{algo}_model.pkl'
            model = load_model_safe(model_path)
            if model is None:
                continue

            print(f"  [{image_name}] {prefix} / {algo.upper()} ...")

            # Determine which pixels have all required features
            if requires is None:
                pixel_mask = np.ones(N, dtype=bool)
            elif requires == 'kuva':
                pixel_mask = kuva_valid
            elif requires == 's2':
                pixel_mask = s2_valid
            else:   # both
                pixel_mask = kuva_valid & s2_valid

            n_valid = pixel_mask.sum()
            if n_valid == 0:
                print(f"    No pixels with complete features — skipping.")
                continue

            # Build full feature matrix for valid pixels
            parts = [vtt_feats[pixel_mask]]
            if requires in ('kuva', 'both'):
                parts.append(kuva_feats[pixel_mask])
            if requires in ('s2', 'both'):
                parts.append(s2_feats[pixel_mask])
            X = np.hstack(parts).astype(np.float32)
            X = np.nan_to_num(X, nan=0.0)

            # Predict in batches
            preds = np.empty(n_valid, dtype=np.float32)
            sub_rows = valid_rows[pixel_mask]
            sub_cols = valid_cols[pixel_mask]

            for start in range(0, n_valid, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n_valid)
                preds[start:end] = model.predict(X[start:end])

            # Write into output grid
            oc_map = np.full((H, W), np.nan, dtype=np.float32)
            oc_map[sub_rows, sub_cols] = preds

            # Save GeoTIFF
            out_path = f'{OUTPUT_DIR}/{image_name}_{prefix}_{algo}_OC.tiff'
            out_meta = vtt_meta.copy()
            out_meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)
            with rasterio.open(out_path, 'w', **out_meta) as dst:
                dst.write(oc_map, 1)

            # Visualisation
            masked = np.ma.masked_invalid(oc_map)
            fig, axes = plt.subplots(2, 1, figsize=(10, 12))

            axes[0].hist(masked.compressed(), bins=50, color='steelblue', alpha=0.8)
            axes[0].axvline(np.nanmean(oc_map), color='red',   linestyle='--',
                            label=f"Mean: {np.nanmean(oc_map):.3f}")
            axes[0].axvline(np.nanmedian(oc_map), color='green', linestyle=':',
                            label=f"Median: {np.nanmedian(oc_map):.3f}")
            axes[0].set_title(f'{image_name} – {prefix} / {algo.upper()} histogram')
            axes[0].set_xlabel('Predicted OC (%)')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            im = axes[1].imshow(masked, cmap='viridis',
                                vmin=VIS_VMIN, vmax=VIS_VMAX)
            plt.colorbar(im, ax=axes[1], label='Predicted OC (%)')
            axes[1].set_title(f'{image_name} – {prefix} / {algo.upper()} OC map\n'
                              f'n_pixels={n_valid}  mean={np.nanmean(oc_map):.3f}  '
                              f'std={np.nanstd(oc_map):.3f}')
            axes[1].axis('off')

            # Overlay reference points if present
            if points:
                with rasterio.open(config['geotiff_path']) as src_ref:
                    from pyproj import Transformer
                    if src_ref.crs.to_epsg() == 4326:
                        for pt in points:
                            col_px = (pt['lon'] - src_ref.transform.c) / src_ref.transform.a
                            row_px = (pt['lat'] - src_ref.transform.f) / src_ref.transform.e
                            axes[1].plot(col_px, row_px, 'r+', markersize=8)
                            axes[1].annotate(f"{pt['OC']:.2f}",
                                             (col_px, row_px), color='white',
                                             fontsize=6, ha='left')
                    else:
                        tr = Transformer.from_crs("EPSG:4326",
                                                  src_ref.crs.to_string(),
                                                  always_xy=True)
                        for pt in points:
                            utm_x, utm_y = tr.transform(pt['lon'], pt['lat'])
                            col_px = (utm_x - src_ref.transform.c) / src_ref.transform.a
                            row_px = (utm_y - src_ref.transform.f) / src_ref.transform.e
                            axes[1].plot(col_px, row_px, 'r+', markersize=8)
                            axes[1].annotate(f"{pt['OC']:.2f}",
                                             (col_px, row_px), color='white',
                                             fontsize=6, ha='left')

            plt.tight_layout()
            vis_path = f'{OUTPUT_DIR}/{image_name}_{prefix}_{algo}_OC_map.png'
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"    Saved: {out_path}")
            print(f"           mean OC={np.nanmean(oc_map):.3f}  "
                  f"std={np.nanstd(oc_map):.3f}  n_pixels={n_valid}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
print("=" * 60)
print("predict_datafusion.py")
print("=" * 60)

# Check which models are available
available = []
for prefix, _, _ in MODEL_CONFIGS:
    for algo in ('rf', 'xgb'):
        p = f'{ARTEFACTS_DIR}/{prefix}_{algo}_model.pkl'
        if os.path.exists(p):
            available.append(f'{prefix}/{algo}')
print(f"\nModels found in {ARTEFACTS_DIR}:")
for m in available:
    print(f"  {m}")

for config in ALL_CONFIGS:
    image_name   = config['image_name']
    geotiff_path = config['geotiff_path']
    calib_file   = f'{CALIB_DIR}/calibration_models_{image_name}.pkl'

    print(f"\n{'='*60}")
    print(f"Field: {image_name}")
    print(f"{'='*60}")

    if not os.path.exists(geotiff_path):
        print(f"  VTT image not found: {geotiff_path} — skipping.")
        continue

    # ---- Load calibration ----
    if not os.path.exists(calib_file):
        print(f"  Calibration file not found: {calib_file} — skipping.")
        continue

    with open(calib_file, 'rb') as f:
        calibration_models = pickle.load(f)

    # ---- Read and calibrate VTT image ----
    with rasterio.open(geotiff_path) as src:
        vtt_meta      = src.meta.copy()
        dst_crs       = src.crs
        dst_transform = src.transform
        H, W          = src.height, src.width
        raw_data      = src.read().astype(np.float32)
        n_bands       = src.count

    if n_bands != 6:
        print(f"  Expected 6 VTT bands, found {n_bands} — skipping.")
        continue

    missing_mask = np.all(raw_data == 0, axis=0)
    reflectance  = np.zeros_like(raw_data)
    for b in range(n_bands):
        band_num = b + 1
        if band_num in calibration_models:
            a, coef = calibration_models[band_num]
            reflectance[b] = a * raw_data[b] + coef
        else:
            reflectance[b] = raw_data[b]
    reflectance[:, missing_mask] = np.nan

    print(f"  VTT image: {H}×{W}  ({(~missing_mask).sum()} valid pixels)")

    # ---- Warp Kuva and S2 to VTT grid ----
    print("  Warping Kuva L2A to VTT grid...")
    kuva_warped = None
    for kuva_path in KUVA_PATHS:
        w = warp_to_grid(kuva_path, dst_crs, dst_transform, (H, W))
        if w is None:
            continue
        # Use the image with the most non-NaN coverage for this field
        if kuva_warped is None:
            kuva_warped = w
        else:
            # Fill NaN gaps in the first image with values from the second
            nan_mask = np.isnan(kuva_warped).any(axis=0)
            kuva_warped[:, nan_mask] = w[:, nan_mask]

    if kuva_warped is not None:
        coverage = (~np.isnan(kuva_warped[0])).sum()
        print(f"    Kuva coverage: {coverage} pixels")
    else:
        print("    No Kuva image available.")

    print("  Warping Sentinel-2 to VTT grid...")
    s2_warped = None
    for s2_path in S2_PATHS:
        w = warp_to_grid(s2_path, dst_crs, dst_transform, (H, W))
        if w is None:
            continue
        if s2_warped is None:
            s2_warped = w
        else:
            nan_mask = np.isnan(s2_warped).any(axis=0)
            s2_warped[:, nan_mask] = w[:, nan_mask]

    if s2_warped is not None:
        coverage = (~np.isnan(s2_warped[0])).sum()
        print(f"    S2 coverage  : {coverage} pixels")
    else:
        print("    No S2 image available.")

    # ---- Build VTT feature matrix ----
    valid_mask = ~missing_mask
    vtt_feats, valid_rows, valid_cols = build_vtt_features(reflectance, valid_mask)

    if vtt_feats is None or len(valid_rows) == 0:
        print("  No valid VTT pixels — skipping.")
        continue

    # ---- Predict with all models ----
    predict_and_save(config, vtt_feats, valid_rows, valid_cols,
                     kuva_warped, s2_warped, (H, W), vtt_meta)

print(f"\nAll predictions saved to {OUTPUT_DIR}/")
