"""
train_datafusion_kuva_sentinel.py

Bare-soil filtering using BOTH Kuva Space L2A hyperspectral and Sentinel-2 imagery,
followed by training Random Forest and XGBoost models in four configurations:
  1. VTT HSI only
  2. VTT HSI + Kuva L2A bands
  3. VTT HSI + Sentinel-2 bands
  4. VTT HSI + Kuva L2A + Sentinel-2 bands (full fusion)

Bare-soil filter: a point is kept if its lowest NDVI across ANY source
(Kuva OR Sentinel-2) is below NDVI_THRESHOLD.

Steps:
  1. Load Kuva L2A and Sentinel-2 images
  2. Extract VTT HSI reflectance for all config points via process_geotiff()
  3. NDVI bare-soil filter using all satellite images combined
  4. Extract Kuva L2A and Sentinel-2 band features per point
  5. Add VTT spectral indices
  6. Train RF + XGBoost for each of the 4 feature configurations
  7. Compare metrics and save plots to artefacts_datafusion/
  8. Sensitivity / feature importance analysis
"""

import os
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pyproj import Transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from config_fields import (
    B2_config, Caritas_config, L1_config, L3_config,
    M5B_config, R6_config, RvK1AB_config, S24_config,
)
from python_functions import process_geotiff, add_spectral_indices


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

# Kuva Space L2A hyperspectral images
KUVA_PATHS = [
    './2025_analysis/images/L2A20250405.tif',
    # './2025_analysis/images/L2A20250509.tif', # did visual check of sentinal 2 images for this date and much more vegetiation, so only use the 20250405 images
    './2025_analysis/images/L2A20250405b.tif',
    # './2025_analysis/images/L2A20250509b.tif', # did visual check of sentinal 2 images for this date and much more vegetiation, so only use the 20250405 images
]
KUVA_RED_IDX = 11   # 674.94 nm (0-based)
KUVA_NIR_IDX = 20   # 850.0 nm  (0-based)

# Sentinel-2 L2A images (10m bands stacked: B2, B3, B4, B8, B8A, B11, B12, ...)
# Adjust S2_RED_IDX / S2_NIR_IDX to match the band order in your specific GeoTIFF.
# For a standard 4-band (B2/B3/B4/B8) stack: RED=2, NIR=3
# For a 10-band (B2/B3/B4/B5/B6/B7/B8/B8A/B11/B12) stack: RED=2, NIR=6
S2_PATHS = [
    './2025_analysis/images/S2_20250405.tif',
    # './2025_analysis/images/S2_20250509.tif', # did visual check of sentinal 2 images for this date and much more vegetiation, so only use the 20250405 images
]
S2_RED_IDX = 2   # Band 4 – 665 nm
S2_NIR_IDX = 6   # Band 8 – 842 nm  ← adjust if your stack differs

NDVI_THRESHOLD = 0.35
ARTEFACTS_DIR  = '2025_analysis/artefacts_datafusion'
RANDOM_STATE   = 42
N_SPLITS       = 5

ALL_CONFIGS = [
    B2_config, Caritas_config, L1_config, L3_config,
    M5B_config, R6_config, RvK1AB_config, S24_config,
]

os.makedirs(ARTEFACTS_DIR, exist_ok=True)


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def load_images(paths, label):
    """Load a list of GeoTIFF paths into (cubes, transforms) lists.
    Missing files are silently skipped with a warning."""
    cubes, transforms = [], []
    for path in paths:
        if not os.path.exists(path):
            print(f"  WARNING: {label} image not found, skipping: {path}")
            continue
        with rasterio.open(path) as src:
            cube = src.read().astype(np.float32)
            t    = src.transform
            h, w = cube.shape[1], cube.shape[2]
            x_min, x_max = t.c, t.c + w * t.a
            y_min, y_max = t.f + h * t.e, t.f
            cubes.append(cube)
            transforms.append(t)
            print(f"  Loaded {os.path.basename(path)}: shape={cube.shape}, CRS={src.crs}")
            print(f"    UTM extent: X=[{x_min:.0f}, {x_max:.0f}]  Y=[{y_min:.0f}, {y_max:.0f}]")
    return cubes, transforms


def extract_pixel(cube, transform, utm_x, utm_y):
    """Return spectrum [n_bands] at (utm_x, utm_y), or None if out-of-bounds / no-data."""
    try:
        row, col = rasterio.transform.rowcol(transform, utm_x, utm_y)
        if row < 0 or col < 0 or row >= cube.shape[1] or col >= cube.shape[2]:
            return None
        pixel = cube[:, row, col]
        if np.all(pixel == 0):
            return None
        return pixel
    except Exception:
        return None


def best_pixel_and_ndvi(cubes, transforms, utm_x, utm_y, red_idx, nir_idx):
    """Return (pixel, ndvi) with the lowest NDVI across all images, or (None, None)."""
    best_pixel, best_ndvi = None, None
    for cube, transform in zip(cubes, transforms):
        pixel = extract_pixel(cube, transform, utm_x, utm_y)
        if pixel is None:
            continue
        red, nir = float(pixel[red_idx]), float(pixel[nir_idx])
        denom = red + nir
        if denom == 0:
            continue
        ndvi = (nir - red) / denom
        if best_ndvi is None or ndvi < best_ndvi:
            best_ndvi = ndvi
            best_pixel = pixel
    return best_pixel, best_ndvi


# -------------------------------------------------------
# STEP 1: Load satellite images
# -------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading satellite images")
print("=" * 60)

print("\nKuva Space L2A:")
kuva_cubes, kuva_transforms = load_images(KUVA_PATHS, "Kuva L2A")

print("\nSentinel-2 L2A:")
s2_cubes, s2_transforms = load_images(S2_PATHS, "Sentinel-2")

n_kuva_bands = kuva_cubes[0].shape[0] if kuva_cubes else 0
n_s2_bands   = s2_cubes[0].shape[0]   if s2_cubes   else 0
print(f"\nKuva bands : {n_kuva_bands}")
print(f"S2 bands   : {n_s2_bands}")

# Transformer: always reproject from WGS84 lon/lat → EPSG:32631
transformer_to_utm31 = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)


# -------------------------------------------------------
# STEP 2: Extract VTT HSI reflectance for all config points
# -------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Extracting VTT HSI reflectance")
print("=" * 60)

vtt_frames = []
for config in ALL_CONFIGS:
    df = process_geotiff(config)
    if not df.empty:
        vtt_frames.append(df)

data_vtt_all = pd.concat(vtt_frames, axis=0, ignore_index=True)
print(f"\nTotal VTT points extracted: {len(data_vtt_all)}")


# -------------------------------------------------------
# STEP 3: NDVI bare-soil filter (Kuva + Sentinel-2 combined)
#   A point is kept if NDVI < NDVI_THRESHOLD in ANY source.
# -------------------------------------------------------
print("\n" + "=" * 60)
print(f"STEP 3: NDVI bare-soil filter (threshold < {NDVI_THRESHOLD})")
print(f"        Sources: Kuva L2A ({len(kuva_cubes)} images)  +  "
      f"Sentinel-2 ({len(s2_cubes)} images)")
print("=" * 60)

kept_indices   = []
kuva_pixels    = []   # best Kuva pixel per kept point (or None)
s2_pixels      = []   # best S2 pixel per kept point (or None)
ndvi_kept      = []   # lowest NDVI across all sources
ndvi_source    = []   # which source gave the lowest NDVI

for df_idx, row in data_vtt_all.iterrows():
    utm_x, utm_y = transformer_to_utm31.transform(row['longitude'], row['latitude'])

    kuva_pix, kuva_ndvi = best_pixel_and_ndvi(
        kuva_cubes, kuva_transforms, utm_x, utm_y, KUVA_RED_IDX, KUVA_NIR_IDX)
    s2_pix, s2_ndvi = best_pixel_and_ndvi(
        s2_cubes, s2_transforms, utm_x, utm_y, S2_RED_IDX, S2_NIR_IDX)

    # Determine the overall lowest NDVI across both sources
    candidates = [(ndvi, src) for ndvi, src in
                  [(kuva_ndvi, 'kuva'), (s2_ndvi, 's2')]
                  if ndvi is not None]

    if not candidates:
        print(f"  Skipped {row['name']}: not found in any image")
        continue

    best_overall_ndvi, best_source = min(candidates, key=lambda x: x[0])

    if best_overall_ndvi >= NDVI_THRESHOLD:
        print(f"  Skipped {row['name']}: min NDVI={best_overall_ndvi:.3f} "
              f"(from {best_source}) >= {NDVI_THRESHOLD}")
        continue

    kept_indices.append(df_idx)
    kuva_pixels.append(kuva_pix)
    s2_pixels.append(s2_pix)
    ndvi_kept.append(best_overall_ndvi)
    ndvi_source.append(best_source)

print(f"\nBare-soil points kept : {len(kept_indices)} / {len(data_vtt_all)}")
if kept_indices:
    from collections import Counter
    src_counts = Counter(ndvi_source)
    print(f"  Determined by Kuva  : {src_counts.get('kuva', 0)}")
    print(f"  Determined by S2    : {src_counts.get('s2', 0)}")
    print(f"NDVI stats (kept)     : min={min(ndvi_kept):.3f}  "
          f"mean={np.mean(ndvi_kept):.3f}  max={max(ndvi_kept):.3f}")
    print(f"\n{'Point':<20}  {'OC':>5}  {'NDVI':>6}  {'Source'}")
    print("-" * 44)
    for idx, ndvi, src in zip(kept_indices, ndvi_kept, ndvi_source):
        row = data_vtt_all.loc[idx]
        print(f"  {str(row['name']):<18}  {row['OC']:>5.2f}  {ndvi:>6.3f}  {src}")

if len(kept_indices) == 0:
    raise ValueError(
        "No points passed the NDVI filter. "
        "Check NDVI_THRESHOLD, band indices, or image alignment."
    )

# NDVI histogram
plt.figure()
plt.hist(ndvi_kept, bins=15)
plt.axvline(NDVI_THRESHOLD, color='r', linestyle='--', label=f'Threshold = {NDVI_THRESHOLD}')
plt.xlabel('NDVI (lowest across all satellite images)')
plt.ylabel('Count')
plt.title('NDVI distribution – bare-soil filtered points')
plt.legend()
plt.tight_layout()
plt.savefig(f'{ARTEFACTS_DIR}/ndvi_distribution.png', dpi=150)
plt.close()


# -------------------------------------------------------
# STEP 4: Build the working dataset
# -------------------------------------------------------
data_bare = data_vtt_all.loc[kept_indices].copy().reset_index(drop=True)
data_bare['ndvi_filter_source'] = ndvi_source

# Attach Kuva L2A band columns (NaN where point has no Kuva coverage)
kuva_band_cols = [f'kuva_band_{i + 1}' for i in range(n_kuva_bands)]
if n_kuva_bands > 0:
    kuva_array = np.array([
        pix if pix is not None else np.full(n_kuva_bands, np.nan)
        for pix in kuva_pixels
    ])
    data_bare = pd.concat(
        [data_bare, pd.DataFrame(kuva_array, columns=kuva_band_cols)], axis=1)

# Attach Sentinel-2 band columns (NaN where point has no S2 coverage)
s2_band_cols = [f's2_band_{i + 1}' for i in range(n_s2_bands)]
if n_s2_bands > 0:
    s2_array = np.array([
        pix if pix is not None else np.full(n_s2_bands, np.nan)
        for pix in s2_pixels
    ])
    data_bare = pd.concat(
        [data_bare, pd.DataFrame(s2_array, columns=s2_band_cols)], axis=1)

# Add VTT spectral indices (first-order derivatives + normalised band differences)
vtt_raw_band_cols = [c for c in data_bare.columns if c.startswith('band_')]
data_bare = add_spectral_indices(data_bare, vtt_raw_band_cols)

vtt_feature_cols = [c for c in data_bare.columns
                    if c.startswith('band_') or c.startswith('d_') or c.startswith('nd_')]

# Define the four feature sets and their training subsets
# (exclude rows where required satellite features are NaN)
feature_sets = {'vtt_only': vtt_feature_cols}

if n_kuva_bands > 0:
    feature_sets['vtt_kuva']    = vtt_feature_cols + kuva_band_cols
if n_s2_bands > 0:
    feature_sets['vtt_s2']      = vtt_feature_cols + s2_band_cols
if n_kuva_bands > 0 and n_s2_bands > 0:
    feature_sets['vtt_kuva_s2'] = vtt_feature_cols + kuva_band_cols + s2_band_cols

print(f"\nVTT features             : {len(vtt_feature_cols)}")
for name, cols in feature_sets.items():
    mask = data_bare[cols].notna().all(axis=1)
    print(f"  {name:<20}: {len(cols)} features,  {mask.sum()} valid samples")

# Save dataset
data_bare.to_csv(f'{ARTEFACTS_DIR}/bare_soil_dataset.csv', index=False, float_format='%.6f')
print(f"\nDataset saved to {ARTEFACTS_DIR}/bare_soil_dataset.csv")


# -------------------------------------------------------
# TRAINING HELPER
# -------------------------------------------------------
def train_and_evaluate(data, feature_cols, target_col, model_prefix):
    """
    Train RF and XGBoost with k-fold CV on rows that have no NaN in feature_cols.
    Saves the final (all-data) model and a scatter + feature-importance plot.
    Returns dict with CV metrics for each model type.
    """
    mask = data[feature_cols].notna().all(axis=1)
    subset = data[mask].reset_index(drop=True)
    X = subset[feature_cols].values
    y = subset[target_col].values

    if len(y) < N_SPLITS:
        print(f"  Skipping {model_prefix}: only {len(y)} samples (need >= {N_SPLITS})")
        return {}

    results = {}

    for model_name, make_model in [
        ('rf',  lambda: RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ('xgb', lambda: XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0)),
    ]:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        rmse_scores, r2_scores = [], []
        oof_preds = np.zeros(len(y))

        print(f"\n  {model_prefix.upper()} | {model_name.upper()} – {N_SPLITS}-Fold CV "
              f"(n={len(y)}):")
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            m = make_model()
            m.fit(X[train_idx], y[train_idx])
            preds = m.predict(X[test_idx])
            oof_preds[test_idx] = preds
            fold_rmse = np.sqrt(mean_squared_error(y[test_idx], preds))
            fold_r2   = r2_score(y[test_idx], preds)
            rmse_scores.append(fold_rmse)
            r2_scores.append(fold_r2)
            print(f"    Fold {fold_idx + 1}: RMSE={fold_rmse:.4f}, R²={fold_r2:.4f}")

        rmse_cv  = float(np.mean(rmse_scores))
        r2_cv    = float(np.mean(r2_scores))
        rmse_std = float(np.std(rmse_scores))
        r2_std   = float(np.std(r2_scores))
        print(f"    CV mean: RMSE={rmse_cv:.4f} ± {rmse_std:.4f},  "
              f"R²={r2_cv:.4f} ± {r2_std:.4f}")

        # Final model trained on all valid data
        final_model = make_model()
        final_model.fit(X, y)
        joblib.dump(final_model, f'{ARTEFACTS_DIR}/{model_prefix}_{model_name}_model.pkl')

        # OOF scatter + top-15 feature importance plot
        _, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(y, oof_preds, alpha=0.8)
        axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        axes[0].set_xlabel(f'Actual {target_col}')
        axes[0].set_ylabel(f'Predicted {target_col}')
        axes[0].set_title(f'{model_name.upper()} [{model_prefix}] – OOF ({N_SPLITS}-Fold CV)')
        axes[0].text(0.05, 0.95,
                     f'RMSE={rmse_cv:.4f} ± {rmse_std:.4f}\nR²={r2_cv:.4f} ± {r2_std:.4f}',
                     transform=axes[0].transAxes, va='top',
                     bbox=dict(facecolor='white', alpha=0.8))
        axes[0].grid(True, alpha=0.3)

        importances = final_model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        axes[1].barh(
            [feature_cols[i] for i in top_idx[::-1]],
            importances[top_idx[::-1]])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Top 15 Feature Importances')
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(f'{ARTEFACTS_DIR}/training_{model_prefix}_{model_name}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        results[model_name] = {
            'rmse': rmse_cv, 'rmse_std': rmse_std,
            'r2': r2_cv,     'r2_std': r2_std,
            'n': len(y),
        }

    return results


# -------------------------------------------------------
# STEPS 5–6: Train all feature-set / model combinations
# -------------------------------------------------------
all_results = {}

step = 5
for fs_name, fs_cols in feature_sets.items():
    print("\n" + "=" * 60)
    print(f"STEP {step}: {fs_name} models (RF + XGBoost)")
    print("=" * 60)
    all_results[fs_name] = train_and_evaluate(data_bare, fs_cols, 'OC', fs_name)
    step += 1


# -------------------------------------------------------
# STEP (step): Summary comparison
# -------------------------------------------------------
print("\n" + "=" * 75)
print(f"{'Model':<30} {'n':>4}  {'RMSE (mean ± std)':>22}  {'R² (mean ± std)':>22}")
print("-" * 75)

summary_rows = []
for fs_name, res in all_results.items():
    for model_name in ('rf', 'xgb'):
        if model_name not in res:
            continue
        label = f"{fs_name} / {model_name.upper()}"
        r = res[model_name]
        summary_rows.append((label, r))
        print(f"{label:<30} {r['n']:>4}  "
              f"{r['rmse']:>8.4f} ± {r['rmse_std']:<8.4f}  "
              f"{r['r2']:>8.4f} ± {r['r2_std']:<8.4f}")
print("=" * 75)

# Bar chart: one group per feature set, RF and XGBoost side by side
fs_names   = list(all_results.keys())
model_keys = ['rf', 'xgb']
model_labels = {'rf': 'RF', 'xgb': 'XGB'}
bar_colors   = {'rf': 'steelblue', 'xgb': 'darkorange'}
x = np.arange(len(fs_names))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(max(10, len(fs_names) * 3), 5))
for i, (mk, offset) in enumerate(zip(model_keys, [-width / 2, width / 2])):
    rmse_vals = [all_results[fs].get(mk, {}).get('rmse', np.nan) for fs in fs_names]
    rmse_stds = [all_results[fs].get(mk, {}).get('rmse_std', 0)  for fs in fs_names]
    r2_vals   = [all_results[fs].get(mk, {}).get('r2',   np.nan) for fs in fs_names]
    r2_stds   = [all_results[fs].get(mk, {}).get('r2_std', 0)    for fs in fs_names]

    axes[0].bar(x + offset, rmse_vals, width, label=model_labels[mk],
                color=bar_colors[mk], yerr=rmse_stds, capsize=4)
    axes[1].bar(x + offset, r2_vals,   width, label=model_labels[mk],
                color=bar_colors[mk], yerr=r2_stds, capsize=4)

for ax, ylabel, title in [
    (axes[0], 'RMSE', 'RMSE (lower is better) – 5-Fold CV'),
    (axes[1], 'R²',   'R² (higher is better) – 5-Fold CV'),
]:
    ax.set_xticks(x)
    ax.set_xticklabels(fs_names, rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Model Comparison – OC Prediction (5-Fold CV)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{ARTEFACTS_DIR}/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Metrics CSV
metrics_records = []
for fs_name, res in all_results.items():
    for mk in model_keys:
        if mk in res:
            r = res[mk]
            metrics_records.append({
                'feature_set': fs_name, 'model': mk.upper(),
                'n': r['n'], 'rmse': r['rmse'], 'rmse_std': r['rmse_std'],
                'r2': r['r2'], 'r2_std': r['r2_std'],
            })
pd.DataFrame(metrics_records).to_csv(
    f'{ARTEFACTS_DIR}/model_evaluation_metrics.csv', index=False, float_format='%.6f')
print(f"\nSaved metrics  → {ARTEFACTS_DIR}/model_evaluation_metrics.csv")
print(f"Saved plots    → {ARTEFACTS_DIR}/")


# -------------------------------------------------------
# STEP (step+1): Sensitivity / feature importance analysis
# -------------------------------------------------------
print("\n" + "=" * 60)
print(f"STEP {step + 1}: Sensitivity analysis")
print("=" * 60)


def sensitivity_analysis(model, feature_cols, X, y, model_label):
    """Compute built-in and permutation importances. Returns ranked DataFrame."""
    perm = permutation_importance(
        model, X, y, n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1)
    df = pd.DataFrame({
        'feature':          feature_cols,
        'builtin':          model.feature_importances_,
        'permutation_mean': perm.importances_mean,
        'permutation_std':  perm.importances_std,
    }).sort_values('permutation_mean', ascending=False).reset_index(drop=True)
    df.to_csv(f'{ARTEFACTS_DIR}/sensitivity_{model_label}.csv',
              index=False, float_format='%.6f')
    print(f"  [{model_label}] top-5 by permutation importance:")
    for _, r in df.head(5).iterrows():
        print(f"    {r['feature']:<25}  perm={r['permutation_mean']:.4f} ± "
              f"{r['permutation_std']:.4f}  builtin={r['builtin']:.4f}")
    return df


def plot_sensitivity(dfs, labels, colors, title, filename, top_n=15):
    """Side-by-side horizontal bar charts of permutation importance."""
    top_features = dfs[0].head(top_n)['feature'].tolist()
    fig, axes = plt.subplots(1, len(dfs), figsize=(7 * len(dfs), 6), sharey=False)
    if len(dfs) == 1:
        axes = [axes]
    for ax, df, label, color in zip(axes, dfs, labels, colors):
        sub  = df[df['feature'].isin(top_features)].set_index('feature').reindex(top_features)
        vals = sub['permutation_mean'].values
        stds = sub['permutation_std'].values
        ax.barh(top_features[::-1], vals[::-1], xerr=stds[::-1],
                color=color, alpha=0.8, capsize=3)
        ax.set_xlabel('Permutation importance (mean ± std)')
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(0, color='black', linewidth=0.8)
    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{ARTEFACTS_DIR}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()


def plot_importance_heatmap(dfs, labels, shared_features, filename, top_n=20):
    """Rank heatmap across models for a shared feature set."""
    rank_data = {}
    for df, label in zip(dfs, labels):
        ranked = df.set_index('feature')['permutation_mean'].reindex(
            shared_features, fill_value=0)
        rank_data[label] = ranked.rank(ascending=False)
    rank_df = pd.DataFrame(rank_data, index=shared_features)
    rank_df['avg_rank'] = rank_df.mean(axis=1)
    top_feat = rank_df.nsmallest(top_n, 'avg_rank').index.tolist()
    rank_df  = rank_df.loc[top_feat].drop(columns='avg_rank')

    fig, ax = plt.subplots(figsize=(len(labels) * 2 + 2, top_n * 0.4 + 2))
    im = ax.imshow(rank_df.values, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticks(range(len(top_feat)))
    ax.set_yticklabels(top_feat)
    plt.colorbar(im, ax=ax, label='Rank (1 = most important)')
    ax.set_title('Feature importance rank across models\n(green = more important)',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{ARTEFACTS_DIR}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()


# Run sensitivity analysis for every trained model
sens_dfs   = {}   # (fs_name, model_key) → DataFrame
sens_labels = []
sens_colors_map = {
    'vtt_only':    {'rf': 'steelblue',   'xgb': 'cornflowerblue'},
    'vtt_kuva':    {'rf': 'darkorange',  'xgb': 'sandybrown'},
    'vtt_s2':      {'rf': 'seagreen',    'xgb': 'mediumseagreen'},
    'vtt_kuva_s2': {'rf': 'purple',      'xgb': 'mediumpurple'},
}

for fs_name, fs_cols in feature_sets.items():
    mask    = data_bare[fs_cols].notna().all(axis=1)
    subset  = data_bare[mask].reset_index(drop=True)
    X_sub   = subset[fs_cols].values
    y_sub   = subset['OC'].values
    if len(y_sub) < N_SPLITS:
        continue
    for mk in model_keys:
        if mk not in all_results.get(fs_name, {}):
            continue
        model_label = f"{fs_name}_{mk}"
        model = joblib.load(f'{ARTEFACTS_DIR}/{fs_name}_{mk}_model.pkl')
        print(f"\n{model_label}:")
        sens_dfs[(fs_name, mk)] = sensitivity_analysis(
            model, fs_cols, X_sub, y_sub, model_label)

# Side-by-side per feature set: RF vs XGB
for fs_name, fs_cols in feature_sets.items():
    dfs_pair, lbls_pair, clrs_pair = [], [], []
    for mk, label_suffix in [('rf', 'RF'), ('xgb', 'XGB')]:
        key = (fs_name, mk)
        if key in sens_dfs:
            dfs_pair.append(sens_dfs[key])
            lbls_pair.append(f"{fs_name} {label_suffix}")
            clrs_pair.append(sens_colors_map.get(fs_name, {}).get(mk, 'grey'))
    if len(dfs_pair) >= 1:
        plot_sensitivity(
            dfs=dfs_pair, labels=lbls_pair, colors=clrs_pair,
            title=f'Permutation importance – {fs_name}',
            filename=f'sensitivity_{fs_name}.png',
        )

# Rank heatmap for VTT features across all models (shared feature set)
heatmap_dfs, heatmap_labels = [], []
for fs_name in feature_sets:
    for mk in model_keys:
        if (fs_name, mk) in sens_dfs:
            df_sub = sens_dfs[(fs_name, mk)]
            df_sub = df_sub[df_sub['feature'].isin(vtt_feature_cols)].reset_index(drop=True)
            heatmap_dfs.append(df_sub)
            heatmap_labels.append(f"{fs_name}\n{mk.upper()}")

if heatmap_dfs:
    plot_importance_heatmap(
        dfs=heatmap_dfs, labels=heatmap_labels,
        shared_features=vtt_feature_cols,
        filename='sensitivity_rank_heatmap_vtt.png',
        top_n=min(20, len(vtt_feature_cols)),
    )

print(f"\nSensitivity outputs saved to {ARTEFACTS_DIR}/")
