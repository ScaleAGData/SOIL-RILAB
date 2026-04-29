import pandas as pd
import matplotlib.pyplot as plt

from config_fields import B2_config, Caritas_config, L1_config, L3_config, M5B_config, R6_config, RvK1AB_config, S24_config
from python_functions import provide_info_geojson, process_geotiff, add_spectral_indices, train_cnn_model, train_rf_model, train_xgb_model
# --------------------------
# Provide general info about geotiff
# --------------------------
# Replace 'path/to/your/geotiff.tif' with the path to your GeoTIFF file
geotiff_files = ['./2025_analysis/images/B2-orthophoto-corrected.tif', 
                 './2025_analysis/images/VTT-Caritas-corrected.tif',
                 './2025_analysis/images/VTT-L1-corrected.tif',
                 './2025_analysis/images/VTT-L3-corrected.tif',
                 './2025_analysis/images/VTT-M5B-corrected.tif',
                 './2025_analysis/images/VTT-R6-corrected.tif',
                 './2025_analysis/images/VTT-RvK1A-B-corrected.tif',
                 './2025_analysis/images/VTT-S24-corrected.tif'
                 ]
                     
for geotiff_path in geotiff_files:
    provide_info_geojson(geotiff_path)

# --------------------------
# Step 1: Get data for sample points
# --------------------------
# Process B2 image
data_VTT_B2 = process_geotiff(B2_config)

# Process Caritas image
data_VTT_Caritas = process_geotiff(Caritas_config)

# Process L1 image
data_VTT_L1 = process_geotiff(L1_config)

# Process L3 image
data_VTT_L3 = process_geotiff(L3_config)

# Process M5B image
data_VTT_M5B = process_geotiff(M5B_config)

# Process R6 image
data_VTT_R6 = process_geotiff(R6_config)

# Process RvK1AB image
data_VTT_RvK1AB = process_geotiff(RvK1AB_config)

# Process S24 image
data_VTT_S24 = process_geotiff(S24_config)

# Combine datasets from all images
data_VTT = pd.concat([data_VTT_B2, data_VTT_Caritas, data_VTT_L1, data_VTT_L3, data_VTT_M5B, data_VTT_R6, data_VTT_RvK1AB, data_VTT_S24], axis=0)

# Add spectral indices (first-order derivatives and normalised band differences)
band_columns = ['band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_6']
data_VTT = add_spectral_indices(data_VTT, band_columns)

feature_columns = [c for c in data_VTT.columns if c.startswith('band_') or c.startswith('d_') or c.startswith('nd_')]
print(f"Training with {len(feature_columns)} features: {feature_columns}")

# --------------------------
# Step 2: Train and compare models
# --------------------------
cnn_results = train_cnn_model(
    data=data_VTT,
    feature_columns=feature_columns,
    target_column='OC',
    model_name='artefacts/oc_prediction_cnn_model'
)

rf_results = train_rf_model(
    data=data_VTT,
    feature_columns=feature_columns,
    target_column='OC',
    model_name='artefacts/oc_prediction_rf_model'
)

xgb_results = train_xgb_model(
    data=data_VTT,
    feature_columns=feature_columns,
    target_column='OC',
    model_name='artefacts/oc_prediction_xgb_model'
)

# --------------------------
# Step 3: Compare models (5-fold CV metrics)
# --------------------------
models     = ['CNN',              'Random Forest',      'XGBoost'             ]
colors     = ['steelblue',        'darkorange',         'seagreen'            ]
rmse_mean  = [cnn_results['rmse'], rf_results['rmse'],  xgb_results['rmse']   ]
rmse_std   = [cnn_results['rmse_std'], rf_results['rmse_std'], xgb_results['rmse_std']]
r2_mean    = [cnn_results['r2'],   rf_results['r2'],    xgb_results['r2']     ]
r2_std     = [cnn_results['r2_std'],   rf_results['r2_std'],    xgb_results['r2_std'] ]

print("\n" + "="*65)
print(f"{'Model':<20} {'RMSE (mean±std)':>20} {'R² (mean±std)':>20}")
print("-"*65)
for name, rmse, rs, r2, r2s in zip(models, rmse_mean, rmse_std, r2_mean, r2_std):
    print(f"{name:<20} {rmse:>8.4f} ± {rs:<8.4f}  {r2:>8.4f} ± {r2s:<8.4f}")
print("="*65)

# Save evaluation metrics to CSV
metrics_df = pd.DataFrame({
    'model':    models,
    'rmse':     rmse_mean,
    'rmse_std': rmse_std,
    'r2':       r2_mean,
    'r2_std':   r2_std,
    'mse':      [cnn_results['mse'], rf_results['mse'], xgb_results['mse']],
})
metrics_path = '2025_analysis/artefacts/model_evaluation_metrics.csv'
metrics_df.to_csv(metrics_path, index=False, float_format='%.6f')
print(f"Evaluation metrics saved to {metrics_path}")

_, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].bar(models, rmse_mean, color=colors, yerr=rmse_std, capsize=5)
axes[0].set_ylabel('RMSE')
axes[0].set_title('RMSE (lower is better) – 5-Fold CV')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(models, r2_mean, color=colors, yerr=r2_std, capsize=5)
axes[1].set_ylabel('R²')
axes[1].set_title('R² (higher is better) – 5-Fold CV')
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Model Comparison – OC Prediction (5-Fold CV)', fontweight='bold')
plt.tight_layout()
plt.savefig('2025_analysis/artefacts/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
