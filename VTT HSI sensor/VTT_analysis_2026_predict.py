
from config_fields import B2_config, Caritas_config, L1_config, L3_config, M5B_config, R6_config, RvK1AB_config, S24_config
from python_functions import predict_oc_for_image_cnn, predict_oc_for_image_rf, predict_oc_for_image_xgb


# configs = [B2_config, Caritas_config, L1_config, L3_config, R6_config, M5B_config, RvK1AB_config, S24_config]
configs = [B2_config]

# --------------------------
# make OC predictions for each field — CNN model
# --------------------------
for config in configs:
    predict_oc_for_image_cnn(
        config,
        model_path='./2025_analysis/artefacts/oc_prediction_cnn_model.keras',
        scaler_path='./2025_analysis/artefacts/scaler_oc_prediction_cnn_model.pkl',
    )

# --------------------------
# make OC predictions for each field — Random Forest model
# --------------------------
for config in configs:
    predict_oc_for_image_rf(
        config,
        model_path='./2025_analysis/artefacts/oc_prediction_rf_model.pkl',
    )

# --------------------------
# make OC predictions for each field — XGBoost model
# --------------------------
for config in configs:
    predict_oc_for_image_xgb(
        config,
        model_path='./2025_analysis/artefacts/oc_prediction_xgb_model.pkl',
    )
