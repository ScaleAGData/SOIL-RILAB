import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — avoids tkinter threading crashes

import rasterio
import pandas as pd
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# import json
import pickle
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau


def provide_info_geojson(geotiff_path):
    # Open the GeoTIFF file
    with rasterio.open(geotiff_path) as src:
        # Print basic info about the dataset
        print("Number of bands:", src.count)
        print("Image dimensions (width x height):", src.width, "x", src.height)

        # Read all bands into a numpy array
        data = src.read()
        # data = src.read(indexes=range(2, 8))  # data shape will be (bands, height, width)

        # General metadata
        metadata = src.meta
        print("Metadata:")
        print(metadata)

        # Additional tags metadata (if available)
        tags = src.tags()
        print("\nTags:")
        print(tags)

        for i in range(1, src.count + 1):  # Bands are 1-based in rasterio
            band_name = src.descriptions[i-1] if src.descriptions[i-1] else f"Band {i}"
            print(f"Band {i}: {band_name}")

    # Example: Accessing pixel values at row 100, column 150 (adjust as needed)
    row, col = 1500, 1500
    pixel_values = data[:, row, col]
    print("Pixel values at ({}, {}):".format(row, col), pixel_values)
    


def process_geotiff(config):
    """
    Process a geotiff image with reference panels and measurement points.
    
    Args:
        config: Dictionary containing all configuration parameters
    
    Returns:
        DataFrame with measurement points and their reflectance values
    """
    print(f"Processing {config['image_name']}...")
    
    # --------------------------
    # Step 1: Read Image and Reorder Bands
    # --------------------------
    with rasterio.open(config['geotiff_path']) as src:
        num_bands = src.count
        ordered_bands = src.read().astype(np.float32)

        # Create a mask for missing data (True where all bands = 0)
        missing_data_mask = np.all(ordered_bands == 0, axis=0)

    # --------------------------
    # Step 2: Extract Mean DN Values from Reference Panel Regions (per band)
    # --------------------------
    # Transform panel coordinates from source CRS to image CRS if needed
    panel_crs = config.get('panel_crs', None)
    with rasterio.open(config['geotiff_path']) as src:
        image_crs = src.crs
    if panel_crs and panel_crs != str(image_crs):
        panel_transformer = Transformer.from_crs(panel_crs, image_crs, always_xy=True)
        reflectance_panels = {}
        for ref_value, band_coords in config['reflectance_panels'].items():
            reflectance_panels[ref_value] = {}
            for band, (min_x, min_y, max_x, max_y) in band_coords.items():
                t_min_x, t_min_y = panel_transformer.transform(min_x, min_y)
                t_max_x, t_max_y = panel_transformer.transform(max_x, max_y)
                reflectance_panels[ref_value][band] = (t_min_x, t_min_y, t_max_x, t_max_y)
        print(f"Transformed panel coordinates from {panel_crs} to {image_crs}")
    else:
        reflectance_panels = config['reflectance_panels']

    panel_dn_values = {band: [] for band in range(1, num_bands + 1)}
    panel_reflectance_values = {band: [] for band in range(1, num_bands + 1)}

    with rasterio.open(config['geotiff_path']) as src:
        # Process each band separately
        for band in range(1, num_bands + 1):
            # Process each reflectance panel
            for ref_value, band_coords in reflectance_panels.items():
                # only select reflectance values you are interested in
                if ref_value not in config['included_reflectance_panels']:
                    continue
                # Get bounding box for this specific band
                if band in band_coords:
                    min_x, min_y, max_x, max_y = band_coords[band]
                    
                    # Convert bounding box to pixel coordinates
                    ul_row, ul_col = src.index(min_x, max_y)  # Upper left
                    lr_row, lr_col = src.index(max_x, min_y)  # Lower right
                    
                    # Ensure proper ordering of row/col for slicing
                    min_row, max_row = min(ul_row, lr_row), max(ul_row, lr_row)
                    min_col, max_col = min(ul_col, lr_col), max(ul_col, lr_col)
                    
                    # Extract region of interest
                    region = ordered_bands[band-1, min_row:max_row+1, min_col:max_col+1]
                    
                    # Create a mask for this region (excluding missing data)
                    region_mask = ~missing_data_mask[min_row:max_row+1, min_col:max_col+1]
                    
                    if np.any(region_mask):  # If any valid pixels exist
                        # Calculate mean DN value for this region (excluding missing data)
                        mean_dn = np.mean(region[region_mask])
                        
                        # Store the mean DN value and corresponding reflectance value
                        panel_dn_values[band].append(mean_dn)
                        panel_reflectance_values[band].append(ref_value)
                        
                        print(f"Band {band}, Panel {ref_value}: Mean DN = {mean_dn:.2f}, " 
                              f"Area = {np.sum(region_mask)} pixels, Bounds = ({min_row}:{max_row}, {min_col}:{max_col})")
                        
                        # Optional: Visualize the panel regions
                        if config.get('visualize_panels', False):
                            plt.figure(figsize=(6, 6))
                            plt.imshow(region, cmap='viridis')
                            plt.colorbar(label='DN Value')
                            plt.title(f"Band {band}, Panel {ref_value}")
                            plt.close()
                    else:
                        print(f"Warning: Band {band}, Panel {ref_value} has no valid pixels in the specified region")
                else:
                    print(f"Warning: No bounding box defined for band {band}, panel {ref_value}")

    # --------------------------
    # Step 3: Improved Calibration Models (One per Band)
    # --------------------------
    calibration_models = {}

    plt.figure(figsize=(12, 8))  # Initialize figure for subplots

    for band in range(1, num_bands + 1):
        ax = plt.subplot(2, 4, band)
        
        if len(panel_dn_values[band]) >= 2:  # Need at least 2 points for a line
            dn_array = np.array(panel_dn_values[band]).reshape(-1, 1)
            reflectance_array = np.array(panel_reflectance_values[band])

            # Sort the arrays by DN values for clearer plotting
            sort_idx = np.argsort(dn_array.flatten())
            dn_array = dn_array[sort_idx]
            reflectance_array = reflectance_array[sort_idx]
            
            # Add origin point (0,0) if needed - assume zero DN means zero reflectance
            # Uncomment if appropriate for your sensor
            # dn_array = np.vstack([np.array([[0]]), dn_array])
            # reflectance_array = np.append([0], reflectance_array)
            
            # Fit a linear regression model (R = a * DN + b)
            if config['force_zero_intercept']:
                # Force intercept through origin
                model = LinearRegression(fit_intercept=False)
                model.fit(dn_array, reflectance_array)
                slope = model.coef_[0]
                intercept = 0.0
                calibration_models[band] = (slope, intercept)
                print(f"Band {band} Calibration: Reflectance = {slope:.6f} * DN")
            else:
                # Regular model with intercept
                model = LinearRegression()
                model.fit(dn_array, reflectance_array)
                slope = model.coef_[0]
                intercept = model.intercept_
                calibration_models[band] = (slope, intercept)
                print(f"Band {band} Calibration: Reflectance = {slope:.6f} * DN + {intercept:.6f}")
            
            # Check if parameters are physically reasonable
            if slope <= 0:
                print(f"Warning: Band {band} has negative slope ({slope:.6f}), indicating inverse relationship")
            if intercept < -0.1 or intercept > 0.1:
                print(f"Warning: Band {band} has large intercept ({intercept:.6f}), might indicate calibration issues")
                
            print(f"Band {band} Calibration: Reflectance = {slope:.6f} * DN + {intercept:.6f}")

            # Plot the calibration line for this band
            plt.scatter(dn_array, reflectance_array, color="blue", label="Reference Data")
            
            # Generate x values for line plotting that include 0
            x_min = min(0, np.min(dn_array))
            x_max = np.max(dn_array)
            x_range = np.linspace(x_min - (x_max-x_min)*0.1, x_max + (x_max-x_min)*0.1, 100).reshape(-1, 1)
            plt.plot(x_range, model.predict(x_range), color="red", linestyle="--", label="Linear Fit")
            
            # Add equation text
            eq_text = f"R = {slope:.4f} * DN + {intercept:.4f}"
            plt.text(0.05, 0.9, eq_text, transform=ax.transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8))
            
            # Calculate and add R² value
            r_squared = model.score(dn_array, reflectance_array)
            r2_text = f"R² = {r_squared:.4f}"
            plt.text(0.05, 0.8, r2_text, transform=ax.transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))
                    
            # Force plot to include origin
            plt.xlim(left=min(0, plt.xlim()[0]))
            plt.ylim(bottom=min(0, plt.ylim()[0]))
        else:
            print(f"Warning: Not enough valid points for band {band} calibration")
            plt.text(0.5, 0.5, f"Insufficient data\nfor calibration", 
                     ha='center', transform=ax.transAxes)
            # Use a default model or skip this band
            calibration_models[band] = (1.0, 0.0)  # Default: pass-through

        plt.xlabel("DN Value")
        plt.ylabel("Reflectance")
        plt.title(f"Band {band}")
        plt.grid(True, linestyle='--', alpha=0.7)
        if band == 1:  # Only add legend to first plot
            plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f"2025_analysis/artefacts/calibration_curves_{config['image_name']}.png")
    plt.close()  # Close calibration plots figure
    
    # save calibration_models
    joblib.dump(calibration_models, f"2025_analysis/artefacts/calibration_models_{config['image_name']}.pkl")

    # --------------------------
    # Step 4: Apply Per-Band Calibration to Entire Image
    # --------------------------
    reflectance_data = np.zeros_like(ordered_bands)

    # Apply per-band correction
    for band in range(num_bands):
        if band + 1 in calibration_models:
            a, b = calibration_models[band + 1]  # Get coefficients
            reflectance_data[band] = (a * ordered_bands[band] + b)  # Apply correction
        else:
            reflectance_data[band] = ordered_bands[band]  # No correction available

    # Set missing data pixels to NaN
    reflectance_data[:, missing_data_mask] = np.nan

    # --------------------------
    # Step: Validate Calibration at Reference Panels
    # --------------------------
    validation_results = []

    with rasterio.open(config['geotiff_path']) as src:
        for ref_value, band_coords in reflectance_panels.items():
            if ref_value not in config['included_reflectance_panels']:
                continue

            for band in range(1, num_bands + 1):
                if band in band_coords:
                    min_x, min_y, max_x, max_y = band_coords[band]
                    
                    # Convert to pixel coordinates
                    ul_row, ul_col = src.index(min_x, max_y)
                    lr_row, lr_col = src.index(max_x, min_y)
                    
                    min_row, max_row = min(ul_row, lr_row), max(ul_row, lr_row)
                    min_col, max_col = min(ul_col, lr_col), max(ul_col, lr_col)
                    
                    # Extract calibrated reflectance values
                    region = reflectance_data[band-1, min_row:max_row+1, min_col:max_col+1]
                    region_mask = ~np.isnan(region)
                    
                    if np.any(region_mask):
                        # Calculate mean reflectance after calibration
                        mean_reflectance = np.mean(region[region_mask])
                        
                        # Calculate error
                        error = mean_reflectance - ref_value
                        percent_error = (error / ref_value) * 100
                        
                        validation_results.append({
                            'band': band,
                            'panel': ref_value,
                            'expected': ref_value,
                            'measured': mean_reflectance,
                            'error': error,
                            'percent_error': percent_error
                        })
                        
                        print(f"Band {band}, Panel {ref_value}: "
                              f"Expected={ref_value:.2f}, Measured={mean_reflectance:.4f}, "
                              f"Error={error:.4f} ({percent_error:.2f}%)")

    # Create validation summary dataframe
    validation_df = pd.DataFrame(validation_results)

    # --------------------------
    # Step 5: Extract Reflectance Values for Measurement Points
    # --------------------------
    results = []
    with rasterio.open(config['geotiff_path']) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        for point in config['points']:
            lon, lat = point["lon"], point["lat"]

            # Transform the point's coordinates to the image CRS
            utm_x, utm_y = transformer.transform(lon, lat)

            # Convert the projected coordinates to pixel indices
            row, col = src.index(utm_x, utm_y)
            
            # Check if pixel is within image bounds
            if (0 <= row < src.height and 0 <= col < src.width):
                # Check if pixel is not in missing data area
                if not missing_data_mask[row, col]:
                    # Extract reflectance values at the pixel location
                    reflectance_values = reflectance_data[:, row, col]  # All bands

                    # Store results
                    point_result = {
                        "name": point["name"],
                        "longitude": lon,
                        "latitude": lat,
                        "utm_x": utm_x,
                        "utm_y": utm_y,
                        "row": row,
                        "col": col,
                        "OC": point["OC"]
                    }
                    
                    # Add band values as separate columns
                    for i in range(num_bands):
                        point_result[f'band_{i+1}'] = reflectance_values[i]
                    
                    results.append(point_result)
                    print(f"Point {point['name']}: Valid reflectance values extracted")
                else:
                    print(f"Warning: Point {point['name']} at ({lon}, {lat}) is in missing data area")
            else:
                print(f"Warning: Point {point['name']} at ({lon}, {lat}) is outside image bounds")

    # Create a DataFrame with the collected results
    if results:
        data_VTT = pd.DataFrame(results)
    else:
        print("Warning: No valid measurement points found")
        data_VTT = pd.DataFrame()

    # --------------------------
    # Step 6: Per-Band Histogram of Reflectance Values
    # --------------------------
    plt.figure(figsize=(12, 8))
    for band in range(num_bands):
        plt.subplot(2, 4, band + 1)
        
        # Only include non-NaN values
        valid_values = reflectance_data[band][~np.isnan(reflectance_data[band])]
        
        if len(valid_values) > 0:  # Avoid empty histograms
            plt.hist(valid_values.flatten(), bins=20, color="blue", alpha=0.7)
            
            # Add statistics
            mean_val = np.mean(valid_values)
            median_val = np.median(valid_values)
            plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            plt.axvline(median_val, color='green', linestyle=':', label=f'Median: {median_val:.3f}')
        
        plt.xlabel("Reflectance Value")
        plt.ylabel("Frequency")
        plt.title(f"Band {band + 1}")
        plt.grid(True, alpha=0.3)
        if band == 0:  # Only add legend to first histogram
            plt.legend(fontsize='x-small', loc='upper right')

    plt.tight_layout()
    plt.savefig(f"2025_analysis/artefacts/reflectance_histograms_{config['image_name']}.png")
    plt.close()

    # --------------------------
    # Step 6: Per-Band Histogram of Reflectance Values (Limited to -1 to 1 range)
    # --------------------------
    plt.figure(figsize=(12, 8))
    for band in range(num_bands):
        plt.subplot(2, 4, band + 1)
        
        # Only include non-NaN values
        valid_values = reflectance_data[band][~np.isnan(reflectance_data[band])]
        
        if len(valid_values) > 0:  # Avoid empty histograms
            # Create histogram with limited range from -1 to 1
            plt.hist(valid_values.flatten(), bins=50, color="blue", alpha=0.7, range=(-2, 2))
            
            # Add statistics (calculated from values within the -1 to 1 range)
            filtered_values = valid_values[(valid_values >= -1) & (valid_values <= 1)]
            if len(filtered_values) > 0:
                mean_val = np.mean(filtered_values)
                median_val = np.median(filtered_values)
                plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                plt.axvline(median_val, color='green', linestyle=':', label=f'Median: {median_val:.3f}')
                
                # Add text with percentage of data within range
                in_range_percent = (len(filtered_values) / len(valid_values)) * 100
                plt.text(0.05, 0.95, f'{in_range_percent:.1f}% of data\nin range', 
                         transform=plt.gca().transAxes, fontsize=8,
                         bbox=dict(facecolor='white', alpha=0.8))
        
        plt.xlabel("Reflectance Value")
        plt.ylabel("Frequency")
        plt.title(f"Band {band + 1}")
        plt.xlim(-2, 2)  # Set x-axis limits explicitly
        plt.grid(True, alpha=0.3)
        if band == 0:  # Only add legend to first histogram
            plt.legend(fontsize='x-small', loc='upper right')

    plt.tight_layout()
    plt.savefig(f"2025_analysis/artefacts/reflectance_histograms_reduced_range_{config['image_name']}.png")
    plt.close()

    return data_VTT


def add_spectral_indices(df, band_columns):
    """
    Add spectral indices to a DataFrame that contains per-band reflectance values.

    Computes two sets of features:
      - First-order spectral derivatives: difference between consecutive bands,
        approximating the slope of the spectral curve at each step.
      - Normalised band differences (NBD): (band_j - band_i) / (band_j + band_i)
        for every pair (i < j), analogous to NDVI but applied to all band pairs.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing at least the columns listed in band_columns.
    band_columns : list of str
        Ordered list of band column names (e.g. ['band_1', ..., 'band_6']).

    Returns:
    --------
    pandas.DataFrame
        Copy of df with new index columns appended.
    """
    df = df.copy()
    n = len(band_columns)

    # First-order spectral derivatives  (band_{i+1} - band_i)
    for i in range(n - 1):
        b1, b2 = band_columns[i], band_columns[i + 1]
        df[f'd_{b1}_{b2}'] = df[b2] - df[b1]

    # Normalised band differences for all pairs  (band_j - band_i) / (band_j + band_i)
    for i in range(n):
        for j in range(i + 1, n):
            b1, b2 = band_columns[i], band_columns[j]
            denom = df[b1] + df[b2]
            df[f'nd_{b1}_{b2}'] = np.where(denom != 0, (df[b2] - df[b1]) / denom, np.nan)

    return df


def create_cnn_model(input_shape, filters=None, kernel_size=3, regularization_size=0.0004,
                     fnn_neurons=None, learning_rate=0.002):
    """
    Create a CNN model for regression.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features)
    filters : list, optional
        List of filter configurations, each as [num_filters, use_pooling, use_batch_norm]
    kernel_size : int, default=7
        Size of convolutional kernel
    regularization_size : float, default=0.0004
        L2 regularization parameter
    fnn_neurons : list, optional
        List of neurons in fully connected layers
    learning_rate : float, default=0.002
        Initial learning rate for Adam optimizer
        
    Returns:
    --------
    tf.keras.models.Sequential
        Compiled CNN model
    """
    # Default values if not provided
    if filters is None:
        filters = [[16, False, False], [8, False, False]]

    if fnn_neurons is None:
        fnn_neurons = [32, 16]
    
    model = Sequential()
    
    # Add convolutional layers based on filter configuration
    for i, (filter_size, use_pooling, use_batch_norm) in enumerate(filters):
        if i == 0:
            # First layer needs input_shape
            model.add(Conv1D(
                filters=filter_size, 
                kernel_size=kernel_size, 
                padding='same', 
                input_shape=input_shape,
                kernel_regularizer=l2(regularization_size)
            ))
        else:
            # Subsequent layers
            model.add(Conv1D(
                filters=filter_size, 
                kernel_size=kernel_size, 
                padding='same',
                kernel_regularizer=l2(regularization_size)
            ))
        
        # Add Leaky ReLU activation
        model.add(LeakyReLU(alpha=0.01))
        
        # Add pooling if specified
        if use_pooling:
            model.add(MaxPooling1D(pool_size=2))
            
        # Add batch normalization if specified
        if use_batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
    
    # Flatten the output
    model.add(Flatten())
    
    # Add dense layers
    for neurons in fnn_neurons:
        model.add(Dense(
            neurons, 
            kernel_regularizer=l2(regularization_size)
        ))
        model.add(LeakyReLU(alpha=0.01))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile the model with Adam optimizer and specified learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


def train_cnn_model(data, feature_columns, target_column, n_splits=5,
                   normalize=True, save_model=True, model_name='cnn_model',
                   epochs=200, batch_size=64, patience=50, verbose=1):
    """
    Train a CNN model for regression using k-fold cross-validation.

    Runs k-fold CV to obtain honest evaluation metrics (mean ± std), then
    trains a final model on all data which is saved for prediction.

    Parameters:
    -----------
    data : pandas.DataFrame
    feature_columns : list
    target_column : str
    n_splits : int, default=5
        Number of folds for cross-validation
    normalize : bool, default=True
    save_model : bool, default=True
    model_name : str, default='cnn_model'
    epochs : int, default=200
    batch_size : int, default=64
    patience : int, default=50
    verbose : int, default=1

    Returns:
    --------
    dict with model, history, CV metrics (mean and std), and out-of-fold predictions
    """
    X = data[feature_columns].values
    y = data[target_column].values

    # --- K-fold cross-validation ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores, r2_scores = [], []
    oof_preds = np.zeros(len(y))

    print(f"\nCNN {n_splits}-Fold Cross-Validation:")
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_fold_train, X_fold_test = X[train_idx], X[test_idx]
        y_fold_train, y_fold_test = y[train_idx], y[test_idx]

        fold_scaler = MinMaxScaler(feature_range=(-1, 1)) if normalize else StandardScaler()
        X_fold_train_s = fold_scaler.fit_transform(X_fold_train)
        X_fold_test_s  = fold_scaler.transform(X_fold_test)

        X_fold_train_cnn = X_fold_train_s.reshape(-1, X_fold_train_s.shape[1], 1)
        X_fold_test_cnn  = X_fold_test_s.reshape(-1, X_fold_test_s.shape[1], 1)

        tf.random.set_seed(0)
        fold_model = create_cnn_model((X_fold_train_cnn.shape[1], 1))
        fold_model.fit(
            X_fold_train_cnn, y_fold_train,
            epochs=epochs, batch_size=batch_size, validation_split=0.2,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5)
            ],
            verbose=0
        )

        fold_pred = fold_model.predict(X_fold_test_cnn, verbose=0).flatten()
        oof_preds[test_idx] = fold_pred

        fold_rmse = np.sqrt(mean_squared_error(y_fold_test, fold_pred))
        fold_r2   = r2_score(y_fold_test, fold_pred)
        rmse_scores.append(fold_rmse)
        r2_scores.append(fold_r2)
        print(f"  Fold {fold_idx + 1}: RMSE={fold_rmse:.4f}, R²={fold_r2:.4f}")

    rmse_cv  = float(np.mean(rmse_scores))
    r2_cv    = float(np.mean(r2_scores))
    rmse_std = float(np.std(rmse_scores))
    r2_std   = float(np.std(r2_scores))
    mse_cv   = rmse_cv ** 2
    print(f"  CV mean: RMSE={rmse_cv:.4f} ± {rmse_std:.4f},  R²={r2_cv:.4f} ± {r2_std:.4f}")

    # --- Final model trained on ALL data ---
    print("\nTraining final CNN model on all data...")
    if normalize:
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X)
    if save_model:
        import os as _os
        _model_dir  = _os.path.dirname(f'2025_analysis/{model_name}')
        _model_base = _os.path.basename(model_name)
        joblib.dump(scaler, f'{_model_dir}/scaler_{_model_base}.pkl')
    X_all_cnn = X_all_scaled.reshape(-1, X_all_scaled.shape[1], 1)

    tf.random.set_seed(0)
    model = create_cnn_model((X_all_cnn.shape[1], 1))
    model.summary()
    history = model.fit(
        X_all_cnn, y,
        epochs=epochs, batch_size=batch_size, validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5)
        ],
        verbose=verbose
    )
    if save_model:
        model.save(f'2025_analysis/{model_name}.keras')

    # --- Plot: training history + out-of-fold predictions ---
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('CNN – Final Model Training History')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y, oof_preds, alpha=0.8)
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    axes[1].set_xlabel(f'Actual {target_column}')
    axes[1].set_ylabel(f'Predicted {target_column}')
    axes[1].set_title(f'CNN – Out-of-Fold Predictions ({n_splits}-Fold CV)')
    axes[1].text(0.05, 0.95,
                 f'RMSE = {rmse_cv:.4f} ± {rmse_std:.4f}\nR² = {r2_cv:.4f} ± {r2_std:.4f}',
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.8))
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'2025_analysis/artefacts/training_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'model':    model,
        'history':  history,
        'mse':      mse_cv,
        'rmse':     rmse_cv,
        'r2':       r2_cv,
        'rmse_std': rmse_std,
        'r2_std':   r2_std,
        'oof_preds': oof_preds,
        'y':        y
    }


def train_rf_model(data, feature_columns, target_column, n_splits=5,
                   save_model=True, model_name='rf_model',
                   n_estimators=200, max_depth=None):
    """
    Train a Random Forest model for regression using k-fold cross-validation.

    Runs k-fold CV to obtain honest evaluation metrics (mean ± std), then
    trains a final model on all data which is saved for prediction.

    Parameters:
    -----------
    data : pandas.DataFrame
    feature_columns : list
    target_column : str
    n_splits : int, default=5
    save_model : bool, default=True
    model_name : str, default='rf_model'
    n_estimators : int, default=200
    max_depth : int or None, default=None

    Returns:
    --------
    dict with model, CV metrics (mean and std), and out-of-fold predictions
    """
    X = data[feature_columns].values
    y = data[target_column].values

    # --- K-fold cross-validation ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores, r2_scores = [], []
    oof_preds = np.zeros(len(y))

    print(f"\nRandom Forest {n_splits}-Fold Cross-Validation:")
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        fold_model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
        )
        fold_model.fit(X[train_idx], y[train_idx])
        fold_pred = fold_model.predict(X[test_idx])
        oof_preds[test_idx] = fold_pred

        fold_rmse = np.sqrt(mean_squared_error(y[test_idx], fold_pred))
        fold_r2   = r2_score(y[test_idx], fold_pred)
        rmse_scores.append(fold_rmse)
        r2_scores.append(fold_r2)
        print(f"  Fold {fold_idx + 1}: RMSE={fold_rmse:.4f}, R²={fold_r2:.4f}")

    rmse_cv  = float(np.mean(rmse_scores))
    r2_cv    = float(np.mean(r2_scores))
    rmse_std = float(np.std(rmse_scores))
    r2_std   = float(np.std(r2_scores))
    mse_cv   = rmse_cv ** 2
    print(f"  CV mean: RMSE={rmse_cv:.4f} ± {rmse_std:.4f},  R²={r2_cv:.4f} ± {r2_std:.4f}")

    # --- Final model trained on ALL data ---
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    if save_model:
        joblib.dump(model, f'2025_analysis/{model_name}.pkl')

    # --- Plot: out-of-fold predictions + feature importances ---
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y, oof_preds, alpha=0.8)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    axes[0].set_xlabel(f'Actual {target_column}')
    axes[0].set_ylabel(f'Predicted {target_column}')
    axes[0].set_title(f'Random Forest – Out-of-Fold Predictions ({n_splits}-Fold CV)')
    axes[0].text(0.05, 0.95,
                 f'RMSE = {rmse_cv:.4f} ± {rmse_std:.4f}\nR² = {r2_cv:.4f} ± {r2_std:.4f}',
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.8))
    axes[0].grid(True, alpha=0.3)

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    axes[1].barh(
        [feature_columns[i] for i in top_idx[::-1]],
        importances[top_idx[::-1]]
    )
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Top 15 Feature Importances')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'2025_analysis/artefacts/training_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'model':    model,
        'mse':      mse_cv,
        'rmse':     rmse_cv,
        'r2':       r2_cv,
        'rmse_std': rmse_std,
        'r2_std':   r2_std,
        'oof_preds': oof_preds,
        'y':        y
    }


def train_xgb_model(data, feature_columns, target_column, n_splits=5,
                    save_model=True, model_name='xgb_model',
                    n_estimators=200, learning_rate=0.1, max_depth=4,
                    subsample=0.8, colsample_bytree=0.8):
    """
    Train an XGBoost model for regression using k-fold cross-validation.

    Runs k-fold CV to obtain honest evaluation metrics (mean ± std), then
    trains a final model on all data which is saved for prediction.

    Parameters:
    -----------
    data : pandas.DataFrame
    feature_columns : list
    target_column : str
    n_splits : int, default=5
    save_model : bool, default=True
    model_name : str, default='xgb_model'
    n_estimators : int, default=200
    learning_rate : float, default=0.1
    max_depth : int, default=4
    subsample : float, default=0.8
    colsample_bytree : float, default=0.8

    Returns:
    --------
    dict with model, CV metrics (mean and std), and out-of-fold predictions
    """
    X = data[feature_columns].values
    y = data[target_column].values

    xgb_kwargs = dict(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        subsample=subsample, colsample_bytree=colsample_bytree,
        random_state=42, n_jobs=-1, verbosity=0
    )

    # --- K-fold cross-validation ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores, r2_scores = [], []
    oof_preds = np.zeros(len(y))

    print(f"\nXGBoost {n_splits}-Fold Cross-Validation:")
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        fold_model = XGBRegressor(**xgb_kwargs)
        fold_model.fit(X[train_idx], y[train_idx])
        fold_pred = fold_model.predict(X[test_idx])
        oof_preds[test_idx] = fold_pred

        fold_rmse = np.sqrt(mean_squared_error(y[test_idx], fold_pred))
        fold_r2   = r2_score(y[test_idx], fold_pred)
        rmse_scores.append(fold_rmse)
        r2_scores.append(fold_r2)
        print(f"  Fold {fold_idx + 1}: RMSE={fold_rmse:.4f}, R²={fold_r2:.4f}")

    rmse_cv  = float(np.mean(rmse_scores))
    r2_cv    = float(np.mean(r2_scores))
    rmse_std = float(np.std(rmse_scores))
    r2_std   = float(np.std(r2_scores))
    mse_cv   = rmse_cv ** 2
    print(f"  CV mean: RMSE={rmse_cv:.4f} ± {rmse_std:.4f},  R²={r2_cv:.4f} ± {r2_std:.4f}")

    # --- Final model trained on ALL data ---
    model = XGBRegressor(**xgb_kwargs)
    model.fit(X, y)
    if save_model:
        joblib.dump(model, f'2025_analysis/{model_name}.pkl')

    # --- Plot: out-of-fold predictions + feature importances ---
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y, oof_preds, alpha=0.8)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    axes[0].set_xlabel(f'Actual {target_column}')
    axes[0].set_ylabel(f'Predicted {target_column}')
    axes[0].set_title(f'XGBoost – Out-of-Fold Predictions ({n_splits}-Fold CV)')
    axes[0].text(0.05, 0.95,
                 f'RMSE = {rmse_cv:.4f} ± {rmse_std:.4f}\nR² = {r2_cv:.4f} ± {r2_std:.4f}',
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.8))
    axes[0].grid(True, alpha=0.3)

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    axes[1].barh(
        [feature_columns[i] for i in top_idx[::-1]],
        importances[top_idx[::-1]]
    )
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Top 15 Feature Importances')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'2025_analysis/artefacts/training_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'model':    model,
        'mse':      mse_cv,
        'rmse':     rmse_cv,
        'r2':       r2_cv,
        'rmse_std': rmse_std,
        'r2_std':   r2_std,
        'oof_preds': oof_preds,
        'y':        y
    }


def train_cnn_lofo(field_datasets, feature_columns, target_column,
                   normalize=True, save_final_model=True, model_name='cnn_model',
                   epochs=200, batch_size=8, patience=50, verbose=1):
    """
    Train and evaluate a CNN using Leave-One-Field-Out cross-validation.

    Each fold holds out one complete field as the test set and trains on all
    remaining fields. After evaluation, a final model is trained on all data.

    Parameters:
    -----------
    field_datasets : dict
        Mapping of field name (str) to a DataFrame for that field
    feature_columns : list
        Column names to use as input features
    target_column : str
        Column name of the regression target
    normalize : bool
        Use MinMaxScaler(-1,1) if True, StandardScaler if False
    save_final_model : bool
        Save the final model (trained on all data) and its scaler
    model_name : str
        Base name for saved files
    epochs : int
        Maximum training epochs per fold
    batch_size : int
        Batch size (keep small for small datasets, e.g. 8)
    patience : int
        Early stopping patience
    verbose : int
        Verbosity level for model.fit()

    Returns:
    --------
    dict with fold_results DataFrame, pooled predictions, overall metrics,
    and the final model trained on all data
    """
    field_names = list(field_datasets.keys())
    fold_results = []
    all_y_true, all_y_pred, all_field_labels = [], [], []

    print(f"\n{'='*60}")
    print(f"Leave-One-Field-Out Cross-Validation")
    print(f"Fields: {field_names}")
    print(f"{'='*60}")

    for held_out_field in field_names:
        print(f"\n--- Fold: held-out = '{held_out_field}' ---")

        # Split into held-out test and training from all other fields
        test_df = field_datasets[held_out_field]
        train_df = pd.concat(
            [df for name, df in field_datasets.items() if name != held_out_field],
            ignore_index=True
        )

        X_train = train_df[feature_columns].values
        y_train = train_df[target_column].values
        X_test  = test_df[feature_columns].values
        y_test  = test_df[target_column].values

        print(f"  Training on {len(y_train)} samples, testing on {len(y_test)} samples")

        # Scale features (fit only on training data)
        scaler = MinMaxScaler(feature_range=(-1, 1)) if normalize else StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # Reshape to (samples, bands, 1) for Conv1D
        X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_cnn  = X_test_scaled.reshape(X_test_scaled.shape[0],  X_test_scaled.shape[1],  1)

        # Build a fresh model for each fold
        model = create_cnn_model(input_shape=(X_train_cnn.shape[1], 1))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5, verbose=0),
        ]

        model.fit(
            X_train_cnn, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=verbose
        )

        # Evaluate on the held-out field
        y_pred = model.predict(X_test_cnn, verbose=0).flatten()
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred) if len(y_test) > 1 else float('nan')

        print(f"  RMSE={rmse:.4f}, R²={r2:.4f}")
        fold_results.append({
            'field':  held_out_field,
            'n_test': len(y_test),
            'rmse':   rmse,
            'mse':    mse,
            'r2':     r2,
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_field_labels.extend([held_out_field] * len(y_test))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    results_df   = pd.DataFrame(fold_results)
    overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    overall_r2   = r2_score(all_y_true, all_y_pred)

    print(f"\n{'='*60}")
    print("LOFO Cross-Validation Summary")
    print(results_df.to_string(index=False))
    print(f"\nOverall RMSE : {overall_rmse:.4f}")
    print(f"Overall R²   : {overall_r2:.4f}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    colors      = plt.cm.tab10(np.linspace(0, 1, len(field_names)))
    field_color = {name: colors[i] for i, name in enumerate(field_names)}

    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: predicted vs actual, coloured by field
    for name in field_names:
        idx = [i for i, f in enumerate(all_field_labels) if f == name]
        axes[0].scatter(
            [all_y_true[i] for i in idx],
            [all_y_pred[i] for i in idx],
            label=name, color=field_color[name], s=60, alpha=0.85
        )
    lo, hi = min(all_y_true), max(all_y_true)
    axes[0].plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='1:1 line')
    axes[0].set_xlabel(f'Actual {target_column}')
    axes[0].set_ylabel(f'Predicted {target_column}')
    axes[0].set_title(f'LOFO CV – Predicted vs Actual\nRMSE={overall_rmse:.4f}, R²={overall_r2:.4f}')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: RMSE per fold
    axes[1].bar(results_df['field'], results_df['rmse'],
                color=[field_color[f] for f in results_df['field']])
    axes[1].axhline(overall_rmse, color='red', linestyle='--',
                    label=f'Overall RMSE={overall_rmse:.4f}')
    axes[1].set_xlabel('Held-out Field')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE per Fold')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'2025_analysis/artefacts/lofo_cv_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # Train final model on ALL data combined
    # ------------------------------------------------------------------
    print("\nTraining final model on all data combined...")
    all_data = pd.concat(list(field_datasets.values()), ignore_index=True)
    X_all = all_data[feature_columns].values
    y_all = all_data[target_column].values

    final_scaler = MinMaxScaler(feature_range=(-1, 1)) if normalize else StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    X_all_cnn    = X_all_scaled.reshape(X_all_scaled.shape[0], X_all_scaled.shape[1], 1)

    final_model = create_cnn_model(input_shape=(X_all_cnn.shape[1], 1))
    final_model.fit(
        X_all_cnn, y_all,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5, verbose=0),
        ],
        verbose=verbose
    )

    if save_final_model:
        import os as _os
        _model_dir  = _os.path.dirname(f'2025_analysis/{model_name}')
        _model_base = _os.path.basename(model_name)
        final_model.save(f'2025_analysis/{model_name}.h5')
        joblib.dump(final_scaler, f'{_model_dir}/scaler_{_model_base}.pkl')
        print(f"Final model saved as '2025_analysis/{model_name}.h5'")
        print(f"Scaler saved as '{_model_dir}/scaler_{_model_base}.pkl'")

    return {
        'fold_results':  results_df,
        'y_true':        all_y_true,
        'y_pred':        all_y_pred,
        'field_labels':  all_field_labels,
        'overall_rmse':  overall_rmse,
        'overall_r2':    overall_r2,
        'final_model':   final_model,
        'final_scaler':  final_scaler,
    }


def predict_oc_for_image_cnn(config, model_path='./2025_analysis/oc_prediction_cnn_model.keras',
                         scaler_path='./2025_analysis/scaler_oc_prediction_cnn_model.pkl'):
    """
    Process a GeoTIFF image and predict organic carbon content using a trained CNN model.
    
    Args:
        config: Dictionary containing configuration for the specific image
        model_path: Path to the trained CNN model
        scaler_path: Path to the saved scaler
    """
    image_name = config['image_name']
    geotiff_path = config['geotiff_path']
    points = config.get('points', [])
    
    # Load the calibration models for this specific image
    calibration_file = f'2025_analysis/artefacts/calibration_models_{image_name}.pkl'
    
    print(f"\n{'='*50}")
    print(f"Processing {image_name} image for OC prediction")
    print(f"{'='*50}")
    
    # --------------------------
    # Step 1: Load the trained model, scaler, and calibration models
    # --------------------------
    print("Loading model and preprocessors...")
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load the saved calibration models
    try:
        with open(calibration_file, 'rb') as f:
            calibration_models = pickle.load(f)
        print(f"Loaded calibration models from {calibration_file}")
    except FileNotFoundError:
        print(f"Warning: Calibration file {calibration_file} not found. Check if it exists.")
        return
    
    # --------------------------
    # Step 2: Process the entire image
    # --------------------------
    print(f"Reading and processing {geotiff_path}...")
    with rasterio.open(geotiff_path) as src:
        # Get metadata for output file
        out_meta = src.meta.copy()
        
        # Read all bands
        raw_data = src.read().astype(np.float32)
        num_bands = src.count
    
    # Create a mask for missing data (where all bands are 0)
    missing_data_mask = np.all(raw_data == 0, axis=0)
    
    # Apply band-wise calibration to convert DN to reflectance
    print("Applying band-wise calibration...")
    reflectance_data = np.zeros_like(raw_data)
    for band in range(num_bands):
        if band + 1 in calibration_models:
            a, b = calibration_models[band + 1]  # Get calibration coefficients
            reflectance_data[band] = a * raw_data[band] + b
        else:
            reflectance_data[band] = raw_data[band]  # No calibration available
    
    # Mask missing data
    reflectance_data[:, missing_data_mask] = np.nan
    
    # --------------------------
    # Step 3: Process valid pixels
    # --------------------------
    # Get valid pixels where we have data
    valid_pixels_mask = ~missing_data_mask
    valid_pixels_indices = np.where(valid_pixels_mask)
    
    # Get the number of valid pixels
    num_valid_pixels = len(valid_pixels_indices[0])
    print(f"Processing {num_valid_pixels} valid pixels...")
    
    # Process in batches to avoid memory issues
    batch_size = 10000
    all_predictions = np.full((src.height, src.width), np.nan)
    
    # Processing counter for progress tracking
    total_batches = int(np.ceil(num_valid_pixels / batch_size))
    
    for batch_idx, batch_start in enumerate(range(0, num_valid_pixels, batch_size)):
        batch_end = min(batch_start + batch_size, num_valid_pixels)
        batch_indices = (
            valid_pixels_indices[0][batch_start:batch_end],
            valid_pixels_indices[1][batch_start:batch_end]
        )
        
        # Print progress
        print(f"Processing batch {batch_idx + 1}/{total_batches} ({batch_end - batch_start} pixels)...")
        
        # Extract reflectance values for this batch
        batch_reflectance = np.array([
            reflectance_data[:, y, x]
            for y, x in zip(batch_indices[0], batch_indices[1])
        ])

        # Add spectral indices to match the 26-feature training set
        band_cols = [f'band_{i+1}' for i in range(num_bands)]
        batch_df = pd.DataFrame(batch_reflectance, columns=band_cols)
        batch_df = add_spectral_indices(batch_df, band_cols)
        feature_cols = [c for c in batch_df.columns
                        if c.startswith('band_') or c.startswith('d_') or c.startswith('nd_')]
        batch_features = np.nan_to_num(batch_df[feature_cols].values, nan=0.0)

        # Scale the data using the saved scaler
        batch_scaled = scaler.transform(batch_features)

        # Reshape for CNN input (samples, time_steps, features)
        cnn_input = batch_scaled.reshape(batch_scaled.shape[0], batch_scaled.shape[1], 1)
        
        # Make predictions
        batch_predictions = model.predict(cnn_input, verbose=0)
        
        # Store predictions
        for i, (y, x) in enumerate(zip(batch_indices[0], batch_indices[1])):
            all_predictions[y, x] = batch_predictions[i][0]
    
    # --------------------------
    # Step 4: Save the output as a GeoTIFF
    # --------------------------
    # Update metadata for output
    out_meta.update(
        dtype=rasterio.float32,
        count=1,
        nodata=np.nan
    )
    
    output_path = f"2025_analysis/prediction_results/{image_name}_OC_CNN.tiff"
    print(f"Saving results to {output_path}...")
    
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(all_predictions.astype(rasterio.float32), 1)
    
    print(f"OC prediction map saved as {output_path}")
    
    # --------------------------
    # Step 5: Generate statistics and visualization
    # --------------------------
    # Create a masked array to properly handle NaN values
    masked_oc = np.ma.masked_invalid(all_predictions)
    
    # Calculate statistics
    min_oc = np.nanmin(all_predictions)
    max_oc = np.nanmax(all_predictions)
    mean_oc = np.nanmean(all_predictions)
    median_oc = np.nanmedian(all_predictions)
    
    print(f"OC Prediction Statistics:")
    print(f"  Min: {min_oc:.4f}")
    print(f"  Max: {max_oc:.4f}")
    print(f"  Mean: {mean_oc:.4f}")
    print(f"  Median: {median_oc:.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot histogram
    plt.subplot(2, 1, 1)
    plt.hist(masked_oc.compressed(), bins=50, alpha=0.7, color='blue')
    plt.axvline(mean_oc, color='red', linestyle='--', label=f'Mean: {mean_oc:.4f}')
    plt.axvline(median_oc, color='green', linestyle=':', label=f'Median: {median_oc:.4f}')
    plt.title(f'{image_name} - Histogram of Predicted OC Values')
    plt.xlabel('Organic Carbon (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot the OC map
    plt.subplot(2, 1, 2)
    # Use a viridis colormap with slightly adjusted range for better contrast
    # vmin = max(0, min_oc)  # Ensure minimum is not negative
    # vmax = min(3.0, max_oc * 1.1)  # Cap at 3.0% or 110% of max, whichever is smaller
    vmin = 0.5 
    vmax = 2
    
    plt.imshow(masked_oc, cmap='viridis', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(label='Predicted OC (%)')
    plt.title(f'{image_name} - Predicted Organic Carbon Content')
    plt.axis('off')
    
    # Save the visualization
    plt.tight_layout()
    vis_path = f'2025_analysis/prediction_results/{image_name}_OC_CNN_prediction_map.png'
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved as {vis_path}")
    
    # --------------------------
    # Step 6: Create a validation view if reference points are available
    # --------------------------
    if points:
        try:
            with rasterio.open(geotiff_path) as src:
                # Create a validation map highlighting the reference points
                plt.figure(figsize=(12, 10))
                plt.imshow(masked_oc, cmap='viridis', vmin=vmin, vmax=vmax)
                
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                
                # Add the reference points to the map
                for point in points:
                    lon, lat = point["lon"], point["lat"]
                    utm_x, utm_y = transformer.transform(lon, lat)
                    row, col = src.index(utm_x, utm_y)
                    
                    if (0 <= row < src.height and 0 <= col < src.width):
                        # Draw circles at the reference points
                        circle = Circle((col, row), radius=10, color='red', fill=False, linewidth=2)
                        plt.gca().add_patch(circle)
                        
                        # Add text label with OC value
                        plt.text(col+15, row, f"{point['name']}: {point['OC']}", 
                                color='white', fontsize=9, fontweight='bold', 
                                bbox=dict(facecolor='black', alpha=0.7))
                
                plt.colorbar(label='Predicted OC (%)')
                plt.title(f'{image_name} - Predicted OC with Reference Points')
                plt.axis('off')
                ref_path = f'2025_analysis/prediction_results/{image_name}_OC_CNN_with_reference_points.png'
                plt.savefig(ref_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Validation map saved as {ref_path}")
        except Exception as e:
            print(f"Could not create validation view: {e}")

    return all_predictions


def predict_oc_for_image_rf(config, model_path='./2025_analysis/oc_prediction_rf_model.pkl'):
    """
    Process a GeoTIFF image and predict organic carbon content using a trained Random Forest model.

    Args:
        config: Dictionary containing configuration for the specific image
        model_path: Path to the trained Random Forest model (.pkl)
    """
    image_name = config['image_name']
    geotiff_path = config['geotiff_path']
    points = config.get('points', [])

    calibration_file = f'2025_analysis/artefacts/calibration_models_{image_name}.pkl'

    print(f"\n{'='*50}")
    print(f"Processing {image_name} image for OC prediction (Random Forest)")
    print(f"{'='*50}")

    # --------------------------
    # Step 1: Load the trained model and calibration models
    # --------------------------
    print("Loading model and calibration...")
    model = joblib.load(model_path)

    try:
        with open(calibration_file, 'rb') as f:
            calibration_models = pickle.load(f)
        print(f"Loaded calibration models from {calibration_file}")
    except FileNotFoundError:
        print(f"Warning: Calibration file {calibration_file} not found. Check if it exists.")
        return

    # --------------------------
    # Step 2: Read and calibrate the image
    # --------------------------
    print(f"Reading and processing {geotiff_path}...")
    with rasterio.open(geotiff_path) as src:
        out_meta = src.meta.copy()
        raw_data = src.read().astype(np.float32)
        num_bands = src.count

    missing_data_mask = np.all(raw_data == 0, axis=0)

    print("Applying band-wise calibration...")
    reflectance_data = np.zeros_like(raw_data)
    for band in range(num_bands):
        if band + 1 in calibration_models:
            a, b = calibration_models[band + 1]
            reflectance_data[band] = a * raw_data[band] + b
        else:
            reflectance_data[band] = raw_data[band]

    reflectance_data[:, missing_data_mask] = np.nan

    # --------------------------
    # Step 3: Process valid pixels in batches
    # --------------------------
    valid_pixels_mask = ~missing_data_mask
    valid_pixels_indices = np.where(valid_pixels_mask)
    num_valid_pixels = len(valid_pixels_indices[0])
    print(f"Processing {num_valid_pixels} valid pixels...")

    img_height, img_width = reflectance_data.shape[1], reflectance_data.shape[2]
    band_columns = [f'band_{i+1}' for i in range(num_bands)]
    batch_size = 10000
    all_predictions = np.full((img_height, img_width), np.nan)
    total_batches = int(np.ceil(num_valid_pixels / batch_size))

    for batch_idx, batch_start in enumerate(range(0, num_valid_pixels, batch_size)):
        batch_end = min(batch_start + batch_size, num_valid_pixels)
        batch_indices = (
            valid_pixels_indices[0][batch_start:batch_end],
            valid_pixels_indices[1][batch_start:batch_end]
        )

        print(f"Processing batch {batch_idx + 1}/{total_batches} ({batch_end - batch_start} pixels)...")

        # Extract reflectance values for this batch
        batch_reflectance = np.array([
            reflectance_data[:, y, x]
            for y, x in zip(batch_indices[0], batch_indices[1])
        ])

        # Build DataFrame and add spectral indices (matching training feature set)
        batch_df = pd.DataFrame(batch_reflectance, columns=band_columns)
        batch_df = add_spectral_indices(batch_df, band_columns)
        feature_columns = [c for c in batch_df.columns
                           if c.startswith('band_') or c.startswith('d_') or c.startswith('nd_')]
        batch_features = batch_df[feature_columns].values

        # Replace any NaN (e.g. from nd_ division by zero) with 0
        batch_features = np.nan_to_num(batch_features, nan=0.0)

        batch_predictions = model.predict(batch_features)

        for i, (y, x) in enumerate(zip(batch_indices[0], batch_indices[1])):
            all_predictions[y, x] = batch_predictions[i]

    # --------------------------
    # Step 4: Save the output as a GeoTIFF
    # --------------------------
    out_meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    output_path = f"2025_analysis/prediction_results/{image_name}_OC_RF.tiff"
    print(f"Saving results to {output_path}...")

    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(all_predictions.astype(rasterio.float32), 1)

    print(f"OC RF prediction map saved as {output_path}")

    # --------------------------
    # Step 5: Generate statistics and visualization
    # --------------------------
    masked_oc = np.ma.masked_invalid(all_predictions)
    min_oc = np.nanmin(all_predictions)
    max_oc = np.nanmax(all_predictions)
    mean_oc = np.nanmean(all_predictions)
    median_oc = np.nanmedian(all_predictions)

    print(f"OC Prediction Statistics (RF):")
    print(f"  Min: {min_oc:.4f}")
    print(f"  Max: {max_oc:.4f}")
    print(f"  Mean: {mean_oc:.4f}")
    print(f"  Median: {median_oc:.4f}")

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.hist(masked_oc.compressed(), bins=50, alpha=0.7, color='darkorange')
    plt.axvline(mean_oc, color='red', linestyle='--', label=f'Mean: {mean_oc:.4f}')
    plt.axvline(median_oc, color='green', linestyle=':', label=f'Median: {median_oc:.4f}')
    plt.title(f'{image_name} – Histogram of Predicted OC Values (RF)')
    plt.xlabel('Organic Carbon (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)

    vmin, vmax = 0.5, 2
    plt.subplot(2, 1, 2)
    plt.imshow(masked_oc, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Predicted OC (%)')
    plt.title(f'{image_name} – Predicted Organic Carbon Content (RF)')
    plt.axis('off')

    plt.tight_layout()
    vis_path = f'2025_analysis/prediction_results/{image_name}_OC_RF_prediction_map.png'
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved as {vis_path}")

    # --------------------------
    # Step 6: Validation view with reference points
    # --------------------------
    if points:
        try:
            with rasterio.open(geotiff_path) as src:
                plt.figure(figsize=(12, 10))
                plt.imshow(masked_oc, cmap='viridis', vmin=vmin, vmax=vmax)

                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

                for point in points:
                    lon, lat = point["lon"], point["lat"]
                    utm_x, utm_y = transformer.transform(lon, lat)
                    row, col = src.index(utm_x, utm_y)

                    if (0 <= row < src.height and 0 <= col < src.width):
                        circle = Circle((col, row), radius=10, color='red', fill=False, linewidth=2)
                        plt.gca().add_patch(circle)
                        plt.text(col+15, row, f"{point['name']}: {point['OC']}",
                                 color='white', fontsize=9, fontweight='bold',
                                 bbox=dict(facecolor='black', alpha=0.7))

                plt.colorbar(label='Predicted OC (%)')
                plt.title(f'{image_name} – Predicted OC with Reference Points (RF)')
                plt.axis('off')
                ref_path = f'2025_analysis/prediction_results/{image_name}_OC_RF_with_reference_points.png'
                plt.savefig(ref_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Validation map saved as {ref_path}")
        except Exception as e:
            print(f"Could not create RF validation view: {e}")

    return all_predictions


def predict_oc_for_image_xgb(config, model_path='./2025_analysis/oc_prediction_xgb_model.pkl'):
    """
    Process a GeoTIFF image and predict organic carbon content using a trained XGBoost model.

    Args:
        config: Dictionary containing configuration for the specific image
        model_path: Path to the trained XGBoost model (.pkl)
    """
    image_name = config['image_name']
    geotiff_path = config['geotiff_path']
    points = config.get('points', [])

    calibration_file = f'2025_analysis/artefacts/calibration_models_{image_name}.pkl'

    print(f"\n{'='*50}")
    print(f"Processing {image_name} image for OC prediction (XGBoost)")
    print(f"{'='*50}")

    # --------------------------
    # Step 1: Load the trained model and calibration models
    # --------------------------
    print("Loading model and calibration...")
    model = joblib.load(model_path)

    try:
        with open(calibration_file, 'rb') as f:
            calibration_models = pickle.load(f)
        print(f"Loaded calibration models from {calibration_file}")
    except FileNotFoundError:
        print(f"Warning: Calibration file {calibration_file} not found. Check if it exists.")
        return

    # --------------------------
    # Step 2: Read and calibrate the image
    # --------------------------
    print(f"Reading and processing {geotiff_path}...")
    with rasterio.open(geotiff_path) as src:
        out_meta = src.meta.copy()
        raw_data = src.read().astype(np.float32)
        num_bands = src.count

    missing_data_mask = np.all(raw_data == 0, axis=0)

    print("Applying band-wise calibration...")
    reflectance_data = np.zeros_like(raw_data)
    for band in range(num_bands):
        if band + 1 in calibration_models:
            a, b = calibration_models[band + 1]
            reflectance_data[band] = a * raw_data[band] + b
        else:
            reflectance_data[band] = raw_data[band]

    reflectance_data[:, missing_data_mask] = np.nan

    # --------------------------
    # Step 3: Process valid pixels in batches
    # --------------------------
    valid_pixels_mask = ~missing_data_mask
    valid_pixels_indices = np.where(valid_pixels_mask)
    num_valid_pixels = len(valid_pixels_indices[0])
    print(f"Processing {num_valid_pixels} valid pixels...")

    img_height, img_width = reflectance_data.shape[1], reflectance_data.shape[2]
    band_columns = [f'band_{i+1}' for i in range(num_bands)]
    batch_size = 10000
    all_predictions = np.full((img_height, img_width), np.nan)
    total_batches = int(np.ceil(num_valid_pixels / batch_size))

    for batch_idx, batch_start in enumerate(range(0, num_valid_pixels, batch_size)):
        batch_end = min(batch_start + batch_size, num_valid_pixels)
        batch_indices = (
            valid_pixels_indices[0][batch_start:batch_end],
            valid_pixels_indices[1][batch_start:batch_end]
        )

        print(f"Processing batch {batch_idx + 1}/{total_batches} ({batch_end - batch_start} pixels)...")

        batch_reflectance = np.array([
            reflectance_data[:, y, x]
            for y, x in zip(batch_indices[0], batch_indices[1])
        ])

        batch_df = pd.DataFrame(batch_reflectance, columns=band_columns)
        batch_df = add_spectral_indices(batch_df, band_columns)
        feature_columns = [c for c in batch_df.columns
                           if c.startswith('band_') or c.startswith('d_') or c.startswith('nd_')]
        batch_features = batch_df[feature_columns].values
        batch_features = np.nan_to_num(batch_features, nan=0.0)

        batch_predictions = model.predict(batch_features)

        for i, (y, x) in enumerate(zip(batch_indices[0], batch_indices[1])):
            all_predictions[y, x] = batch_predictions[i]

    # --------------------------
    # Step 4: Save the output as a GeoTIFF
    # --------------------------
    out_meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    output_path = f"2025_analysis/prediction_results/{image_name}_OC_XGB.tiff"
    print(f"Saving results to {output_path}...")

    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(all_predictions.astype(rasterio.float32), 1)

    print(f"OC XGBoost prediction map saved as {output_path}")

    # --------------------------
    # Step 5: Generate statistics and visualization
    # --------------------------
    masked_oc = np.ma.masked_invalid(all_predictions)
    min_oc    = np.nanmin(all_predictions)
    max_oc    = np.nanmax(all_predictions)
    mean_oc   = np.nanmean(all_predictions)
    median_oc = np.nanmedian(all_predictions)

    print(f"OC Prediction Statistics (XGBoost):")
    print(f"  Min: {min_oc:.4f}")
    print(f"  Max: {max_oc:.4f}")
    print(f"  Mean: {mean_oc:.4f}")
    print(f"  Median: {median_oc:.4f}")

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.hist(masked_oc.compressed(), bins=50, alpha=0.7, color='seagreen')
    plt.axvline(mean_oc,   color='red',   linestyle='--', label=f'Mean: {mean_oc:.4f}')
    plt.axvline(median_oc, color='green', linestyle=':',  label=f'Median: {median_oc:.4f}')
    plt.title(f'{image_name} – Histogram of Predicted OC Values (XGBoost)')
    plt.xlabel('Organic Carbon (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)

    vmin, vmax = 0.5, 2
    plt.subplot(2, 1, 2)
    plt.imshow(masked_oc, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Predicted OC (%)')
    plt.title(f'{image_name} – Predicted Organic Carbon Content (XGBoost)')
    plt.axis('off')

    plt.tight_layout()
    vis_path = f'2025_analysis/prediction_results/{image_name}_OC_XGB_prediction_map.png'
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved as {vis_path}")

    # --------------------------
    # Step 6: Validation view with reference points
    # --------------------------
    if points:
        try:
            with rasterio.open(geotiff_path) as src:
                plt.figure(figsize=(12, 10))
                plt.imshow(masked_oc, cmap='viridis', vmin=vmin, vmax=vmax)

                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

                for point in points:
                    lon, lat = point["lon"], point["lat"]
                    utm_x, utm_y = transformer.transform(lon, lat)
                    row, col = src.index(utm_x, utm_y)

                    if (0 <= row < src.height and 0 <= col < src.width):
                        circle = Circle((col, row), radius=10, color='red', fill=False, linewidth=2)
                        plt.gca().add_patch(circle)
                        plt.text(col+15, row, f"{point['name']}: {point['OC']}",
                                 color='white', fontsize=9, fontweight='bold',
                                 bbox=dict(facecolor='black', alpha=0.7))

                plt.colorbar(label='Predicted OC (%)')
                plt.title(f'{image_name} – Predicted OC with Reference Points (XGBoost)')
                plt.axis('off')
                ref_path = f'2025_analysis/prediction_results/{image_name}_OC_XGB_with_reference_points.png'
                plt.savefig(ref_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Validation map saved as {ref_path}")
        except Exception as e:
            print(f"Could not create XGBoost validation view: {e}")

    return all_predictions