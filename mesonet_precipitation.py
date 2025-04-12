"""
Advanced Machine Learning - Mesonet Precipitation Prediction

This script implements a probabilistic neural network to predict precipitation
using the Sinh-Arcsinh distribution.

The implementation follows an inner/outer model architecture:
- The inner model produces parameters for the Sinh-Arcsinh distribution (mean, std dev, skewness, tailweight)
- The outer model creates the actual probability distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import datetime
import pickle

# Import keras (important to use tf_keras for TensorFlow Probability compatibility)
import tf_keras as keras
from tf_keras.models import Sequential, Model
from tf_keras.layers import Dense, BatchNormalization, Dropout, Input
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
from tf_keras.utils import plot_model

# Import the provided support functions
from mesonet_support import get_mesonet_folds, extract_station_timeseries, SinhArcsinh

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Default plotting parameters
FIGURESIZE = (10, 8)
FONTSIZE = 14

plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

class NanLossCallback(keras.callbacks.Callback):
    """
    Callback to handle NaN loss values during training
    """
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            print(f'Batch {batch}: Invalid loss, terminating training')
            self.model.stop_training = True
            
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        
        if (loss is not None and (np.isnan(loss) or np.isinf(loss))) or \
           (val_loss is not None and (np.isnan(val_loss) or np.isinf(val_loss))):
            print(f'Epoch {epoch}: Invalid loss detected, terminating training')
            print(f'Training loss: {loss}, Validation loss: {val_loss}')
            self.model.stop_training = True

def load_data(rotation, dataset_path='/home/fagg/datasets/mesonet/allData1994_2000.csv'):
    """
    Load the Mesonet data for a specific rotation
    
    Parameters:
    -----------
    rotation : int
        The rotation index (0-7)
    dataset_path : str
        Path to the Mesonet dataset
        
    Returns:
    --------
    tuple of numpy arrays: 
        ins_training, outs_training, ins_validation, outs_validation, ins_testing, outs_testing,
        train_nstations, valid_nstations, test_nstations
    """
    # Get the data folds
    ins_training, outs_training, train_nstations, \
    ins_validation, outs_validation, valid_nstations, \
    ins_testing, outs_testing, test_nstations = get_mesonet_folds(
        dataset_fname=dataset_path,
        ntrain_folds=6,
        nvalid_folds=1,
        ntest_folds=1,
        rotation=rotation
    )
    
    return (ins_training, outs_training, ins_validation, outs_validation, 
            ins_testing, outs_testing, train_nstations, valid_nstations, test_nstations)

def create_inner_model(n_inputs):
    """
    Create the inner model that produces parameters for the Sinh-Arcsinh distribution
    
    Parameters:
    -----------
    n_inputs : int
        Number of input features
        
    Returns:
    --------
    keras.Model
        The inner model that outputs the distribution parameters
    """
    # Create input layer
    input_layer = Input(shape=(n_inputs,))
    
    # Batch normalization of inputs (recommended in instructions)
    x = BatchNormalization()(input_layer)
    
    # Custom initializer for better numerical stability
    initializer = keras.initializers.GlorotNormal(seed=42)
    
    # Hidden layers with appropriate sizes
    x = Dense(256, activation='elu', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(128, activation='elu', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='elu', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    
    x = Dense(32, activation='elu', kernel_initializer=initializer)(x)
    
    # Output layers for the distribution parameters
    # Use a smaller initialization for final layer parameters
    final_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=42)
    
    # Mean and skewness can be any real value
    mean = Dense(1, activation='linear', name='mean', kernel_initializer=final_initializer)(x)
    skewness = Dense(1, activation='linear', name='skewness', kernel_initializer=final_initializer)(x)
    
    # Standard deviation and tailweight must be positive
    std_dev = Dense(1, activation='softplus', name='std_dev', kernel_initializer=final_initializer)(x)
    tailweight = Dense(1, activation='softplus', name='tailweight', kernel_initializer=final_initializer)(x)
    
    # Build the model
    model_inner = Model(
        inputs=input_layer, 
        outputs=[mean, std_dev, skewness, tailweight],
        name='inner_model'
    )
    
    return model_inner

def create_outer_model(model_inner):
    """
    Create the outer model that produces the Sinh-Arcsinh distribution
    
    Parameters:
    -----------
    model_inner : keras.Model
        The inner model that generates distribution parameters
        
    Returns:
    --------
    keras.Model
        The outer model that outputs the distribution
    """
    # Input layer with same shape as inner model
    input_layer = Input(shape=model_inner.input.shape[1:])
    
    # Get outputs from inner model
    mean, std_dev, skewness, tailweight = model_inner(input_layer)
    
    # Create the distribution layer using the SinhArcsinh class
    distribution_layer = SinhArcsinh.create_layer()
    distribution = distribution_layer([mean, std_dev, skewness, tailweight])
    
    # Build the model
    model_outer = Model(
        inputs=input_layer, 
        outputs=distribution,
        name='outer_model'
    )
    
    # Compile with negative log likelihood loss
    # Use a more conservative learning rate and add clipnorm for gradient stability
    model_outer.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0005,
            clipnorm=1.0,
            epsilon=1e-7,
            beta_1=0.9,
            beta_2=0.999
        ),
        loss=SinhArcsinh.mdn_loss
    )
    
    return model_outer

def train_models(rotation, epochs=500, batch_size=128, verbose=1):
    """
    Train models for a specific rotation and return results
    
    Parameters:
    -----------
    rotation : int
        The rotation index (0-7)
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    verbose : int
        Verbosity level for training
        
    Returns:
    --------
    tuple
        model_inner, model_outer, history, test_data
    """
    # Load the data
    ins_training, outs_training, ins_validation, outs_validation, ins_testing, outs_testing, _, _, _ = load_data(rotation)
    
    # Get the number of inputs
    n_inputs = ins_training.shape[1]
    
    # Create models
    model_inner = create_inner_model(n_inputs)
    model_outer = create_outer_model(model_inner)
    
    # Create callbacks
    callbacks = [
        # NanLossCallback to handle NaN losses
        NanLossCallback(),
        
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate scheduler - reduce learning rate when plateauing
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    # Train the model
    history = model_outer.fit(
        x=ins_training,
        y=outs_training,
        validation_data=(ins_validation, outs_validation),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks
    )
    
    # Store test data for later evaluation
    test_data = (ins_testing, outs_testing)
    
    return model_inner, model_outer, history, test_data

def predict_distribution_params(model_inner, inputs):
    """
    Predict parameters of the distribution
    
    Parameters:
    -----------
    model_inner : keras.Model
        Inner model that predicts distribution parameters
    inputs : numpy.ndarray
        Input features
        
    Returns:
    --------
    tuple of numpy.ndarray
        mean, std_dev, skewness, tailweight
    """
    return model_inner.predict(inputs)

def calculate_percentiles(model_outer, inputs, percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    """
    Calculate distribution percentiles for given inputs
    
    Parameters:
    -----------
    model_outer : keras.Model
        Outer model that predicts the distribution
    inputs : numpy.ndarray
        Input features
    percentiles : list
        List of percentiles to calculate
        
    Returns:
    --------
    dict
        Dictionary mapping percentiles to predicted values
    """
    # Get the distribution
    distribution = model_outer(inputs)
    
    # Calculate percentiles
    result = {p: distribution.quantile(p*tf.ones(inputs.shape[0], dtype=tf.float32)).numpy().flatten() for p in percentiles}
    
    # Add mean
    result['mean'] = distribution.mean().numpy().flatten()
    
    return result

def calculate_mad(actual, predicted):
    """
    Calculate Mean Absolute Difference
    
    Parameters:
    -----------
    actual : numpy.ndarray
        Actual values
    predicted : numpy.ndarray
        Predicted values
        
    Returns:
    --------
    float
        Mean Absolute Difference
    """
    return np.mean(np.abs(actual - predicted))

def extract_station_data(rotation, station_idx=0):
    """
    Extract time series data for a specific station
    
    Parameters:
    -----------
    rotation : int
        Rotation index
    station_idx : int
        Station index
        
    Returns:
    --------
    tuple
        (station_ins, station_outs)
    """
    # Load all data for this rotation
    ins_training, outs_training, ins_validation, outs_validation, ins_testing, outs_testing, train_nstations, valid_nstations, test_nstations = load_data(rotation)
    
    # Extract the time series for a specific station
    station_ins, station_outs = extract_station_timeseries(ins_testing, outs_testing, test_nstations, station_idx)
    
    return station_ins, station_outs

def save_rotation_data(rotation, model_inner, model_outer, history, test_data, output_dir):
    """
    Save rotation-specific data and plots
    
    Parameters:
    -----------
    rotation : int
        Rotation index
    model_inner : keras.Model
        Inner model
    model_outer : keras.Model
        Outer model
    history : keras.callbacks.History
        Training history
    test_data : tuple
        (ins_test, outs_test)
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    dict
        Dictionary with paths to saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}
    
    # Unpack test data
    ins_test, outs_test = test_data
    
    # Save model architecture
    plot_model(model_inner, to_file=os.path.join(output_dir, 'model_inner.png'), 
               show_shapes=True, show_layer_names=True)
    saved_files['model_inner_diagram'] = os.path.join(output_dir, 'model_inner.png')
    
    # Save training history as plot
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Rotation {rotation}: Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Negative Log Likelihood)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    saved_files['training_history_plot'] = os.path.join(output_dir, 'training_history.png')
    
    # Save training history as pickle
    with open(os.path.join(output_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    saved_files['history_pickle'] = os.path.join(output_dir, 'history.pkl')
    
    # Get distribution parameters for test data
    params = predict_distribution_params(model_inner, ins_test)
    mean, std_dev, skewness, tailweight = params
    
    # Save test parameters
    np.savez(os.path.join(output_dir, 'test_parameters.npz'),
             actual=outs_test,
             mean=mean,
             std_dev=std_dev,
             skewness=skewness,
             tailweight=tailweight)
    saved_files['test_parameters'] = os.path.join(output_dir, 'test_parameters.npz')
    
    # Calculate percentiles
    percentiles = calculate_percentiles(model_outer, ins_test)
    
    # Calculate MAD
    mad_mean = calculate_mad(outs_test.flatten(), percentiles['mean'])
    mad_median = calculate_mad(outs_test.flatten(), percentiles[0.5])
    
    # Save MAD results
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f"MAD (Mean): {mad_mean:.6f}\n")
        f.write(f"MAD (Median): {mad_median:.6f}\n")
    saved_files['results_txt'] = os.path.join(output_dir, 'results.txt')
    
    # Save station time series
    station_data = extract_station_data(rotation, station_idx=0)
    station_ins, station_outs = station_data
    
    # Plot a sample of the time series (first 100 points)
    plt.figure(figsize=(15, 8))
    # Extract a subset of the time series
    length = 100
    ins_subset = station_ins[:length]
    outs_subset = station_outs[:length]
    
    # Get distribution percentiles
    ts_percentiles = calculate_percentiles(model_outer, ins_subset)
    
    # Observed precipitation
    plt.plot(np.arange(length), outs_subset, 'ko-', label='Observed', alpha=0.7)
    
    # Distribution mean
    plt.plot(np.arange(length), ts_percentiles['mean'], 'r-', label='Mean', linewidth=2)
    
    # Percentiles
    plt.fill_between(
        range(length),
        ts_percentiles[0.1],
        ts_percentiles[0.9],
        alpha=0.3,
        color='blue',
        label='10-90th percentile'
    )
    
    plt.fill_between(
        range(length),
        ts_percentiles[0.25],
        ts_percentiles[0.75],
        alpha=0.4,
        color='green',
        label='25-75th percentile'
    )
    
    plt.title(f'Rotation {rotation}: Precipitation Prediction with Percentiles')
    plt.xlabel('Time (days)')
    plt.ylabel('Precipitation (inches)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'station_timeseries.png'))
    plt.close()
    saved_files['station_timeseries'] = os.path.join(output_dir, 'station_timeseries.png')
    
    # Distribution parameter plots (only for a smaller subset for clarity)
    sample_size = min(1000, len(outs_test))
    sample_indices = np.random.choice(len(outs_test), sample_size, replace=False)
    
    # Create 2x2 plot for distribution parameters
    plt.figure(figsize=(16, 14))
    
    # Mean vs Observed
    plt.subplot(2, 2, 1)
    plt.scatter(outs_test.flatten()[sample_indices], mean.flatten()[sample_indices], alpha=0.5, s=5)
    plt.title('Predicted Mean vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Mean')
    plt.grid(True, alpha=0.3)
    
    # Std Dev vs Observed
    plt.subplot(2, 2, 2)
    plt.scatter(outs_test.flatten()[sample_indices], std_dev.flatten()[sample_indices], alpha=0.5, s=5)
    plt.title('Predicted Standard Deviation vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Standard Deviation')
    plt.grid(True, alpha=0.3)
    
    # Skewness vs Observed
    plt.subplot(2, 2, 3)
    plt.scatter(outs_test.flatten()[sample_indices], skewness.flatten()[sample_indices], alpha=0.5, s=5)
    plt.title('Predicted Skewness vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Skewness')
    plt.grid(True, alpha=0.3)
    
    # Tailweight vs Observed
    plt.subplot(2, 2, 4)
    plt.scatter(outs_test.flatten()[sample_indices], tailweight.flatten()[sample_indices], alpha=0.5, s=5)
    plt.title('Predicted Tailweight vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Tailweight')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_parameters.png'))
    plt.close()
    saved_files['distribution_parameters'] = os.path.join(output_dir, 'distribution_parameters.png')
    
    return {
        'mad_mean': mad_mean,
        'mad_median': mad_median,
        'files': saved_files
    }

def generate_combined_analysis(results_dir):
    """
    Generate combined analysis figures from all rotations
    
    Parameters:
    -----------
    results_dir : str
        Directory containing rotation results
        
    Returns:
    --------
    dict
        Dictionary with paths to saved figures
    """
    # Create output directory
    combined_dir = os.path.join(results_dir, 'combined_analysis')
    os.makedirs(combined_dir, exist_ok=True)
    
    # Load data from all rotations
    rotation_data = []
    for i in range(8):
        rotation_dir = os.path.join(results_dir, f'rotation_{i}')
        if os.path.exists(rotation_dir):
            history_file = os.path.join(rotation_dir, 'history.pkl')
            params_file = os.path.join(rotation_dir, 'test_parameters.npz')
            
            if os.path.exists(history_file) and os.path.exists(params_file):
                # Load history
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)
                
                # Load parameters
                params = np.load(params_file)
                
                # Load results file for MAD values
                results_file = os.path.join(rotation_dir, 'results.txt')
                mad_mean = None
                mad_median = None
                
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) >= 2:
                            try:
                                mad_mean = float(lines[0].split(': ')[1])
                                mad_median = float(lines[1].split(': ')[1])
                            except:
                                print(f"Warning: Could not parse MAD values from {results_file}")
                
                rotation_data.append({
                    'rotation': i,
                    'history': history,
                    'params': params,
                    'mad_mean': mad_mean,
                    'mad_median': mad_median
                })
    
    if not rotation_data:
        print("No rotation data found. Make sure all rotations have completed successfully.")
        return {}
    
    print(f"Found data for {len(rotation_data)} rotations")
    saved_figures = {}
    
    # FIGURE 0: Copy the inner model architecture from rotation 0
    model_inner_src = os.path.join(results_dir, 'rotation_0', 'model_inner.png')
    if os.path.exists(model_inner_src):
        model_inner_dest = os.path.join(combined_dir, 'figure0_inner_model_architecture.png')
        import shutil
        shutil.copy(model_inner_src, model_inner_dest)
        saved_figures['figure0'] = model_inner_dest
        print(f"Created Figure 0: {model_inner_dest}")
    else:
        print(f"Warning: Could not find inner model architecture at {model_inner_src}")
    
    # FIGURE 1a: Training loss for all rotations
    plt.figure(figsize=(12, 8))
    for data in rotation_data:
        plt.plot(data['history']['loss'], label=f'Rotation {data["rotation"]}')
    plt.title('Training Loss Across Rotations')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Negative Log Likelihood)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    figure1a_path = os.path.join(combined_dir, 'figure1a_training_loss.png')
    plt.savefig(figure1a_path)
    plt.close()
    saved_figures['figure1a'] = figure1a_path
    print(f"Created Figure 1a: {figure1a_path}")
    
    # FIGURE 1b: Validation loss for all rotations
    plt.figure(figsize=(12, 8))
    for data in rotation_data:
        plt.plot(data['history']['val_loss'], label=f'Rotation {data["rotation"]}')
    plt.title('Validation Loss Across Rotations')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Negative Log Likelihood)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    figure1b_path = os.path.join(combined_dir, 'figure1b_validation_loss.png')
    plt.savefig(figure1b_path)
    plt.close()
    saved_figures['figure1b'] = figure1b_path
    print(f"Created Figure 1b: {figure1b_path}")
    
    # FIGURE 2: Time series visualization
    # Copy from rotation 0 to maintain consistency
    timeseries_src = os.path.join(results_dir, 'rotation_0', 'station_timeseries.png')
    if os.path.exists(timeseries_src):
        # Main figure 2
        figure2_path = os.path.join(combined_dir, 'figure2_timeseries.png')
        import shutil
        shutil.copy(timeseries_src, figure2_path)
        saved_figures['figure2'] = figure2_path
        print(f"Created Figure 2: {figure2_path}")
        
        # Also create an alternative version as in nat's implementation
        figure2_alt_path = os.path.join(combined_dir, 'figure2_timeseries_alt.png')
        shutil.copy(timeseries_src, figure2_alt_path)
        saved_figures['figure2_alt'] = figure2_alt_path
        print(f"Created Figure 2 (alt): {figure2_alt_path}")
    else:
        print(f"Warning: Could not find time series plot at {timeseries_src}")
    
    # FIGURE 3: Distribution parameter scatter plots
    # Combine data from all rotations
    all_observed = []
    all_mean = []
    all_std = []
    all_skewness = []
    all_tailweight = []
    
    for data in rotation_data:
        params = data['params']
        all_observed.append(params['actual'].flatten())
        all_mean.append(params['mean'].flatten())
        all_std.append(params['std_dev'].flatten())
        all_skewness.append(params['skewness'].flatten())
        all_tailweight.append(params['tailweight'].flatten())
    
    # Convert to numpy arrays
    all_observed = np.concatenate(all_observed)
    all_mean = np.concatenate(all_mean)
    all_std = np.concatenate(all_std)
    all_skewness = np.concatenate(all_skewness)
    all_tailweight = np.concatenate(all_tailweight)
    
    # Sample a subset for clarity (max 10,000 points)
    if len(all_observed) > 10000:
        indices = np.random.choice(len(all_observed), 10000, replace=False)
        all_observed = all_observed[indices]
        all_mean = all_mean[indices]
        all_std = all_std[indices]
        all_skewness = all_skewness[indices]
        all_tailweight = all_tailweight[indices]
    
    # FIGURE 3a: Mean vs Observed
    plt.figure(figsize=(10, 8))
    plt.scatter(all_observed, all_mean, alpha=0.3, s=3)
    plt.title('Predicted Mean vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Mean')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    figure3a_path = os.path.join(combined_dir, 'figure3a_mean.png')
    plt.savefig(figure3a_path)
    plt.close()
    saved_figures['figure3a'] = figure3a_path
    print(f"Created Figure 3a: {figure3a_path}")
    
    # FIGURE 3b: Std Dev vs Observed
    plt.figure(figsize=(10, 8))
    plt.scatter(all_observed, all_std, alpha=0.3, s=3)
    plt.title('Predicted Standard Deviation vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Standard Deviation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    figure3b_path = os.path.join(combined_dir, 'figure3b_std.png')
    plt.savefig(figure3b_path)
    plt.close()
    saved_figures['figure3b'] = figure3b_path
    print(f"Created Figure 3b: {figure3b_path}")
    
    # FIGURE 3c: Skewness vs Observed
    plt.figure(figsize=(10, 8))
    plt.scatter(all_observed, all_skewness, alpha=0.3, s=3)
    plt.title('Predicted Skewness vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Skewness')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    figure3c_path = os.path.join(combined_dir, 'figure3c_skewness.png')
    plt.savefig(figure3c_path)
    plt.close()
    saved_figures['figure3c'] = figure3c_path
    print(f"Created Figure 3c: {figure3c_path}")
    
    # FIGURE 3d: Tailweight vs Observed
    plt.figure(figsize=(10, 8))
    plt.scatter(all_observed, all_tailweight, alpha=0.3, s=3)
    plt.title('Predicted Tailweight vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Tailweight')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    figure3d_path = os.path.join(combined_dir, 'figure3d_tailweight.png')
    plt.savefig(figure3d_path)
    plt.close()
    saved_figures['figure3d'] = figure3d_path
    print(f"Created Figure 3d: {figure3d_path}")
    
    # FIGURE 4: MAD comparison bar plot
    # Get MAD values from rotation_data
    mad_mean_values = []
    mad_median_values = []
    rotation_indices = []
    
    for data in rotation_data:
        if data['mad_mean'] is not None and data['mad_median'] is not None:
            mad_mean_values.append(data['mad_mean'])
            mad_median_values.append(data['mad_median'])
            rotation_indices.append(data['rotation'])
    
    if len(mad_mean_values) > 0:
        # Sort by rotation index
        sorted_indices = np.argsort(rotation_indices)
        rotation_indices = [rotation_indices[i] for i in sorted_indices]
        mad_mean_values = [mad_mean_values[i] for i in sorted_indices]
        mad_median_values = [mad_median_values[i] for i in sorted_indices]
        
        # Create bar plot
        plt.figure(figsize=(14, 8))
        
        bar_width = 0.35
        x = np.arange(len(rotation_indices))
        
        plt.bar(x - bar_width/2, mad_mean_values, bar_width, label='MAD: Observed vs Mean')
        plt.bar(x + bar_width/2, mad_median_values, bar_width, label='MAD: Observed vs Median')
        
        plt.xlabel('Rotation')
        plt.ylabel('Mean Absolute Difference (MAD)')
        plt.title('MAD Comparison Across Rotations')
        plt.xticks(x, [f'Rotation {i}' for i in rotation_indices])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        figure4_path = os.path.join(combined_dir, 'figure4_mad.png')
        plt.savefig(figure4_path)
        plt.close()
        saved_figures['figure4'] = figure4_path
        print(f"Created Figure 4: {figure4_path}")
    else:
        print("Warning: No MAD values found for Figure 4")
    
    # Create an updated reflection template using actual MAD values if available
    if len(mad_mean_values) > 0:
        avg_mad_mean = np.mean(mad_mean_values)
        avg_mad_median = np.mean(mad_median_values)
        
        reflection = f"""# Reflection: Mesonet Precipitation Prediction

## Model Performance Consistency

The model's performance across the {len(rotation_indices)} rotations shows an average MAD (Mean Absolute Difference) of {avg_mad_median:.6f} for median predictions and {avg_mad_mean:.6f} for mean predictions.

[Discuss how consistent your model performance is across the different rotations]

## Probability Density Function Shape

[Based on the time-series plots, describe and explain the shape of the probability density function and how it changes over time]

## Skewness Utilization

[How is skewness utilized by the model? Is there a consistent variation in this parameter?]

## Tailweight Utilization

[How is tailweight utilized by the model? Is there a consistent variation in this parameter?]

## Appropriateness of Sinh-Arcsinh Distribution

[Is the Sinh-Arcsinh distribution appropriate for modeling this phenomenon? Provide a detailed explanation]

## Model Effectiveness

[Are your models effective at predicting precipitation? Justify your answer]
"""
    else:
        reflection = """# Reflection: Mesonet Precipitation Prediction

## Model Performance Consistency

[Discuss how consistent your model performance is across the different rotations]

## Probability Density Function Shape

[Based on the time-series plots, describe and explain the shape of the probability density function and how it changes over time]

## Skewness Utilization

[How is skewness utilized by the model? Is there a consistent variation in this parameter?]

## Tailweight Utilization

[How is tailweight utilized by the model? Is there a consistent variation in this parameter?]

## Appropriateness of Sinh-Arcsinh Distribution

[Is the Sinh-Arcsinh distribution appropriate for modeling this phenomenon? Provide a detailed explanation]

## Model Effectiveness

[Are your models effective at predicting precipitation? Justify your answer]
"""
    
    # Save reflection template
    reflection_path = os.path.join(results_dir, 'reflection_template.md')
    with open(reflection_path, 'w') as f:
        f.write(reflection)
    saved_figures['reflection'] = reflection_path
    print(f"Created reflection template: {reflection_path}")
    
    return saved_figures

def run_experiments(n_rotations=8, epochs=500, batch_size=128):
    """
    Run a series of experiments with different rotations
    
    Parameters:
    -----------
    n_rotations : int
        Number of rotations to run
    epochs : int
        Number of epochs for training
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    tuple
        results_dir, all_rotation_results, combined_figures
    """
    print(f"Running {n_rotations} rotations with {epochs} epochs and batch size {batch_size}")
    
    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"rotation_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Store results for each rotation
    all_rotation_results = []
    
    # Run each rotation
    for rotation in range(n_rotations):
        print(f"\n{'='*70}\nRotation {rotation}\n{'='*70}")
        
        # Create directory for this rotation
        rotation_dir = os.path.join(results_dir, f"rotation_{rotation}")
        os.makedirs(rotation_dir, exist_ok=True)
        
        # Train models
        model_inner, model_outer, history, test_data = train_models(
            rotation=rotation,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Create inner model visualization
        try:
            # Generate a visualization of the inner model architecture
            inner_model_path = os.path.join(rotation_dir, "model_inner.png")
            plot_model(model_inner, to_file=inner_model_path, show_shapes=True, show_layer_names=True)
            print(f"Inner model architecture saved to {inner_model_path}")
        except Exception as e:
            print(f"Warning: Could not generate model visualization: {str(e)}")
        
        # Plot training history
        plt.figure(figsize=(12, 8))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss - Rotation {rotation}')
        plt.ylabel('Loss (Negative Log Likelihood)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        history_path = os.path.join(rotation_dir, "training_history.png")
        plt.savefig(history_path)
        plt.close()
        print(f"Training history plot saved to {history_path}")
        
        # Save history for later analysis
        with open(os.path.join(rotation_dir, "history.pkl"), "wb") as f:
            pickle.dump(history.history, f)
        
        # Generate predictions for test data
        ins_test, outs_test = test_data
        
        # Extract distribution parameters for test data
        mean, std_dev, skewness, tailweight = predict_distribution_params(model_inner, ins_test)
        
        # Save parameters for later analysis
        np.savez(
            os.path.join(rotation_dir, "test_parameters.npz"),
            mean=mean,
            std_dev=std_dev,
            skewness=skewness,
            tailweight=tailweight,
            actual=outs_test
        )
        
        # Calculate percentiles for visualization
        distribution_outputs = calculate_percentiles(model_outer, ins_test)
        
        # Calculate MAD for mean and median
        mad_mean = calculate_mad(outs_test, mean)
        mad_median = calculate_mad(outs_test, distribution_outputs['p50'])  # 50th percentile = median
        
        # Save results
        with open(os.path.join(rotation_dir, "results.txt"), "w") as f:
            f.write(f"MAD (Mean): {mad_mean}\n")
            f.write(f"MAD (Median): {mad_median}\n")
        
        # Extract a station time series for visualization
        station_ins, station_outs = extract_station_data(rotation)
        
        # Predict for this station
        station_mean, station_std, station_skew, station_tail = predict_distribution_params(model_inner, station_ins)
        station_percentiles = calculate_percentiles(model_outer, station_ins)
        
        # Plot time series with percentiles
        plt.figure(figsize=(15, 8))
        time_index = np.arange(len(station_outs))
        
        # Plot observed values
        plt.plot(time_index, station_outs, 'k-', label='Observed Precipitation', linewidth=2)
        
        # Plot mean prediction
        plt.plot(time_index, station_mean, 'r-', label='Mean Prediction', linewidth=1.5)
        
        # Plot percentiles
        plt.fill_between(time_index, station_percentiles['p10'], station_percentiles['p90'], 
                         color='blue', alpha=0.2, label='10-90 Percentile Range')
        plt.fill_between(time_index, station_percentiles['p25'], station_percentiles['p75'], 
                         color='blue', alpha=0.3, label='25-75 Percentile Range')
        
        plt.title(f'Precipitation Time Series with Predicted Distribution - Rotation {rotation}')
        plt.xlabel('Time (Days)')
        plt.ylabel('Precipitation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        station_path = os.path.join(rotation_dir, "station_timeseries.png")
        plt.savefig(station_path)
        plt.close()
        print(f"Station time series plot saved to {station_path}")
        
        # Plot distribution parameters against precipitation
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 2, 1)
        plt.scatter(outs_test, mean, alpha=0.3, s=5)
        plt.title('Mean vs Precipitation')
        plt.xlabel('Observed Precipitation')
        plt.ylabel('Predicted Mean')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.scatter(outs_test, std_dev, alpha=0.3, s=5)
        plt.title('Standard Deviation vs Precipitation')
        plt.xlabel('Observed Precipitation')
        plt.ylabel('Predicted Standard Deviation')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.scatter(outs_test, skewness, alpha=0.3, s=5)
        plt.title('Skewness vs Precipitation')
        plt.xlabel('Observed Precipitation')
        plt.ylabel('Predicted Skewness')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.scatter(outs_test, tailweight, alpha=0.3, s=5)
        plt.title('Tailweight vs Precipitation')
        plt.xlabel('Observed Precipitation')
        plt.ylabel('Predicted Tailweight')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        params_path = os.path.join(rotation_dir, "distribution_parameters.png")
        plt.savefig(params_path)
        plt.close()
        print(f"Distribution parameters plot saved to {params_path}")
        
        # Save rotation results
        rotation_result = {
            'rotation': rotation,
            'model_inner': model_inner,
            'model_outer': model_outer,
            'history': history.history,
            'test_data': test_data,
            'mean': mean,
            'std_dev': std_dev,
            'skewness': skewness, 
            'tailweight': tailweight,
            'mad_mean': mad_mean,
            'mad_median': mad_median,
            'percentiles': distribution_outputs
        }
        
        all_rotation_results.append(rotation_result)
        
        print(f"Rotation {rotation} completed")
    
    # Generate combined analysis
    print("\nGenerating combined analysis...")
    combined_figures = generate_combined_analysis(results_dir)
    
    print("All rotations completed")
    return results_dir, all_rotation_results, combined_figures

def main():
    """Main function to run the entire experiment"""
    print("Starting Mesonet Precipitation Prediction Experiment")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run the experiments
    results_dir, all_rotation_results, combined_figures = run_experiments(
        n_rotations=8,  # Run all 8 rotations
        epochs=500,     # Full training for 500 epochs
        batch_size=128
    )
    
    print("=" * 70)
    print("Experiment completed successfully!")
    print("\nGenerated files structure:")
    print(f"- {results_dir}/")
    for i in range(8):  # Updated to show all 8 rotations
        print(f"  - rotation_{i}/")
        print(f"    - model_inner.png")
        print(f"    - training_history.png")
        print(f"    - history.pkl")
        print(f"    - test_parameters.npz")
        print(f"    - results.txt")
        print(f"    - station_timeseries.png")
        print(f"    - distribution_parameters.png")
    print(f"  - combined_analysis/")
    for key, value in combined_figures.items():
        if os.path.basename(value).startswith('figure'):
            print(f"    - {os.path.basename(value)}")
    print(f"  - config.txt")
    print(f"  - reflection_template.md")
    
    print("\nPlease complete the reflection document based on your results and observations.")

if __name__ == "__main__":
    main()