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
import argparse

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

# Default plotting parameters
FIGURESIZE = (10, 8)
FONTSIZE = 14

plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Mesonet Rainfall Prediction Using PNN')
    
    # Model architecture
    parser.add_argument('--hidden_layers', type=str, default="64,32,16",
                        help='Comma-separated list of hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for hidden layers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate for optimizer')
    
    # Data parameters
    parser.add_argument('--dataset_path', type=str, 
                        default='/home/fagg/datasets/mesonet/allData1994_2000.csv',
                        help='Path to the Mesonet dataset')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
                        
    # Debug mode
    parser.add_argument('--debug', type=bool, default=False,
                        help='Enable debug mode for additional output and checks')
    
    # Rotation parameter
    parser.add_argument('--rotation', type=int, default=0,
                        help='Rotation index for cross-validation (0-7)')
    
    return parser.parse_args()

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

def safe_plot_histogram(samples, bins=50, title="", xlabel="", actual=None, mean=None, debug=False):
    """
    Safely plot a histogram of samples, handling NaN and Inf values.
    
    Parameters:
    -----------
    samples : array
        Samples to plot
    bins : int
        Number of bins for histogram
    title : str
        Plot title
    xlabel : str
        X-axis label
    actual : float or None
        Actual value to mark with vertical line
    mean : float or None
        Mean value to mark with vertical line
    debug : bool
        Whether to print debug information
    """
    # Check for NaN or Inf values
    if debug:
        print(f"Samples stats for {title}: min={np.nanmin(samples)}, max={np.nanmax(samples)}, "
              f"mean={np.nanmean(samples)}, nan_count={np.isnan(samples).sum()}, "
              f"inf_count={np.isinf(samples).sum()}")
    
    # Filter out NaN and Inf values
    valid_samples = samples[~np.isnan(samples) & ~np.isinf(samples)]
    
    if len(valid_samples) == 0:
        print(f"Warning: No valid samples for {title}, skipping histogram")
        return
    
    # Print percentage of valid samples
    if debug:
        print(f"Valid samples: {len(valid_samples)}/{len(samples)} "
              f"({len(valid_samples)/len(samples)*100:.2f}%)")
    
    # Calculate histogram range with a bit of padding
    sample_min = np.min(valid_samples)
    sample_max = np.max(valid_samples)
    range_width = sample_max - sample_min
    plot_min = max(0, sample_min - 0.1 * range_width)  # Ensure non-negative for rainfall
    plot_max = sample_max + 0.1 * range_width
    
    # If actual and mean are outside the range, expand range
    if actual is not None and (actual < plot_min or actual > plot_max):
        plot_min = min(plot_min, actual * 0.9)
        plot_max = max(plot_max, actual * 1.1)
    
    if mean is not None and (mean < plot_min or mean > plot_max):
        plot_min = min(plot_min, mean * 0.9)
        plot_max = max(plot_max, mean * 1.1)
    
    # Ensure range is valid
    if plot_min >= plot_max:
        plot_max = plot_min + 1.0  # Ensure a minimum range
    
    # Handle extreme values by clipping to a reasonable range
    if plot_max > 1e6:
        if debug:
            print(f"Warning: Extreme max value {plot_max}, clipping to 1000.0")
        plot_max = 1000.0
    
    # Plot histogram with explicit range
    plt.hist(valid_samples, bins=bins, density=True, alpha=0.7, range=(plot_min, plot_max))
    
    if actual is not None:
        plt.axvline(actual, color='red', linestyle='dashed', linewidth=2, label='Actual')
    
    if mean is not None:
        plt.axvline(mean, color='green', linestyle='solid', linewidth=2, label='Mean')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    if actual is not None or mean is not None:
        plt.legend()

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
    
    # Normalize input features to improve numerical stability
    # Compute mean and standard deviation on training data
    feature_mean = np.mean(ins_training, axis=0, keepdims=True)
    feature_std = np.std(ins_training, axis=0, keepdims=True)
    # Avoid division by zero
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    
    # Apply normalization to all datasets
    ins_training = (ins_training - feature_mean) / feature_std
    ins_validation = (ins_validation - feature_mean) / feature_std
    ins_testing = (ins_testing - feature_mean) / feature_std
    
    return (ins_training, outs_training, ins_validation, outs_validation, 
            ins_testing, outs_testing, train_nstations, valid_nstations, test_nstations)

def fully_connected_stack(n_inputs, n_hidden, n_output, activation, activation_out, dropout=None):
    """
    Create a fully connected neural network stack.
    
    Parameters:
    -----------
    n_inputs : int
        Number of input features
    n_hidden : list of int
        Number of units in each hidden layer
    n_output : list of int
        Number of units in each output layer
    activation : str or list of str
        Activation function for hidden layers
    activation_out : list of str
        Activation function for output layers
    dropout : float or None
        Dropout rate, if None no dropout is applied
        
    Returns:
    --------
    input_tensor : tf.Tensor
        Input tensor of the model
    output_tensors : list of tf.Tensor
        Output tensors of the model
    """
    
    # Input tensor
    input_tensor = Input(shape=(n_inputs,))
    
    # Hidden layers
    tensor = input_tensor
    
    # Custom initializer for better numerical stability
    initializer = keras.initializers.GlorotNormal(seed=42)
    
    # Add hidden layers
    for i, n_units in enumerate(n_hidden):
        tensor = Dense(n_units, activation=activation, name=f'hidden_{i}', 
                      kernel_initializer=initializer)(tensor)
        if dropout is not None:
            tensor = Dropout(dropout, name=f'dropout_{i}')(tensor)
    
    # Output layers for the distribution parameters
    # Use a smaller initialization for final layer parameters
    final_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=42)
    
    # Output layers with custom initialization for distribution parameters
    output_tensors = []
    for i, (n_units, act) in enumerate(zip(n_output, activation_out)):
        # For the final layer (distribution parameters), use a smaller initialization
        output_tensors.append(Dense(n_units, activation=act, name=f'output_{i}',
                                 kernel_initializer=final_initializer)(tensor))
    
    return input_tensor, output_tensors

def create_inner_model(n_inputs, hidden_layers, dropout=0.2):
    """
    Create the inner model that produces parameters for the Sinh-Arcsinh distribution
    
    Parameters:
    -----------
    n_inputs : int
        Number of input features
    hidden_layers : list
        List of hidden layer sizes
    dropout : float
        Dropout rate
        
    Returns:
    --------
    keras.Model
        The inner model that outputs the distribution parameters
    """
    # Number of outputs = 4 parameters (loc, scale, skewness, tailweight)
    n_outputs = SinhArcsinh.num_params()
    
    # Build the model using the fully connected stack function
    input_tensor, output_tensors = fully_connected_stack(
        n_inputs=n_inputs,
        n_hidden=hidden_layers,
        n_output=[n_outputs],  # Combined output for all parameters
        activation='elu',
        activation_out=['linear'],
        dropout=dropout
    )
    
    # Create the inner model
    model_inner = Model(inputs=input_tensor, outputs=output_tensors[0])
    
    return model_inner

def create_outer_model(model_inner, learning_rate=0.0005):
    """
    Create the outer model that produces the Sinh-Arcsinh distribution
    
    Parameters:
    -----------
    model_inner : keras.Model
        The inner model that generates distribution parameters
    learning_rate : float
        Learning rate for optimizer
        
    Returns:
    --------
    keras.Model
        The outer model that outputs the distribution
    """
    # Create new input tensor for the outer model
    input_tensor = Input(shape=(model_inner.input.shape[1],))
    
    # Use the inner model to get the distribution parameters
    output_tensor_inner = model_inner(input_tensor)
    
    # Split the output into 4 separate parameter tensors
    loc, scale, skewness, tailweight = tf.split(
        output_tensor_inner, 
        num_or_size_splits=SinhArcsinh.num_params(),
        axis=-1
    )
    
    # Apply constraints to ensure valid parameters
    # Location parameter can be any real number
    loc = tf.clip_by_value(loc, -10.0, 10.0)
    
    # Scale must be positive - use softplus
    scale = 0.1 + tf.math.softplus(scale) * 0.5
    
    # Skewness can be any real number, but clip to reasonable range
    skewness = tf.clip_by_value(skewness, -3.0, 5.0)
    
    # Tailweight must be positive
    tailweight = 0.5 + tf.math.softplus(tailweight) * 0.5
    
    # Create the Sinh-Arcsinh distribution
    dist_params = [loc, scale, skewness, tailweight]
    sas_dist = SinhArcsinh.create_layer()(dist_params)
    
    # Build the outer model
    model_outer = Model(inputs=input_tensor, outputs=sas_dist)
    
    # Compile with negative log likelihood loss
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
        clipvalue=0.5
    )
    model_outer.compile(optimizer=optimizer, loss=SinhArcsinh.mdn_loss)
    
    return model_outer

def run_single_rotation(args):
    """
    Run a single rotation of the model
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    None
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse hidden layers from string to list of integers
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
    
    print(f"Running rotation {args.rotation}")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Output directory: {args.output_dir}")
    
    # Load the dataset
    ins_training, outs_training, ins_validation, outs_validation, ins_testing, outs_testing, \
    train_nstations, valid_nstations, test_nstations = load_data(
        rotation=args.rotation,
        dataset_path=args.dataset_path
    )
    
    print(f"Training data shape: {ins_training.shape}, {outs_training.shape}, {train_nstations} stations")
    print(f"Validation data shape: {ins_validation.shape}, {outs_validation.shape}, {valid_nstations} stations")
    print(f"Test data shape: {ins_testing.shape}, {outs_testing.shape}, {test_nstations} stations")
    
    # Get the number of inputs
    n_inputs = ins_training.shape[1]
    
    # Create models
    model_inner = create_inner_model(
        n_inputs=n_inputs,
        hidden_layers=hidden_layers,
        dropout=args.dropout
    )
    
    model_outer = create_outer_model(
        model_inner=model_inner,
        learning_rate=args.learning_rate
    )
    
    # Display model summaries
    print("Inner model summary:")
    model_inner.summary()
    print("\nOuter model summary:")
    model_outer.summary()
    
    # Save model architecture diagram
    try:
        plot_model(
            model_inner, 
            to_file=os.path.join(args.output_dir, 'model_inner.png'), 
            show_shapes=True
        )
        print(f"Model diagram saved to {os.path.join(args.output_dir, 'model_inner.png')}")
    except Exception as e:
        print(f"Warning: Could not save model diagram. Error: {e}")
    
    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.output_dir, 'model_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    nan_callback = NanLossCallback()
    
    callbacks = [early_stopping, checkpoint, nan_callback]
    
    # Train the model
    history = model_outer.fit(
        x=ins_training,
        y=outs_training,
        validation_data=(ins_validation, outs_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks
    )
    
    # Save the training history
    with open(os.path.join(args.output_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Rotation {args.rotation}: Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Negative Log Likelihood)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    plt.close()
    
    # Evaluate on test set
    test_loss = model_outer.evaluate(ins_testing, outs_testing, verbose=1)
    print(f"Test loss: {test_loss}")
    
    # Save test loss
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f"Model architecture: {hidden_layers}\n")
        f.write(f"Test loss: {test_loss}\n")
    
    # Generate predictions for test data
    # Extract distribution parameters for test data
    output_tensor_inner = model_inner.predict(ins_testing)
    
    # Split the output into 4 separate parameter tensors
    loc, scale, skewness, tailweight = np.split(
        output_tensor_inner, 
        indices_or_sections=SinhArcsinh.num_params(),
        axis=-1
    )
    
    # Extract a station time series for visualization
    station_ins, station_outs = extract_station_timeseries(
        ins=ins_testing, 
        outs=outs_testing, 
        nstations=test_nstations, 
        station_index=0
    )
    
    # Generate predictions for the station
    predicted_distribution = model_outer(station_ins)
    
    # Calculate percentiles
    percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    percentile_values = {}
    for p in percentiles:
        percentile_values[p] = predicted_distribution.quantile(p*tf.ones(station_ins.shape[0], dtype=tf.float32)).numpy().flatten()
    
    # Also get the mean
    mean_pred = predicted_distribution.mean().numpy().flatten()
    
    # Plot station time series with percentiles
    plt.figure(figsize=(15, 8))
    time_index = np.arange(len(station_outs))
    
    # Plot first 100 points for clarity
    plot_length = min(100, len(station_outs))
    
    # Observed precipitation
    plt.plot(time_index[:plot_length], station_outs[:plot_length], 'ko-', label='Observed', alpha=0.7)
    
    # Distribution mean
    plt.plot(time_index[:plot_length], mean_pred[:plot_length], 'r-', label='Mean', linewidth=2)
    
    # 10-90 percentile range
    plt.fill_between(
        time_index[:plot_length],
        percentile_values[0.1][:plot_length],
        percentile_values[0.9][:plot_length],
        color='blue',
        alpha=0.2,
        label='10-90th percentile'
    )
    
    # 25-75 percentile range
    plt.fill_between(
        time_index[:plot_length],
        percentile_values[0.25][:plot_length],
        percentile_values[0.75][:plot_length],
        color='green',
        alpha=0.3,
        label='25-75th percentile'
    )
    
    plt.title(f'Rotation {args.rotation}: Precipitation Time Series')
    plt.xlabel('Time (days)')
    plt.ylabel('Precipitation (mm)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'station_timeseries.png'))
    plt.close()
    
    # Save distribution parameters for the test data
    np.savez(
        os.path.join(args.output_dir, 'test_predictions.npz'),
        loc=loc,
        scale=scale,
        skewness=skewness,
        tailweight=tailweight,
        actual=outs_testing
    )
    
    # Sample a few test examples and visualize the distribution
    n_samples = 5
    sample_indices = np.random.choice(len(station_ins), n_samples, replace=False)
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(sample_indices):
        # Get the input and actual output
        x = station_ins[idx:idx+1]
        actual = station_outs[idx]
        
        # Get the predicted distribution
        pred_dist = model_outer(x)
        
        try:
            # Sample from the distribution
            samples = pred_dist.sample(1000).numpy().flatten()
            
            # Use the safe plotting function
            plt.subplot(n_samples, 1, i+1)
            safe_plot_histogram(
                samples, 
                bins=50, 
                title=f'Precipitation Distribution for Day {idx}',
                xlabel='Precipitation (mm)',
                actual=actual,
                mean=pred_dist.mean().numpy().flatten()[0],
                debug=args.debug
            )
        except Exception as e:
            print(f"Error plotting sample for day {idx}: {e}")
            # Create an empty subplot to maintain the layout
            plt.subplot(n_samples, 1, i+1)
            plt.title(f'Error in prediction for Day {idx}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'prediction_samples.png'))
    plt.close()
    
    # Create scatter plots of distribution parameters vs observed values
    # Use a sample to avoid overcrowding the plot
    sample_size = min(1000, len(outs_testing))
    sample_indices = np.random.choice(len(outs_testing), sample_size, replace=False)
    
    plt.figure(figsize=(16, 14))
    
    # Mean vs Observed
    plt.subplot(2, 2, 1)
    plt.scatter(outs_testing.flatten()[sample_indices], loc.flatten()[sample_indices], alpha=0.5, s=5)
    plt.title('Predicted Mean vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Mean')
    plt.grid(True)
    
    # Scale vs Observed
    plt.subplot(2, 2, 2)
    plt.scatter(outs_testing.flatten()[sample_indices], scale.flatten()[sample_indices], alpha=0.5, s=5)
    plt.title('Predicted Scale vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Scale')
    plt.grid(True)
    
    # Skewness vs Observed
    plt.subplot(2, 2, 3)
    plt.scatter(outs_testing.flatten()[sample_indices], skewness.flatten()[sample_indices], alpha=0.5, s=5)
    plt.title('Predicted Skewness vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Skewness')
    plt.grid(True)
    
    # Tailweight vs Observed
    plt.subplot(2, 2, 4)
    plt.scatter(outs_testing.flatten()[sample_indices], tailweight.flatten()[sample_indices], alpha=0.5, s=5)
    plt.title('Predicted Tailweight vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Tailweight')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'distribution_parameters.png'))
    plt.close()
    
    # Calculate MAD for mean and median
    # Mean MAD
    mean_mad = np.mean(np.abs(outs_testing - loc))
    
    # For median, use the 50th percentile
    # First get the median for all test points
    predicted_distribution_all = model_outer(ins_testing)
    median_pred = predicted_distribution_all.quantile(0.5*tf.ones(ins_testing.shape[0], dtype=tf.float32)).numpy()
    median_mad = np.mean(np.abs(outs_testing - median_pred))
    
    # Save MAD values
    with open(os.path.join(args.output_dir, 'mad_results.txt'), 'w') as f:
        f.write(f"Mean MAD: {mean_mad.item()}\n")
        f.write(f"Median MAD: {median_mad.item()}\n")
    
    print(f"Rotation {args.rotation} completed")
    print(f"Results saved to {args.output_dir}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Run the model for the specified rotation
    run_single_rotation(args)

if __name__ == "__main__":
    main()