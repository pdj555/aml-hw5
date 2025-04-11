"""
Mesonet Rainfall Prediction Using Probabilistic Neural Networks with Sinh-Arcsinh Distribution
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
# Use tf_keras instead of tensorflow.keras to match mesonet_support.py
import tf_keras as keras
from tf_keras.layers import Input, Dense, Dropout
from tf_keras.models import Model
import os
import argparse
import pickle
# Import the support code
from mesonet_support import get_mesonet_folds, extract_station_timeseries, SinhArcsinh

# Namespace shortcuts
tfd = tfp.distributions
tfpl = tfp.layers

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
    parser.add_argument('--learning_rate', type=float, default=0.001,
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
    
    # Output layers with custom initialization for distribution parameters
    output_tensors = []
    for i, (n_units, act) in enumerate(zip(n_output, activation_out)):
        # For the final layer (distribution parameters), use a smaller initialization
        final_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=42)
        output_tensors.append(Dense(n_units, activation=act, name=f'output_{i}',
                                 kernel_initializer=final_initializer)(tensor))
    
    return input_tensor, output_tensors

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

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print debug information
    if args.debug:
        print("Debug mode enabled")
        print(f"Arguments: {args}")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"NumPy version: {np.__version__}")
    
    # Parse hidden layers from string to list of integers
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
    
    # Step 1: Load the dataset
    # Note: Use provided support function to load and split the dataset
    train_ins, train_outs, train_nstations, valid_ins, valid_outs, valid_nstations, test_ins, test_outs, test_nstations = get_mesonet_folds(
        dataset_fname=args.dataset_path,
        rotation=args.rotation
    )

    print(f"Training data shape: {train_ins.shape}, {train_outs.shape}, {train_nstations} stations")
    print(f"Validation data shape: {valid_ins.shape}, {valid_outs.shape}, {valid_nstations} stations")
    print(f"Test data shape: {test_ins.shape}, {test_outs.shape}, {test_nstations} stations")
    
    # Normalize input features to improve numerical stability
    # Compute mean and standard deviation on training data
    feature_mean = np.mean(train_ins, axis=0, keepdims=True)
    feature_std = np.std(train_ins, axis=0, keepdims=True)
    # Avoid division by zero
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    
    # Apply normalization to all datasets
    train_ins = (train_ins - feature_mean) / feature_std
    valid_ins = (valid_ins - feature_mean) / feature_std
    test_ins = (test_ins - feature_mean) / feature_std
    
    # For outputs (rainfall), a log-transform might be appropriate since it's skewed
    # Add a small constant before taking the log to handle zeros
    epsilon = 1e-5
    train_outs_log = np.log(train_outs + epsilon)
    valid_outs_log = np.log(valid_outs + epsilon)
    test_outs_log = np.log(test_outs + epsilon)
    
    # Compute statistics for the log-transformed outputs
    output_mean = np.mean(train_outs_log)
    output_std = np.std(train_outs_log)
    
    # Store original outputs for evaluation
    train_outs_original = train_outs.copy()
    valid_outs_original = valid_outs.copy()
    test_outs_original = test_outs.copy()
    
    # Decide whether to use log-transformed outputs
    use_log_transform = True
    if use_log_transform:
        train_outs = train_outs_log
        valid_outs = valid_outs_log
        test_outs = test_outs_log
    
    # Check for NaN values in the data
    if args.debug:
        print(f"Training inputs NaN count: {np.isnan(train_ins).sum()}")
        print(f"Training outputs NaN count: {np.isnan(train_outs).sum()}")
        print(f"Validation inputs NaN count: {np.isnan(valid_ins).sum()}")
        print(f"Validation outputs NaN count: {np.isnan(valid_outs).sum()}")
        print(f"Test inputs NaN count: {np.isnan(test_ins).sum()}")
        print(f"Test outputs NaN count: {np.isnan(test_outs).sum()}")
        
        # Print data statistics
        print(f"Training outputs: min={np.min(train_outs)}, max={np.max(train_outs)}, mean={np.mean(train_outs)}")
        print(f"Validation outputs: min={np.min(valid_outs)}, max={np.max(valid_outs)}, mean={np.mean(valid_outs)}")
        print(f"Test outputs: min={np.min(test_outs)}, max={np.max(test_outs)}, mean={np.mean(test_outs)}")

    # Step 2: Build the inner model that produces Sinh-Arcsinh parameters
    # The input dimension is the number of features in the dataset
    n_inputs = train_ins.shape[1]

    # The output is 1-dimensional rainfall
    d_output = 1

    # The Sinh-Arcsinh distribution has 4 parameters:
    # loc (location), scale, skewness, and tailweight
    # For each parameter, we need d_output values
    n_outputs = SinhArcsinh.num_params() * d_output

    # Define the neural network architecture
    input_tensor, output_tensors = fully_connected_stack(
        n_inputs=n_inputs,
        n_hidden=hidden_layers,
        n_output=[n_outputs],
        activation='elu',
        activation_out=['linear'],
        dropout=args.dropout
    )

    # Create the inner model
    model_inner = keras.models.Model(inputs=input_tensor, outputs=output_tensors[0])

    # Step 3: Create the outer model that uses the Sinh-Arcsinh distribution
    # Create new input tensor for the outer model
    input_tensor_outer = Input(shape=(n_inputs,))

    # Use the inner model to get the distribution parameters
    output_tensor_inner = model_inner(input_tensor_outer)

    # Split the output into 4 separate parameter tensors
    # Each parameter has d_output (1) values
    loc, scale, skewness, tailweight = tf.split(
        output_tensor_inner, 
        num_or_size_splits=SinhArcsinh.num_params(),
        axis=-1
    )

    # Apply constraints to ensure valid parameters
    # Use special initialization for stability with larger networks
    
    # Location parameter can be any real number, but initialize near 0
    # For rainfall, we expect this to be positive for the most part
    # But we'll allow negative values and let the model learn
    loc = tf.clip_by_value(loc, -10.0, 10.0)
    
    # Scale must be positive - use softplus with appropriate offset
    # Start with small scale values (concentrated distribution)
    scale = 0.1 + tf.math.softplus(scale) * 0.5  # smaller initial scale 
    
    # Skewness: for rainfall we expect positive skew (long tail to the right)
    # Initialize with slight positive bias
    skewness = tf.clip_by_value(skewness, -3.0, 5.0)
    
    # Tailweight controls kurtosis, higher values = heavier tails
    # Initialize with moderate values
    tailweight = 0.5 + tf.math.softplus(tailweight) * 0.5
    
    # Create the Sinh-Arcsinh distribution
    dist_params = [loc, scale, skewness, tailweight]
    sas_dist = SinhArcsinh.create_layer()(dist_params)

    # Build the outer model
    model_outer = keras.models.Model(inputs=input_tensor_outer, outputs=sas_dist)

    # Step 4: Compile the model using the Sinh-Arcsinh loss function
    # Use a custom loss function that computes negative log likelihood
    optimizer = keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        clipnorm=1.0,  # Clip gradients by norm
        clipvalue=0.5  # Clip gradients by value
    )
    model_outer.compile(optimizer=optimizer, loss=SinhArcsinh.mdn_loss)

    # Display model summaries
    print("Inner model summary:")
    model_inner.summary()
    print("\nOuter model summary:")
    model_outer.summary()

    # Save model architecture diagram
    try:
        keras.utils.plot_model(
            model_inner, 
            to_file=os.path.join(args.output_dir, 'model_inner.png'), 
            show_shapes=True
        )
        keras.utils.plot_model(
            model_outer, 
            to_file=os.path.join(args.output_dir, 'model_outer.png'), 
            show_shapes=True
        )
    except Exception as e:
        print(f"Warning: Could not save model diagrams. Error: {e}")

    # Step 5: Train the model
    # Define early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Create a model checkpoint callback
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.output_dir, 'model_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Add NaN checking callback
    class NanLossCallback(keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            logs = logs or {}
            loss = logs.get('loss')
            if loss is not None and (np.isnan(loss) or np.isinf(loss)):
                print(f'Batch {batch}: Invalid loss, terminating training')
                self.model.stop_training = True
    
    nan_callback = NanLossCallback()
    
    # Set up callbacks
    callbacks = [early_stopping, checkpoint]
    if args.debug:
        callbacks.append(nan_callback)

    # Train the model
    history = model_outer.fit(
        train_ins, train_outs,
        validation_data=(valid_ins, valid_outs),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Step 6: Evaluate the model on test set
    test_loss = model_outer.evaluate(test_ins, test_outs, verbose=0)
    print(f"Test loss: {test_loss}")
    
    # Save the test loss to a text file
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f"Model architecture: {hidden_layers}\n")
        f.write(f"Test loss: {test_loss}\n")

    # Step 7: Visualize the training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "training_history.png"))
    plt.close()

    # Step 8: Visualize predictions for a single station
    # Extract data for a single station
    station_index = 0  # Use the first station for demonstration
    station_test_ins, station_test_outs = extract_station_timeseries(
        test_ins, test_outs, test_nstations, station_index
    )

    # Generate predictions
    predicted_distribution = model_outer(station_test_ins)

    # Extract distribution parameters
    loc_pred = predicted_distribution.parameters['loc'].numpy()
    scale_pred = predicted_distribution.parameters['scale'].numpy()
    skewness_pred = predicted_distribution.parameters['skewness'].numpy()
    tailweight_pred = predicted_distribution.parameters['tailweight'].numpy()
    
    # Check for NaN values in the parameters
    if args.debug:
        print(f"loc_pred NaN count: {np.isnan(loc_pred).sum()}")
        print(f"scale_pred NaN count: {np.isnan(scale_pred).sum()}")
        print(f"skewness_pred NaN count: {np.isnan(skewness_pred).sum()}")
        print(f"tailweight_pred NaN count: {np.isnan(tailweight_pred).sum()}")
        
        # Print parameter statistics
        print(f"loc_pred: min={np.nanmin(loc_pred)}, max={np.nanmax(loc_pred)}, mean={np.nanmean(loc_pred)}")
        print(f"scale_pred: min={np.nanmin(scale_pred)}, max={np.nanmax(scale_pred)}, mean={np.nanmean(scale_pred)}")
        print(f"skewness_pred: min={np.nanmin(skewness_pred)}, max={np.nanmax(skewness_pred)}, mean={np.nanmean(skewness_pred)}")
        print(f"tailweight_pred: min={np.nanmin(tailweight_pred)}, max={np.nanmax(tailweight_pred)}, mean={np.nanmean(tailweight_pred)}")

    # Calculate prediction intervals (mean ± 2*scale as a simple approximation)
    # Replace NaN values with 0 for plotting
    mean_pred = np.nan_to_num(loc_pred, nan=0.0)
    scale_pred_clean = np.nan_to_num(scale_pred, nan=0.01)  # Small positive value for NaNs
    lower_bound = mean_pred - 2 * scale_pred_clean
    upper_bound = mean_pred + 2 * scale_pred_clean

    # Visualize the results
    plt.figure(figsize=(12, 6))
    # Plot actual values
    plt.scatter(range(len(station_test_outs)), station_test_outs, color='blue', label='Actual Rainfall', alpha=0.7)
    # Plot predicted mean
    plt.plot(range(len(mean_pred)), mean_pred, color='red', label='Predicted Mean')
    # Plot prediction intervals
    plt.fill_between(range(len(mean_pred)), lower_bound.flatten(), upper_bound.flatten(), 
                    color='red', alpha=0.2, label='Prediction Interval')

    plt.xlabel('Day')
    plt.ylabel('Rainfall (mm)')
    plt.title(f'Rainfall Prediction for Station {station_index}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "station_prediction.png"))
    plt.close()

    # Step 9: Generate and visualize samples from the predicted distribution
    # Select a few data points to visualize
    n_points = 5
    sample_indices = np.random.choice(len(station_test_ins), n_points, replace=False)

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(sample_indices):
        # Get the input and actual output
        x = station_test_ins[idx:idx+1]
        actual = station_test_outs[idx]
        
        # Get the predicted distribution
        pred_dist = model_outer(x)
        
        try:
            # Sample from the distribution
            samples = pred_dist.sample(1000).numpy().flatten()
            
            # Check for NaN values
            if np.isnan(samples).any() or np.isinf(samples).any():
                if args.debug:
                    print(f"Warning: NaN or Inf values in samples for day {idx}")
                
                # Use the safe plotting function
                plt.subplot(n_points, 1, i+1)
                safe_plot_histogram(
                    samples, 
                    bins=50, 
                    title=f'Prediction Distribution for Day {idx}',
                    xlabel='Rainfall (mm)',
                    actual=actual,
                    mean=pred_dist.mean().numpy().flatten()[0] if not np.isnan(pred_dist.mean().numpy()).any() else None,
                    debug=args.debug
                )
            else:
                # No NaN values, use regular plotting
                plt.subplot(n_points, 1, i+1)
                plt.hist(samples, bins=50, density=True, alpha=0.7)
                plt.axvline(actual, color='red', linestyle='dashed', linewidth=2, label='Actual')
                plt.axvline(pred_dist.mean().numpy().flatten()[0], color='green', linestyle='solid', linewidth=2, label='Mean')
                plt.title(f'Prediction Distribution for Day {idx}')
                plt.xlabel('Rainfall (mm)')
                plt.ylabel('Density')
                plt.legend()
        except Exception as e:
            print(f"Error plotting sample for day {idx}: {e}")
            # Create an empty subplot to maintain the layout
            plt.subplot(n_points, 1, i+1)
            plt.title(f'Error in prediction for Day {idx}')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "prediction_samples.png"))
    plt.close()

    # Step 10: Analyze the skewness and tailweight parameters
    plt.figure(figsize=(12, 10))

    # Plot skewness
    plt.subplot(2, 1, 1)
    safe_plot_histogram(
        skewness_pred.flatten(), 
        bins=30, 
        title='Distribution of Skewness Parameter',
        xlabel='Skewness Parameter',
        debug=args.debug
    )
    plt.grid(True)

    # Plot tailweight
    plt.subplot(2, 1, 2)
    safe_plot_histogram(
        tailweight_pred.flatten(), 
        bins=30, 
        title='Distribution of Tailweight Parameter',
        xlabel='Tailweight Parameter',
        debug=args.debug
    )
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "distribution_parameters.png"))
    plt.close()

    # Save distribution parameters for the single station
    np.savez(
        os.path.join(args.output_dir, "station_parameters.npz"),
        loc=loc_pred,
        scale=scale_pred,
        skewness=skewness_pred,
        tailweight=tailweight_pred,
        actual=station_test_outs
    )

    # Save the training history
    with open(os.path.join(args.output_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

    # After generating predictions:
    # Save the predictions for combined analysis
    np.savez(
        os.path.join(args.output_dir, 'test_predictions.npz'),
        loc=loc_pred,
        scale=scale_pred,
        skewness=skewness_pred,
        tailweight=tailweight_pred,
        actual=station_test_outs
    )

    print("PNN Rainfall Prediction Model Summary:")
    print("--------------------------------------")
    print(f"Number of features: {n_inputs}")
    print(f"Model architecture: {hidden_layers}")
    print(f"Distribution: Sinh-Arcsinh with {SinhArcsinh.num_params()} parameters")
    print(f"Test loss: {test_loss}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 