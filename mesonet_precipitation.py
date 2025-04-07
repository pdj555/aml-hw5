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
    
    # Hidden layers with appropriate sizes
    x = Dense(256, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(128, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='elu')(x)
    x = BatchNormalization()(x)
    
    x = Dense(32, activation='elu')(x)
    
    # Output layers for the distribution parameters
    # Mean and skewness can be any real value
    mean = Dense(1, activation='linear', name='mean')(x)
    skewness = Dense(1, activation='linear', name='skewness')(x)
    
    # Standard deviation and tailweight must be positive
    std_dev = Dense(1, activation='softplus', name='std_dev')(x)
    tailweight = Dense(1, activation='softplus', name='tailweight')(x)
    
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
    model_outer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
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
    
    # Train the model
    history = model_outer.fit(
        x=ins_training,
        y=outs_training,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(ins_validation, outs_validation),
        verbose=verbose
    )
    
    return model_inner, model_outer, history, (ins_testing, outs_testing)

def predict_distribution_params(model_inner, inputs):
    """
    Predict the distribution parameters using the inner model
    
    Parameters:
    -----------
    model_inner : keras.Model
        The inner model
    inputs : numpy.ndarray
        Input data
        
    Returns:
    --------
    tuple
        mean, std_dev, skewness, tailweight arrays
    """
    return model_inner.predict(inputs)

def calculate_percentiles(model_outer, inputs, percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    """
    Calculate specific percentiles from the predicted distributions
    
    Parameters:
    -----------
    model_outer : keras.Model
        The outer model that produces distributions
    inputs : numpy.ndarray
        Input data
    percentiles : list
        List of percentiles to calculate
        
    Returns:
    --------
    dict
        Dictionary with percentiles and mean
    """
    distributions = model_outer(inputs)
    
    results = {}
    for p in percentiles:
        results[p] = distributions.quantile(p).numpy().flatten()
    
    # Also get the mean
    results['mean'] = distributions.mean().numpy().flatten()
    
    return results

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
        Mean absolute difference
    """
    return np.mean(np.abs(actual - predicted))

def extract_station_data(rotation, station_idx=0):
    """
    Extract time series data for a specific station
    
    Parameters:
    -----------
    rotation : int
        The rotation index
    station_idx : int
        Station index to extract
        
    Returns:
    --------
    tuple
        station_ins, station_outs
    """
    # Load the data
    ins_training, outs_training, ins_validation, outs_validation, ins_testing, outs_testing, _, _, test_nstations = load_data(rotation)
    
    # Extract the station data from test set
    station_ins, station_outs = extract_station_timeseries(
        ins_testing, outs_testing, 
        nstations=test_nstations,
        station_index=station_idx
    )
    
    return station_ins, station_outs

def plot_model_architecture(model_inner, filename='figure0_inner_model_architecture.png'):
    """
    Generate Figure 0: Inner model architecture visualization
    
    Parameters:
    -----------
    model_inner : keras.Model
        The inner model
    filename : str
        Output filename
        
    Returns:
    --------
    str
        Path to the saved figure
    """
    plot_model(model_inner, to_file=filename, 
               show_shapes=True, show_layer_names=True, show_dtype=False)
    return filename

def plot_training_history(all_histories, filename_prefix='figure1'):
    """
    Generate Figures 1a & 1b: Training and validation loss across rotations
    
    Parameters:
    -----------
    all_histories : list
        List of training histories for each rotation
    filename_prefix : str
        Prefix for output filenames
        
    Returns:
    --------
    list
        Paths to the saved figures
    """
    # Figure 1a: Training loss
    plt.figure(figsize=(12, 8))
    for i, history in enumerate(all_histories):
        plt.plot(history.history['loss'], label=f'Rotation {i}')
    plt.title('Training Loss (Negative Log Likelihood)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{filename_prefix}a_training_loss.png')
    plt.close()
    
    # Figure 1b: Validation loss
    plt.figure(figsize=(12, 8))
    for i, history in enumerate(all_histories):
        plt.plot(history.history['val_loss'], label=f'Rotation {i}')
    plt.title('Validation Loss (Negative Log Likelihood)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{filename_prefix}b_validation_loss.png')
    plt.close()
    
    return [f'{filename_prefix}a_training_loss.png', f'{filename_prefix}b_validation_loss.png']

def plot_timeseries_examples(model_outer, station_data, start_idx=0, length=100, filename='figure2_timeseries.png'):
    """
    Generate Figure 2: Time series examples showing observed and predicted values
    
    Parameters:
    -----------
    model_outer : keras.Model
        The outer model
    station_data : tuple
        (station_ins, station_outs)
    start_idx : int
        Starting index for the time series
    length : int
        Number of time points to plot
    filename : str
        Output filename
        
    Returns:
    --------
    str
        Path to the saved figure
    """
    ins, outs = station_data
    
    # Extract a subset of the time series
    ins_subset = ins[start_idx:start_idx+length]
    outs_subset = outs[start_idx:start_idx+length]
    
    # Get distribution percentiles
    percentiles = calculate_percentiles(model_outer, ins_subset)
    
    # Plot the time series
    plt.figure(figsize=(15, 8))
    
    # Observed precipitation
    plt.plot(np.arange(length), outs_subset, 'ko-', label='Observed', alpha=0.7)
    
    # Distribution mean
    plt.plot(np.arange(length), percentiles['mean'], 'r-', label='Mean', linewidth=2)
    
    # Percentiles
    plt.fill_between(
        range(length),
        percentiles[0.1],
        percentiles[0.9],
        alpha=0.3,
        color='blue',
        label='10-90th percentile'
    )
    
    plt.fill_between(
        range(length),
        percentiles[0.25],
        percentiles[0.75],
        alpha=0.4,
        color='green',
        label='25-75th percentile'
    )
    
    plt.title('Precipitation Prediction with Percentiles')
    plt.xlabel('Time (days)')
    plt.ylabel('Precipitation (inches)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()
    
    return filename

def plot_parameter_scatter(all_rotation_data, filename_prefix='figure3'):
    """
    Generate Figures 3a-3d: Scatter plots of distribution parameters vs observed precipitation
    
    Parameters:
    -----------
    all_rotation_data : list
        List of (observed, mean, std, skew, tail) tuples for each rotation
    filename_prefix : str
        Prefix for output filenames
        
    Returns:
    --------
    list
        Paths to the saved figures
    """
    # Unpack the data
    all_observed = []
    all_means = []
    all_stds = []
    all_skewness = []
    all_tailweight = []
    
    for rotation_data in all_rotation_data:
        observed, (mean, std, skew, tail) = rotation_data
        all_observed.extend(observed.flatten())
        all_means.extend(mean.flatten())
        all_stds.extend(std.flatten())
        all_skewness.extend(skew.flatten())
        all_tailweight.extend(tail.flatten())
    
    # Convert to numpy arrays
    all_observed = np.array(all_observed)
    all_means = np.array(all_means)
    all_stds = np.array(all_stds)
    all_skewness = np.array(all_skewness)
    all_tailweight = np.array(all_tailweight)
    
    # Figure 3a: Mean vs Observed
    plt.figure(figsize=(10, 8))
    plt.scatter(all_observed, all_means, alpha=0.5, s=5)
    plt.title('Predicted Mean vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Mean')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{filename_prefix}a_mean.png')
    plt.close()
    
    # Figure 3b: Standard Deviation vs Observed
    plt.figure(figsize=(10, 8))
    plt.scatter(all_observed, all_stds, alpha=0.5, s=5)
    plt.title('Predicted Standard Deviation vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Standard Deviation')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{filename_prefix}b_std.png')
    plt.close()
    
    # Figure 3c: Skewness vs Observed
    plt.figure(figsize=(10, 8))
    plt.scatter(all_observed, all_skewness, alpha=0.5, s=5)
    plt.title('Predicted Skewness vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Skewness')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{filename_prefix}c_skewness.png')
    plt.close()
    
    # Figure 3d: Tailweight vs Observed
    plt.figure(figsize=(10, 8))
    plt.scatter(all_observed, all_tailweight, alpha=0.5, s=5)
    plt.title('Predicted Tailweight vs Observed Precipitation')
    plt.xlabel('Observed Precipitation')
    plt.ylabel('Predicted Tailweight')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{filename_prefix}d_tailweight.png')
    plt.close()
    
    return [f'{filename_prefix}{suffix}.png' for suffix in ['a_mean', 'b_std', 'c_skewness', 'd_tailweight']]

def plot_mad_barplot(mad_results, filename='figure4_mad.png'):
    """
    Generate Figure 4: Bar plot of MAD values for each rotation
    
    Parameters:
    -----------
    mad_results : list
        List of dictionaries with 'median' and 'mean' MAD results for each rotation
    filename : str
        Output filename
        
    Returns:
    --------
    str
        Path to the saved figure
    """
    # Unpack results
    rotations = list(range(len(mad_results)))
    mad_median = [res['median'] for res in mad_results]
    mad_mean = [res['mean'] for res in mad_results]
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(rotations))
    width = 0.35
    
    plt.bar(x - width/2, mad_median, width, label='MAD: Observed vs Median')
    plt.bar(x + width/2, mad_mean, width, label='MAD: Observed vs Mean')
    
    plt.xlabel('Rotation')
    plt.ylabel('Mean Absolute Difference (MAD)')
    plt.title('MAD by Rotation')
    plt.xticks(x, [f'Rotation {r}' for r in rotations])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename)
    plt.close()
    
    return filename

def generate_reflection_template(mad_results, filename='reflection_template.md'):
    """
    Generate a template for the reflection document
    
    Parameters:
    -----------
    mad_results : list
        List of dictionaries with MAD results for each rotation
    filename : str
        Output filename
        
    Returns:
    --------
    str
        Path to the saved file
    """
    # Calculate average MAD across rotations
    avg_mad_median = np.mean([res['median'] for res in mad_results])
    avg_mad_mean = np.mean([res['mean'] for res in mad_results])
    
    reflection = f"""# Reflection: Mesonet Precipitation Prediction

## Model Performance Consistency

The model's performance across the 8 rotations shows an average MAD (Mean Absolute Difference) of {avg_mad_median:.4f} for median predictions and {avg_mad_mean:.4f} for mean predictions.

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
    
    with open(filename, 'w') as f:
        f.write(reflection)
    
    return filename

def run_experiments(n_rotations=8, epochs=500, batch_size=128):
    """
    Run the complete experiment suite across multiple rotations
    
    Parameters:
    -----------
    n_rotations : int
        Number of rotations to run
    epochs : int
        Number of epochs for each training run
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    tuple
        Results of the experiments and generated figures
    """
    # Lists to store results
    all_models = []
    all_histories = []
    all_rotation_data = []
    mad_results = []
    
    # Run for each rotation
    for rotation in range(n_rotations):
        print(f"Starting rotation {rotation}...")
        
        # Train models for this rotation
        model_inner, model_outer, history, test_data = train_models(
            rotation=rotation,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Store the results
        all_models.append((model_inner, model_outer))
        all_histories.append(history)
        
        # Extract test data
        ins_test, outs_test = test_data
        
        # Get distribution parameters
        params = predict_distribution_params(model_inner, ins_test)
        
        # Calculate percentiles
        percentiles = calculate_percentiles(model_outer, ins_test)
        
        # Calculate MADs
        mad_median = calculate_mad(outs_test.flatten(), percentiles[0.5])
        mad_mean = calculate_mad(outs_test.flatten(), percentiles['mean'])
        
        mad_results.append({
            'median': mad_median,
            'mean': mad_mean
        })
        
        # Store data for scatter plots
        all_rotation_data.append((outs_test, params))
        
        print(f"Rotation {rotation} completed.")
        print(f"MAD (Median): {mad_median:.4f}")
        print(f"MAD (Mean): {mad_mean:.4f}")
        print("-" * 50)
    
    # Generate all required figures
    print("Generating figures...")
    
    # Figure 0: Model architecture
    fig0 = plot_model_architecture(all_models[0][0])
    
    # Figures 1a & 1b: Training and validation loss
    fig1 = plot_training_history(all_histories)
    
    # Figure 2: Time series examples (using station from rotation 0)
    station_data = extract_station_data(rotation=0, station_idx=0)
    fig2 = plot_timeseries_examples(all_models[0][1], station_data)
    
    # Generate another time series example with a different start index
    # to show an interesting period
    fig2_alt = plot_timeseries_examples(
        all_models[0][1], 
        station_data, 
        start_idx=100, 
        filename='figure2_timeseries_alt.png'
    )
    
    # Figures 3a-3d: Parameter scatter plots
    fig3 = plot_parameter_scatter(all_rotation_data)
    
    # Figure 4: MAD bar plot
    fig4 = plot_mad_barplot(mad_results)
    
    # Generate reflection template
    reflection = generate_reflection_template(mad_results)
    
    print("All figures generated.")
    print("Experiment completed successfully!")
    
    return all_models, all_histories, all_rotation_data, mad_results, [fig0, fig1, fig2, fig2_alt, fig3, fig4, reflection]

def main():
    """Main function to run the entire experiment"""
    print("Starting Mesonet Precipitation Prediction Experiment")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Disable GPU for compatibility
    tf.config.set_visible_devices([], 'GPU')
    
    # Run the experiments
    all_models, all_histories, all_rotation_data, mad_results, figures = run_experiments(
        n_rotations=8,
        epochs=500,
        batch_size=128
    )
    
    print("=" * 70)
    print("Experiment completed successfully!")
    print("Generated files:")
    for fig in figures:
        if isinstance(fig, list):
            for f in fig:
                print(f"- {f}")
        else:
            print(f"- {fig}")
    
    print("\nPlease complete the reflection document based on your results and observations.")

if __name__ == "__main__":
    main()