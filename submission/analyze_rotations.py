"""
Create combined figures from all rotations for the Mesonet Rainfall PNN project.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tf_keras.models import load_model
import tensorflow_probability as tfp

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze rotation results from Mesonet PNN model')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing all rotation results')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save combined analysis figures')
    return parser.parse_args()

def load_rotation_data(results_dir):
    """Load data from all rotations."""
    rotations = []
    for i in range(8):
        rotation_dir = os.path.join(results_dir, f'rotation_{i}')
        if os.path.exists(rotation_dir):
            # Load history
            history_file = os.path.join(rotation_dir, 'history.pkl')
            if os.path.exists(history_file):
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)
                
                # Load predictions
                predictions_file = os.path.join(rotation_dir, 'test_predictions.npz')
                if os.path.exists(predictions_file):
                    predictions = np.load(predictions_file)
                    
                    # Load MAD results if available
                    mad_file = os.path.join(rotation_dir, 'mad_results.txt')
                    mean_mad = None
                    median_mad = None
                    
                    if os.path.exists(mad_file):
                        with open(mad_file, 'r') as f:
                            lines = f.readlines()
                            if len(lines) >= 2:
                                try:
                                    mean_mad = float(lines[0].split(': ')[1])
                                    median_mad = float(lines[1].split(': ')[1])
                                except:
                                    print(f"Warning: Could not parse MAD values from {mad_file}")
                    
                    rotations.append({
                        'rotation': i,
                        'history': history,
                        'predictions': predictions,
                        'mean_mad': mean_mad,
                        'median_mad': median_mad
                    })
    
    return rotations

def plot_training_curves(rotations, output_dir):
    """Create Figure 1a,b: Training and validation loss curves for all rotations."""
    plt.figure(figsize=(12, 10))
    
    # Figure 1a: Training loss
    plt.subplot(2, 1, 1)
    for rotation in rotations:
        plt.plot(rotation['history']['loss'], label=f'Rotation {rotation["rotation"]}')
    plt.title('Training Loss Across Rotations')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood')
    plt.grid(True)
    plt.legend()
    
    # Figure 1b: Validation loss
    plt.subplot(2, 1, 2)
    for rotation in rotations:
        plt.plot(rotation['history']['val_loss'], label=f'Rotation {rotation["rotation"]}')
    plt.title('Validation Loss Across Rotations')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure1_training_curves.png'), dpi=300)
    plt.close()

def plot_parameter_scatter(rotations, output_dir):
    """Create Figure 3a,b,c,d: Scatter plots of distribution parameters vs. observed precipitation."""
    # Combine data from all rotations
    all_actual = []
    all_loc = []
    all_scale = []
    all_skewness = []
    all_tailweight = []
    
    for rotation in rotations:
        predictions = rotation['predictions']
        all_actual.append(predictions['actual'])
        all_loc.append(predictions['loc'])
        all_scale.append(predictions['scale'])
        all_skewness.append(predictions['skewness'])
        all_tailweight.append(predictions['tailweight'])
    
    # Convert to flat arrays
    all_actual = np.concatenate(all_actual).flatten()
    all_loc = np.concatenate(all_loc).flatten()
    all_scale = np.concatenate(all_scale).flatten()
    all_skewness = np.concatenate(all_skewness).flatten()
    all_tailweight = np.concatenate(all_tailweight).flatten()
    
    # Sample a subset for clearer visualization
    if len(all_actual) > 10000:
        sample_indices = np.random.choice(len(all_actual), 10000, replace=False)
        all_actual = all_actual[sample_indices]
        all_loc = all_loc[sample_indices]
        all_scale = all_scale[sample_indices]
        all_skewness = all_skewness[sample_indices]
        all_tailweight = all_tailweight[sample_indices]
    
    # Create scatter plots
    plt.figure(figsize=(16, 14))
    
    # Figure 3a: Mean vs. Observed
    plt.subplot(2, 2, 1)
    plt.scatter(all_actual, all_loc, alpha=0.3, s=10)
    plt.title('Mean vs. Observed Precipitation')
    plt.xlabel('Observed Precipitation (mm)')
    plt.ylabel('Predicted Mean (mm)')
    plt.grid(True)
    
    # Figure 3b: Scale vs. Observed
    plt.subplot(2, 2, 2)
    plt.scatter(all_actual, all_scale, alpha=0.3, s=10)
    plt.title('Scale vs. Observed Precipitation')
    plt.xlabel('Observed Precipitation (mm)')
    plt.ylabel('Predicted Scale')
    plt.grid(True)
    
    # Figure 3c: Skewness vs. Observed
    plt.subplot(2, 2, 3)
    plt.scatter(all_actual, all_skewness, alpha=0.3, s=10)
    plt.title('Skewness vs. Observed Precipitation')
    plt.xlabel('Observed Precipitation (mm)')
    plt.ylabel('Predicted Skewness')
    plt.grid(True)
    
    # Figure 3d: Tailweight vs. Observed
    plt.subplot(2, 2, 4)
    plt.scatter(all_actual, all_tailweight, alpha=0.3, s=10)
    plt.title('Tailweight vs. Observed Precipitation')
    plt.xlabel('Observed Precipitation (mm)')
    plt.ylabel('Predicted Tailweight')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure3_parameter_scatter.png'), dpi=300)
    plt.close()

def plot_mad_comparison(rotations, output_dir):
    """Create Figure 4: MAD comparison bar plot."""
    mean_mads = []
    median_mads = []
    rotation_indices = []
    
    for rotation in rotations:
        # If we have pre-computed MAD values, use them
        if rotation['mean_mad'] is not None and rotation['median_mad'] is not None:
            mean_mads.append(rotation['mean_mad'])
            median_mads.append(rotation['median_mad'])
            rotation_indices.append(rotation['rotation'])
        else:
            # Otherwise, calculate MAD from predictions
            predictions = rotation['predictions']
            actual = predictions['actual'].flatten()
            loc = predictions['loc'].flatten()  # Mean
            
            # Calculate mean absolute difference
            mean_mad = np.mean(np.abs(actual - loc))
            mean_mads.append(mean_mad)
            
            # For median, we need to approximate median from distribution parameters
            # This is a simplified approach
            scale = predictions['scale'].flatten()
            skewness = predictions['skewness'].flatten()
            
            # Approximate median calculation
            median_approx = np.zeros_like(loc)
            for i in range(len(loc)):
                if abs(skewness[i]) < 0.1:
                    median_approx[i] = loc[i]
                else:
                    # Approximation for median with non-zero skewness
                    median_approx[i] = loc[i] - 0.2 * scale[i] * skewness[i]
            
            median_mad = np.mean(np.abs(actual - median_approx))
            median_mads.append(median_mad)
            rotation_indices.append(rotation['rotation'])
    
    # Sort by rotation index
    sorted_indices = np.argsort(rotation_indices)
    rotation_indices = [rotation_indices[i] for i in sorted_indices]
    mean_mads = [mean_mads[i] for i in sorted_indices]
    median_mads = [median_mads[i] for i in sorted_indices]
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    
    bar_width = 0.35
    x = np.arange(len(rotation_indices))
    
    plt.bar(x - bar_width/2, mean_mads, bar_width, label='Mean MAD')
    plt.bar(x + bar_width/2, median_mads, bar_width, label='Median MAD')
    
    plt.xlabel('Rotation')
    plt.ylabel('Mean Absolute Difference (mm)')
    plt.title('MAD Comparison Across Rotations')
    plt.xticks(x, [f'Rotation {i}' for i in rotation_indices])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'figure4_mad_comparison.png'), dpi=300)
    plt.close()

def create_reflection_template(rotations, output_dir):
    """Create a reflection template with some statistics from the results."""
    if not rotations:
        return
    
    # Calculate average MAD across rotations
    mean_mads = []
    median_mads = []
    
    for rotation in rotations:
        if rotation['mean_mad'] is not None and rotation['median_mad'] is not None:
            mean_mads.append(rotation['mean_mad'])
            median_mads.append(rotation['median_mad'])
    
    if mean_mads and median_mads:
        avg_mean_mad = np.mean(mean_mads)
        avg_median_mad = np.mean(median_mads)
        
        template = f"""# Reflection: Mesonet Precipitation Prediction

## Model Performance Consistency

The model's performance across the {len(rotations)} rotations shows an average MAD (Mean Absolute Difference) of {avg_median_mad:.6f} for median predictions and {avg_mean_mad:.6f} for mean predictions.

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
        template = """# Reflection: Mesonet Precipitation Prediction

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
    
    with open(os.path.join(output_dir, 'reflection_template.md'), 'w') as f:
        f.write(template)

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data from all rotations
    print("Loading rotation data...")
    rotations = load_rotation_data(args.results_dir)
    print(f"Loaded data from {len(rotations)} rotations")
    
    if len(rotations) == 0:
        print("No rotation data found. Exiting.")
        return
    
    # Create figures
    print("Generating Figure 1: Training and validation curves...")
    plot_training_curves(rotations, args.output_dir)
    
    print("Generating Figure 3: Parameter scatter plots...")
    plot_parameter_scatter(rotations, args.output_dir)
    
    print("Generating Figure 4: MAD comparison...")
    plot_mad_comparison(rotations, args.output_dir)
    
    # Copy Figure 0 (model architecture) from first rotation
    model_figure_src = os.path.join(args.results_dir, 'rotation_0', 'model_inner.png')
    if os.path.exists(model_figure_src):
        import shutil
        model_figure_dest = os.path.join(args.output_dir, 'figure0_model_architecture.png')
        shutil.copy(model_figure_src, model_figure_dest)
        print(f"Copied Figure 0 (model architecture) to {model_figure_dest}")
    else:
        print(f"Warning: Could not find model architecture figure at {model_figure_src}")
    
    # Copy Figure 2 (time series) from first rotation
    timeseries_src = os.path.join(args.results_dir, 'rotation_0', 'station_timeseries.png')
    if os.path.exists(timeseries_src):
        import shutil
        timeseries_dest = os.path.join(args.output_dir, 'figure2_station_timeseries.png')
        shutil.copy(timeseries_src, timeseries_dest)
        print(f"Copied Figure 2 (station time series) to {timeseries_dest}")
    else:
        print(f"Warning: Could not find station time series figure at {timeseries_src}")
    
    # Create reflection template
    create_reflection_template(rotations, args.output_dir)
    print(f"Created reflection template at {os.path.join(args.output_dir, 'reflection_template.md')}")
    
    print(f"All figures generated and saved to {args.output_dir}")

if __name__ == "__main__":
    main()