# Oklahoma Mesonet Rainfall Prediction

This project implements a Probabilistic Neural Network (PNN) with a Sinh-Arcsinh distribution to predict daily rainfall based on meteorological variables from the Oklahoma Mesonet dataset.

## Project Overview

The Oklahoma Mesonet is a network of environmental monitoring stations across Oklahoma. The dataset used spans from 1994 to 2000 and includes measurements from 136 stations. The goal is to predict daily rainfall (RAIN) using various meteorological variables.

## Implementation Details

### Data

The dataset includes the following variables used for prediction:
- Air temperature at 9 meters (TAIR)
- Relative humidity (RELH)
- Wind speed (WSPD)
- Wind direction (WDIR)
- Solar radiation (SRAD)
- Atmospheric pressure (PRES)
- Soil temperature at 10 cm (SOIL)
- Rainfall (RAIN) - the target variable

### Model Architecture

The model follows an inner/outer architecture:

1. **Inner Model**: 
   - A neural network with configurable hidden layers
   - Takes meteorological variables as input
   - Outputs 4 parameters for the Sinh-Arcsinh distribution: location, scale, skewness, and tailweight

2. **Outer Model**:
   - Takes the inner model's outputs and constructs a Sinh-Arcsinh distribution
   - Uses the distribution to model the rainfall prediction probabilistically

### Sinh-Arcsinh Distribution

The Sinh-Arcsinh distribution is particularly suited for this task because:
- It can handle skewed data (rainfall data is typically right-skewed)
- It has flexible tails to account for rare extreme events
- It can model both zero and non-zero rainfall amounts

### Training Process

The model is trained using:
- Adam optimizer with configurable learning rate
- A custom loss function (negative log likelihood)
- Early stopping to prevent overfitting
- Validation data to monitor training progress

### Evaluation

The model is evaluated by:
- Calculating the test loss
- Visualizing predictions for individual stations
- Analyzing the distribution parameters
- Generating samples from the predicted distributions

## Files

- `pnn-rainfall-solution.py`: Main implementation of the rainfall prediction model
- `mesonet_support.py`: Support code for loading data and implementing the Sinh-Arcsinh distribution
- `run_mesonet_pnn.sh`: Batch script for running experiments on SCHOONER
- `mesonet.pdf`: Documentation about the Oklahoma Mesonet dataset

## Usage

### Running Locally

To run the implementation locally:

```bash
python pnn-rainfall-solution.py --hidden_layers 64,32,16 --dropout 0.2 --epochs 100 --batch_size 128 --output_dir results
```

### Running on SCHOONER

To run on SCHOONER, submit the batch job:

```bash
sbatch run_mesonet_pnn.sh
```

The batch script will:
1. Run the model with different hidden layer configurations (32,16,8), (64,32,16), and (128,64,32,16)
2. Run the model with different dropout rates (0.1, 0.2, 0.3, 0.4) for the best architecture
3. Save results in timestamped directories
4. Use the normal_hdr100 partition with 32 CPU cores and 64GB memory

### Command Line Arguments

The `pnn-rainfall-solution.py` script accepts the following arguments:

- `--hidden_layers`: Comma-separated list of hidden layer sizes (default: "64,32,16")
- `--dropout`: Dropout rate for hidden layers (default: 0.2)
- `--epochs`: Number of epochs to train (default: 100)
- `--batch_size`: Batch size for training (default: 128)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--dataset_path`: Path to the Mesonet dataset (default: '/home/fagg/datasets/mesonet/allData1994_2000.csv')
- `--output_dir`: Directory to save results (default: 'results')

## Experiments

The batch script runs two sets of experiments:

1. **Architecture Exploration**: Tests different hidden layer configurations to determine the optimal network depth and width
   - Small network: [32,16,8] - Simple architecture with fewer parameters
   - Medium network: [64,32,16] - Moderate complexity
   - Large network: [128,64,32,16] - Deep architecture with more parameters

2. **Regularization Tuning**: For the best architecture, tests the impact of different dropout rates
   - Low dropout: 0.1 - Minimal regularization
   - Moderate dropout: 0.2 - Standard level of regularization
   - Higher dropout: 0.3 and 0.4 - Stronger regularization

## Results

The model produces:
- Probabilistic predictions of rainfall
- Uncertainty estimates through distribution parameters
- Visualizations of the predicted distributions

Each experiment will save:
- Model architecture diagrams
- Training and validation loss curves
- Prediction visualizations for sample stations
- Distribution parameter histograms
- Best model weights

The Sinh-Arcsinh distribution allows the model to capture the skewed nature of rainfall data, providing more accurate predictions for both common and extreme weather events. 