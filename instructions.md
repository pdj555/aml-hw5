# The Problem

The Oklahoma Mesonet is a network of weather stations scattered across Oklahoma—with at least one station in every county. Each station measures various meteorological variables every 5 minutes. The dataset contains a single summary sample for each of the 136 stations for every day from 1994 to 2000.

The measured variables (described in the Mesonet Daily Summary Data document) are as follows:

## Temperature & Derived Statistics
- **TMAX, TMIN, TAVG**
- **DMAX, DMIN, DAVG**

## Other Meteorological Variables
- **VDEF**
- **SMAX, SMIN, SAVG**
- **BMAX, BMIN, BAVG**
- **HMAX, HMIN, HAVG**
- **PMAX, PMIN, PAVG**
- **MSLP, AMAX**
- **ATOT**
- **WSMX, WSMN**
- **WSPD, WDEV, WMAX**
- **9AVG**
- **2MAX, 2MIN, 2AVG, 2DEV**
- **HDEG, CDEG, HTMX, WCMN**

**Dataset Location:**  
Available on SCHOONER at  
`/home/fagg/datasets/mesonet/allData1994_2000.csv`

---

# Supporting Code

The provided code includes several key functionalities:

## Loading Datasets

```python
get_mesonet_folds(
    dataset_fname: str,
    ntrain_folds: int = 6, 
    nvalid_folds: int = 1, 
    ntest_folds: int = 1,
    rotation: int = 0
)
```

**Returns:**  
Numpy arrays: `ins_training`, `outs_training`, `ins_validation`, `outs_validation`, `ins_testing`, `outs_testing`

**Note:**  
Each fold contains data from different Mesonet stations.

## Extracting Data for a Specific Station

```python
extract_station_timeseries(
    ins: np.array, 
    outs: np.array,
    nstations: int, 
    station_index: int
)
```

**Purpose:**  
Retrieve the input and output data for one specific station (in temporal order) from a dataset containing multiple stations.

## Sinh-Arcsinh Distribution Implementation

The `SinhArcsinh` class provides three key methods:

- **`num_params()`**  
  Returns the number of parameters required for the distribution. Each parameter corresponds to one Tensor (for mean, standard deviation, skewness, and tailweight).

- **`create_layer()`**  
  Returns a proper Keras 3 Layer that accepts a sequence of 4 Keras Tensors. It assumes that standard deviation and tailweight are positive. When passed TF Tensor data, the layer returns Tensorflow Probability Distributions (not TF Tensors).

- **`mdn_loss(y, dist)`**  
  Can be used as a loss function when compiling the outer model. It returns the negative log likelihood for each true value `y` given the parameterized distribution `dist`.

**Additional Resources:**
- Probabilistic Neural Networks Demo: `pnn-solution.ipynb`
- Synthetic Data
- Normal Distribution

---

# Deep Learning Experiment

Construct a model that:

## Inputs
- Daily summary data from a single Mesonet station (a row in `ins_*`)

## Output
- A probability distribution predicting likely rainfall measurements (a row in `outs_*`)

## Design Approach
Use an **inner/outer model design**:

- **Inner Model:**  
  Transforms the station data into a set of parameters for a Sinh-Arcsinh distribution.

- **Outer Model:**  
  Produces the corresponding probability distribution.

## Model Specifics
- The Sinh-Arcsinh distribution requires **four input parameters**:
  - Mean
  - Standard deviation
  - Skewness
  - Tailweight

  Each parameter is a vector (one value per input example, conditioned on the station data).

- **Constraints:**
  - Standard deviation and tailweight must be strictly positive (use a softplus non-linearity).
  - The other two parameters are unbounded.

- **Loss Function:**  
  Use negative log likelihood.

- **Architecture:**  
  Allocate an appropriate set of hidden layers (and sizes) for the inner model.

---

# Performance Reporting

After selecting a reasonable architecture and hyper-parameters, perform **eight rotations** of experiments and produce the following:

- **Figure 0:**  
  Inner network architecture (via `plot_model()`).

- **Figures 1a & 1b:**  
  Training and validation set negative likelihood vs. epoch for each rotation (each figure contains eight curves).

- **Figure 2:**  
  Several time-series examples from a test dataset showing:
  - Observed precipitation
  - Distribution mean
  - 10th, 25th, 75th, and 90th distribution percentiles  
  *(Choose interesting time periods.)*

- **Figures 3a, 3b, 3c, 3d:**  
  A scatter plot combining data from all eight rotations showing predicted:
  - Mean
  - Standard deviation
  - Skewness
  - Tailweight  
  Plotted as functions of observed precipitation.

- **Figure 4:**  
  For each rotation, compute the mean absolute difference (MAD) between:
  - Observed precipitation and median predicted precipitation
  - Observed precipitation and mean predicted precipitation  
  Display these MADs using a bar plot with twelve bars, organized logically.

---

# Reflection

Discuss the following in detail:

- How consistent is your model performance across the different rotations?
- Based on the time-series plots, describe and explain the shape of the probability density function (pdf) and how it changes over time.
- How is skewness utilized by the model? Is there a consistent variation in this parameter?
- How is tailweight utilized by the model? Is there a consistent variation in this parameter?
- Is the Sinh-Arcsinh distribution appropriate for modeling this phenomenon? Provide a detailed explanation.
- Are your models effective at predicting precipitation? Justify your answer.

---

# Hints

- **Model Structure:**  
  Use the structure from the PNN demo code released this week (inner/outer model design with the outer model returning a probability distribution).

- **Model Outputs:**  
  - `model_outer.predict(...)` returns a sequence of samples from the learned distribution (one per example, conditioned on the input).  
  - `model_outer(...)` returns a sequence of distributions (one per example, conditioned on the input).

- **Data Normalization:**  
  The range of input variables varies dramatically.  
  - Add a batch normalization step between your inputs and your first hidden layer.  
  - Consider additional batch normalization or kernel initializations to limit the magnitude of initial parameters (do not initialize weights to zero).

- **Training Advice:**  
  Be patient—there is significant learning to be achieved through training.

- **Additional Resource:**  
  See the Sinh-Arcsinh distribution documentation for tips on parameterized distributions (helpful for creating some figures).

- **Note:**  
  Tensorflow Probability is sensitive to the specific combination of package versions in the dnn environment.

- **GPU:**  
  Not required for this assignment.

---

# What to Hand In

Submit a single zip file containing:

- All Python code (`.py` files) and notebook files (`.ipynb`)
- Figures 0–4
- Reflection document

**Do not submit pickle files.**

---

# Grading

- **10 pts:** Clean, general code for model building (including in-code documentation)
- **10 pts:** Figure 0
- **10 pts:** Figure 1
- **10 pts:** Figure 2
- **10 pts:** Figure 3
- **10 pts:** Figure 4
- **15 pts:** Reasonable test set performance for all rotations
- **25 pts:** Reflection
- **Bonus 5 pts:** Compute the MADs in a metric function declared in `model.compile()` and called for every epoch
