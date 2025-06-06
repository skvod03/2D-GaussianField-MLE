# 2D Gaussian Field MLE Prediction

This project implements Maximum Likelihood Estimation (MLE) for predicting contaminant concentrations in a 2D Gaussian field using kernel-based methods. Instead of assuming a perfect model, we recognize that there may be errors in our measurements. The parameter sigma (σ), representing the variance of the noise in the observations, is estimated using MLE from a multivariate Gaussian distribution. This approach accounts for both the underlying spatial dependencies of the contaminant field and the uncertainty introduced by measurement errors.

We believe that the observed contaminant levels, modeled as a 2D Gaussian field, are influenced by spatial correlations that can be modeled using a kernel function. The model helps to predict the concentration at unobserved locations based on known data, while also providing conditional statistics (mean and variance) that incorporate measurement error. This makes the model both flexible and parsimonious, allowing it to fit real-world scenarios with inherent noise.

## Installation

1. Clone this repository.
2. Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

To use the model, instantiate the class `GaussianFieldMLE` and call the methods to fit the model and make predictions. The model estimates the nugget variance (sigma) through MLE of the multivariate Gaussian and uses kernel methods to predict contaminant concentrations at new locations.

### Example:

```python
# Instantiate the model
model = GaussianFieldMLE(mean=2.0, var=0.5, kernel_type='exponential', kernel_param=3.0)

# Load data from a file
model.load_data('GF_Data.txt')  # Replace with your actual data file path

# Fit the model (estimate nugget variance)
model.fit()

# Make a prediction at a new location (for example, at (0, 0))
new_coord = (0, 0)
mu_cond, sigma_cond_sq = model.predict(new_coord)

# Print the predicted value and conditional variance
print(f"Predicted value at {new_coord}: {mu_cond}")
print(f"Conditional variance at {new_coord}: {sigma_cond_sq}")
```

### How the Model Works:

1. **Data Model**:

   * Observed data consists of coordinates and corresponding contaminant levels.
   * We assume that the contaminant concentrations follow a 2D Gaussian process with some inherent noise (measurement error).
   * The underlying field is modeled using a kernel function, and the measurement error is captured in the nugget variance (sigma).

2. **Maximum Likelihood Estimation (MLE)**:

   * The nugget variance (sigma) is estimated using MLE, assuming that the errors in the observations are normally distributed with mean zero and variance sigma.
   * MLE is applied to the multivariate Gaussian distribution, which is used to estimate the covariance matrix that governs the spatial relationships between the observations.

3. **Prediction**:

   * The model uses the estimated nugget variance and kernel to predict the concentration at new locations.
   * It also provides conditional statistics (mean and variance) for the predicted concentration, accounting for both the spatial structure of the data and the measurement error.

### Important Concepts:

* **Parsimonious Statistical Model**: The model is parsimonious in that it uses a limited number of parameters (mean, variance, kernel function, and nugget variance) to describe the spatial structure and uncertainty in the data. It avoids overfitting and provides a simple yet powerful framework for making predictions in the presence of noise.

* **Kernel-Based Methods**: The spatial dependencies of the contaminant field are modeled using kernel functions, which capture the similarity between pairs of points in space. This allows the model to predict contaminant levels at new locations based on the spatial proximity to the observed data.
