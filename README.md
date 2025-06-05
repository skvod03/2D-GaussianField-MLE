# 2D Gaussian Field MLE Prediction

This project implements Maximum Likelihood Estimation (MLE) for predicting contaminant concentrations in a 2D Gaussian field using kernel-based methods. It uses data in the form of coordinates and observed contaminant levels and models the field with Gaussian processes.

## Installation

1. Clone this repository.
2. Install dependencies using:
```
pip install -r requirements.txt
```
Usage
To use the model, instantiate the class GaussianFieldMLE and call the methods to fit the model and make predictions.

Example:
```
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
