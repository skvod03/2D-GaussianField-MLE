from GaussianFieldMLE import GaussianFieldMLE
# Example usage:
if __name__ == "__main__":
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