import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import inv
from kernel_functions import gaussian_kernel, exponential_kernel, rational_quadratic_kernel, polynomial_kernel


class GaussianFieldMLE:
    def __init__(self, mean, var, kernel_type='gaussian', kernel_param=3.0, degree=3, coef0=1, alpha=1.0):
        """
        Initialize the Gaussian Field MLE model.
        
        Args:
            mean (float): The mean of the data (log-transformed).
            var (float): The variance of the data (log-transformed).
            kernel_type (str): The type of kernel to use ('gaussian', 'exponential', 'polynomial', 'rational_quadratic').
            kernel_param (float): The kernel parameter (length scale for Gaussian/Exponential, variance for others).
            degree (int): Degree of the polynomial kernel (only used for 'polynomial' kernel).
            coef0 (float): The independent term in the polynomial kernel (only used for 'polynomial' kernel).
            alpha (float): The alpha parameter for the Rational Quadratic kernel (only used for 'rational_quadratic' kernel).
        """
        self.mean = mean
        self.var = var
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param
        self.degree = degree  # Used for Polynomial kernel
        self.coef0 = coef0    # Used for Polynomial kernel
        self.alpha = alpha    # Used for Rational Quadratic kernel
        self.coords = None
        self.obs = None
        self.sigma_sq_mle = None

        # Select the kernel function based on user input
        if kernel_type == 'gaussian':
            self.kernel_func = gaussian_kernel
            self.kernel_kwargs = {'var': self.var, 'length_scale': self.kernel_param}
        elif kernel_type == 'exponential':
            self.kernel_func = exponential_kernel
            self.kernel_kwargs = {'var': self.var, 'length_scale': self.kernel_param}
        elif kernel_type == 'polynomial':
            self.kernel_func = polynomial_kernel
            self.kernel_kwargs = {'var': self.var, 'degree': self.degree, 'coef0': self.coef0}
        elif kernel_type == 'rational_quadratic':
            self.kernel_func = rational_quadratic_kernel
            self.kernel_kwargs = {'var': self.var, 'length_scale': self.kernel_param, 'alpha': self.alpha}
        else:
            raise ValueError("Invalid kernel type. Use 'gaussian', 'exponential', 'polynomial', or 'rational_quadratic'.")

    def compute_kernel_value(self, i, j):
        """
        Compute the kernel value between the coordinates `i` and `j` using the selected kernel function.
        
        Args:
            i (int): Index of the first coordinate.
            j (int): Index of the second coordinate.
        
        Returns:
            float: The kernel value between the two points.
        """
        return self.kernel_func(self.coords[i], self.coords[j], **self.kernel_kwargs)

    def load_data(self, data_file):
        """
        Load the data from a file.
        
        Args:
            data_file (str): Path to the file containing coordinates and observations.
        """
        data = np.loadtxt(data_file)
        self.coords = data[:, :2]  # Extract coordinates
        self.obs = data[:, 2]  # Extract observations

    def build_cov_matrix(self):
        """
        Build the covariance matrix for all observations using the selected kernel.
        """
        n = len(self.coords)
        cov_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Call the kernel function and pass the required parameters
                cov_mat[i, j] = self.kernel_func(self.coords[i], self.coords[j], **self.kernel_kwargs)
            cov_mat[i, i] += self.sigma_sq_mle  # Add nugget effect (uncertainty) to the diagonal
        return cov_mat

    def neg_log_likelihood(self, sigma_sq):
        """
        Negative log-likelihood function for MLE estimation of nugget variance (sigma_sq).
        
        Args:
            sigma_sq (float): Nugget variance to be estimated.
        
        Returns:
            float: The negative log-likelihood value.
        """
        self.sigma_sq_mle = sigma_sq
        K = self.build_cov_matrix()  # Compute the covariance matrix
        inv_K = inv(K)  # Compute the inverse of the covariance matrix
        return np.linalg.slogdet(K)[1] + (self.obs - self.mean).T @ inv_K @ (self.obs - self.mean)

    def fit(self):
        """
        Fit the model by optimizing the nugget variance using MLE.
        """
        # Use scalar minimization to estimate the best nugget variance (sigma_sq)
        result = minimize_scalar(self.neg_log_likelihood, bounds=(1e-6, 2), method='bounded')
        self.sigma_sq_mle = result.x

    def Z_cov(self, coord):
        """
        Calculate the covariance vector between a new point and all observed points.
        
        Args:
            coord (tuple): Coordinates of the new point to calculate covariance.
        
        Returns:
            np.array: Covariance vector between new point and observations.
        """
        n = len(self.coords)
        cov = np.zeros(n)
        for i in range(n):
            # Call the kernel function and pass the required parameters
            cov[i] = self.kernel_func(coord, self.coords[i], **self.kernel_kwargs)
        return cov

    def predict(self, new_coord):
        """
        Predict the value at a new location.
        
        Args:
            new_coord (tuple): Coordinates of the new location to predict.
        
        Returns:
            tuple: Predicted value (mean) and conditional variance at the new location.
        """
        K = self.build_cov_matrix()
        inv_K = inv(K)
        z_cov = self.Z_cov(new_coord)
        
        # Conditional mean for the new location
        mu_cond = self.mean + z_cov.T @ inv_K @ (self.obs - self.mean)
        
        # Conditional variance for the new location
        sigma_cond_sq = self.var - z_cov.T @ inv_K @ z_cov
        
        return mu_cond, sigma_cond_sq

