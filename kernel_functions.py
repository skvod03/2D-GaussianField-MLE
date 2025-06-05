import numpy as np

# Gaussian Kernel
def gaussian_kernel(x1, x2, var, length_scale):
    """
    Gaussian kernel (Squared Exponential Kernel).
    
    Args:
        x1 (numpy.array): The first coordinate (point).
        x2 (numpy.array): The second coordinate (point).
        var (float): The variance (scaling factor) of the kernel.
        length_scale (float): The length scale parameter that controls the width of the kernel.

    Returns:
        float: The kernel value (covariance) between the two points.
    """
    dist = np.linalg.norm(x1 - x2)  # Euclidean distance between the two points
    return var * np.exp(-dist ** 2 / (2 * length_scale ** 2))

# Exponential Kernel
def exponential_kernel(x1, x2, var, length_scale):
    """
    Exponential kernel (Laplace kernel).
    
    Args:
        x1 (numpy.array): The first coordinate (point).
        x2 (numpy.array): The second coordinate (point).
        var (float): The variance (scaling factor) of the kernel.
        length_scale (float): The length scale parameter that controls the width of the kernel.

    Returns:
        float: The kernel value (covariance) between the two points.
    """
    dist = np.linalg.norm(x1 - x2)  # Euclidean distance between the two points
    return var * np.exp(-dist / length_scale)  # Exponential decay

# Polynomial Kernel
def polynomial_kernel(x1, x2, var, degree, coef0):
    """
    Polynomial kernel.
    
    Args:
        x1 (numpy.array): The first coordinate (point).
        x2 (numpy.array): The second coordinate (point).
        var (float): The variance (scaling factor) of the kernel.
        degree (int): The degree of the polynomial.
        coef0 (float): The independent term in the polynomial kernel.

    Returns:
        float: The kernel value (covariance) between the two points.
    """
    return var * (np.dot(x1, x2) + coef0) ** degree  # Polynomial kernel: K(x1, x2) = (x1^T * x2 + coef0)^degree

# Rational Quadratic Kernel
def rational_quadratic_kernel(x1, x2, var, length_scale, alpha):
    """
    Rational Quadratic kernel.
    
    Args:
        x1 (numpy.array): The first coordinate (point).
        x2 (numpy.array): The second coordinate (point).
        var (float): The variance (scaling factor) of the kernel.
        length_scale (float): The length scale parameter that controls the width of the kernel.
        alpha (float): A parameter that controls the relative weighting of short-term and long-term correlations.

    Returns:
        float: The kernel value (covariance) between the two points.
    """
    dist = np.linalg.norm(x1 - x2)  # Euclidean distance between the two points
    return var * (1 + (dist ** 2) / (2 * alpha * length_scale ** 2)) ** (-alpha)