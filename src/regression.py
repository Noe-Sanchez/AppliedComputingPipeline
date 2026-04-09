# Numpy linreg
import numpy as np

# Data comes from colums of pd dataframe
def fit_regression(X, y, degree): 
  beta = None

  # Simple regression
  if degree == 1:
    # Add bias term
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    # Calculate coefficients using normal equation
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
  # Polynomial regression
  else:
    # Create polynomial features
    X_poly = np.hstack([X ** i for i in range(1, degree + 1)])
    # Add bias term
    X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))
    # Calculate coefficients using normal equation
    beta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

  return beta
