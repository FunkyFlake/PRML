import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures

def transform_features_poly(x:np.ndarray,deg: int):
    """Builds the feature matrix for polynomial basis functions of the 
    specified degree

    Args:
        x (np.ndarray): Training data
        deg (int): Degree of polynomial

    Returns:
        np.ndarray: Polynomial feature matrix
    """
    return PolynomialFeatures(degree=deg).fit_transform(x.reshape(-1,1))

def transform_features_gauss(x: np.ndarray,std: np.ndarray, num: int):
    """Builds the feature matrix for a linear combination of gaussian basis
    functions with constant standard deviation and linear spacing.

    Args:
        x (np.ndarray): Training data
        std (np.ndarray): Standard deviation for gaussian basis 
        num (int): number of gaussian basis functions

    Returns:
        np.ndarray: Feature matrix
    """
    locs = np.linspace(x[0],x[-1],num) 
    Phi = np.zeros([x.size,num])
    for i in range(num):
        Phi[:,i] = norm.pdf(x,loc=locs[i],scale=std)
    return Phi