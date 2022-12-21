import numpy as np
from regression._regression import Regression

class RidgeRegression(Regression):
    """Ridge Regression Model
        Least squares with regularized L2 norm.
        
        w_map = arg min ||f(x,w) - t||^2 + lamb * ||w||^2

        lamb = regularization parameter lambda = alpha/beta
        
        alpha: precision of the prior gaussian distribution of w
        beta: precision of the likelihood
    """
    def __init__(self,lamb):
        self.lamb = lamb
        self.w_map = None
        self.sigma = None

    def fit(self, phi: np.ndarray, t: np.ndarray, dev: bool = False):
        """Maximum posterior probability fit of training data
            w_map = (lamb*I + phi_T @ phi)^-1 @ phi_T @ t

        Args:
            phi (np.ndarray): Feature matrix -> cols: features, rows: features per datapoint 
            t (np.ndarray): target variable vector, rows corresponding to phi
            dev (bool): flag if the estimated standard deviation of the predictive distribution is returned
        """
        b =  phi.T @ t
        A = self.lamb * np.eye(phi.shape[1]) + phi.T @ phi

        self.w_map = np.linalg.solve(A,b)

        if dev:
            sse = np.sum(np.square(phi @ self.w_map - t))
            self.sigma = np.sqrt(sse/t.shape[0])
            return self.w_map,self.sigma
        else:
            return self.w_map

    def predict(self, phi: np.ndarray):
        """Maximum posterior probability prediction for the target variable

        Args:
            phi (np.ndarray): Features for the training input 
        """
        if self.w_map is not None:
            return phi @ self.w_map  
        else:
            return -1
            
# Test Code
if "__name__" == "__main__":
    ridge = RidgeRegression(lamb = 0.000001)
    