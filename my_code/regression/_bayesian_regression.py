import numpy as np
from regression._regression import Regression

# %%
class BayesianRegression(Regression):
    """
        Regression model for fully bayesian approach to linear 
        regression with weights w and target variable t treated 
        as random variables.
        Predictions are calculated by marginalizing the posterior
        parameter w distribution.

        w ~ N(0 | 0, alpha*I) Gaussian prior with independent weights
        t ~ N(t | f(x,w), beta^-1) Likelihood
        
        Posterior of weigths:
        w ~ N(w | m, S)
    """
    def __init__(self,alpha: float = 1., beta: float = 1.):
        """Initialize Bayesian Regression model with hyperparameters.
        alpha/beta = lambda from the regularization term in ridge regression.

        m is the mean of the posterior distribution of the weights
        S_inv is the precision matrix of the posterior distribution of the weights
        S is the covariance matrix (inverse of S_inv)
        The posterior distribution of the weights is regarded as a new prior after the
        first training.
        m is equal to the maximum posterior probability estimate for the weights
        Args:
            alpha (float, optional): Precision of the prior weights. Defaults to 1.
            beta (float, optional): Precision of the likelihood. Defaults to 1.
        """
        self.alpha = alpha
        self.beta = beta
        self.m = None
        self.S_inv = None
        self.S = None

    def _is_prior_defined(self) -> bool:
        return self.m is not None and self.S_inv is not None

    def _get_prior(self,ndim: int) -> bool:
        if self._is_prior_defined():
            return self.m, self.S_inv
        else:
            # Only the special case of zero mean, diagonal Sariance gaussian prior is implemented
            return np.zeros(ndim), self.alpha*np.eye(ndim)

    def fit(self, phi: np.ndarray, t: np.ndarray):
        """Bayesian update of the model. Use current posterior as new prior for the weights.
        Likelihood of new data is used to calculate the new posterior.
        
        Args:
            phi (np.ndarray): Feature matrix -> cols: features, rows: features per datapoint 
            t (np.ndarray): target variable vector, rows corresponding to phi

        Returns:
            m (np.ndarray): mean of posterior distribution of w
            S (np.ndarray): covariance matrix of posterior distribution of w
        """
        # get prior or initialize prior if this is the first training
        m_prior, S_inv_prior = self._get_prior(np.size(phi,1)) 

        # get new posterior precision S_inv
        self.S_inv = S_inv_prior + self.beta * phi.T @ phi   # PRML (3.51)

        # get new posterior mean m
        b = S_inv_prior @ m_prior + self.beta * phi.T @ t
        self.m = np.linalg.solve(self.S_inv,b)
        
        # calculate new Sariance matrix
        self.S = np.linalg.inv(self.S_inv)
        
        return self.m, self.S

    def predict(self, phi: np.ndarray, return_std: bool = False, sample_size: int = None):
        """Predict target value for new test data. 
        Bayesian prediction by marginalizing over the posterior weight distribution
        p(t) = int/p(t | w) * p(w) dw

        Args:
            phi (np.ndarray): Feature matrix -> cols: features, rows: features per datapoint
            return_std: Flag to return the standard deviation of the predicitve distribution for 
                        the given features
        
        Returns:
            t_mean (np.ndarray): Maximum posterior probability estimate for the target variable
            t_std (np.ndarray): Standard deviation for the predicted target variable
        """

        t_mean = phi @ self.m   # MAP estimate

        if return_std:
            # (3.59) calculation of diagonal elements of quadratic form only 
            t_var = 1/self.beta +  (phi @ self.S * phi).sum(1) 
            t_std = np.sqrt(t_var)
            return t_mean, t_std
        
        return t_mean

    def predict_samples(self, phi: np.ndarray, sample_size: int = 1):
        """Draws samples from the posterior parameter distribution and predicts the target t for
        these weights. 

        Args:
            phi (np.ndarray): Feature matrix -> cols: features, each row is a datapoint
            sample_size (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            t_sample (np.ndarray): for a single sample, this is a prediction vector, otherwise a matrix
            with predictions for each row in the feature matrix
        """
        w_sample = np.random.multivariate_normal(self.m, self.S, size=sample_size)
        t_sample = phi @ w_sample.T
        return t_sample

# %% Test Code
if "__name__" == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures

