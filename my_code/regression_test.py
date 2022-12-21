# %%
# Curvefitting test script
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures
from regression._ridge_regression import RidgeRegression
from regression._bayesian_regression import BayesianRegression
import regression.features as rf

# %%
def ground_truth(x):
    return np.sin(2 * np.pi * x)

def add_noise(x,sig):
    return x + np.random.normal(loc=0,scale=sig,size=np.shape(x))



# %% Training data generation
x_true = np.arange(0,1.01,0.01)
y_true = ground_truth(x_true)

# Equidistant noisy samples as training data
N = 20  # Datapoints
x_train = np.arange(0,1+1/N,1/N)
#x_train = np.array([0.2,0.4,0.6,0.8])
y_train = ground_truth(x_train)
noise_sigma = 0.2
y_train = add_noise(y_train,sig=noise_sigma)

# %% Ridge Regression
d = 9 # Degree of polynomial
num = 5 # Number of gaussian basis functions
std_basis = 0.2 # Standard deviation of gaussian basis functions

#Phi = rf.transform_features_poly(x_train,deg=d) # Feature matrix for training data
Phi = rf.transform_features_gauss(x_train, std=std_basis, num=num)

lamb = 0.001      # alpha/beta # Regularization factor for ridge regression

ridge = RidgeRegression(lamb)
w_map, sigma = ridge.fit(Phi,y_train,True)


# %% Plot of training data and ground truth for ridge regression
#y_ridge = ridge.predict(rf.transform_features_poly(x_true,d))
y_ridge = ridge.predict(rf.transform_features_gauss(x_true,std=std_basis, num=num))

plt.plot(x_true,y_true,'k',label='Ground Truth')
plt.plot(x_train,y_train,linestyle='none',color='b',ms=7,marker='o',label='Training Data')
plt.plot(x_true,y_ridge,'r',label='Model Mean (MAP)')
plt.fill_between(x_true, y_ridge - sigma, y_ridge + sigma, color="pink", label="std.")
plt.ylim([-1.2,1.2])
plt.legend()
plt.title('Ridge Regression')

# %% Bayesian Regression
# Hyperparameters
alpha = 0.5    # Precision of prior, approaches uniform distribution for alpha -> 0
beta = 5      # Precision of likelihood (inverse variance of target parameter)

bayes = BayesianRegression(alpha=alpha, beta=beta)
w_mean, w_cov = bayes.fit(Phi,y_train)

# %% Plot of training data and ground truth for bayesian regression
#y_bayes,y_std = bayes.predict(rf.transform_features_poly(x_true,d), return_std=True)
Phi_gauss = rf.transform_features_gauss(x_true,std=std_basis, num=num)
y_bayes,y_std = bayes.predict(Phi_gauss, return_std=True)
y_bayes_samples = bayes.predict_samples(Phi_gauss,5)

plt.plot(x_true,y_true,'k',label='Ground Truth')
plt.plot(x_train,y_train,linestyle='none',color='b',ms=7,marker='o',label='Training Data')
plt.plot(x_true,y_bayes,'r',label='Model Mean (MAP)')
plt.fill_between(x_true, y_bayes - y_std, y_bayes + y_std, color="pink", label="std.")
#plt.plot(x_true,y_bayes_samples,'b',label='Samples')
plt.ylim([-1.2,1.2])
plt.legend()
plt.title('Bayesian Regression')
# %%
