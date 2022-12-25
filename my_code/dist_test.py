#%%
import sys
sys.path.insert(0, 'C:/Users/janla/Documents/PRML')

import numpy as np
import matplotlib.pyplot as plt
%matplotlib widget

from dist.beta import Beta
from dist.bernoulli import Bernoulli


#%%
bern = Bernoulli(mu=None)
x = np.array([0,0,0,1])
bern.fit(x)
print(f"mu={bern.mu}")

#%%
x = np.array([1,1,1,1,0,0,1])
bern.fit(x)
print(f"mu={bern.mu}")

#%%
x = np.array([[0,0],[1,1],[1,1],[1,1],[1,0]])
bern.fit(x)
print(f"mu={bern.mu}")

#%%
beta = Beta(n_zeros = 3, n_ones = 7)
bern = Bernoulli(mu = beta)
x = np.array([0,0])
bern.fit(x)
print(f"n_ones = {bern.mu.n_ones}")
print(f"n_zeros = {bern.mu.n_zeros}")
bern._map(np.array([]))
print(f"mu = {bern.prob}")  