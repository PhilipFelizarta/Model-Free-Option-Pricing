import numpy as np
from scipy.stats import lognorm
import yfinance as yf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Model

ticker = 'OCGN'

K_min = 2.5 #Min Strike to analyze
K_max = 20.0 #Max Strike
s0 = 6.89 #Current Spot Price
days = 2 #Days till expiration
interval = (30*24)/6.5 #intervals per trading day (2m intervals for 6.5 hours)
T = days*interval #Calculate how many intervals of return till expiration

#Obtain return distribution aka r(x)
fn, pl, return_distr = Model.return_distr(ticker, verbose=False)
log_return = np.log(np.array(return_distr) + 1)

#We will use these parameters to solve for call price
mu = np.mean(log_return)
sigma = np.std(log_return)

#Create Surface
c = np.zeros((100,100))

for i in range(100): #Strike subsection
	K = (K_max - K_min)*(i/99) + K_min

	for j in range(100): #Time subsection
		t = T*(j/99) + 1e-6
		c[i][j] = Model.call_price_pless(s0, K, mu, sigma, t)

X = (np.arange(100)/100)*(K_max - K_min) + K_min
Y = (np.arange(100)/100)*T

x, y = np.meshgrid(Y, X)

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, c)
plt.show()

print(Model.call_price_pless(s0, 10.0, mu, sigma, T))


