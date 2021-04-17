import numpy as np
from scipy.stats import lognorm
import yfinance as yf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Model

ticker = 'SPY'

K_min = 500 #Min Strike to analyze
K_max = 600 #Max Strike
s0 = 411.87 #Current Spot Price
days = 200 #Days till expiration
interval = (30*6.5)#intervals per trading day (2m intervals for 6.5 hours)
T = days*interval #Calculate how many intervals of return till expiration

#Obtain return distribution aka r(x)
fn, pl, return_distr = Model.return_distr(ticker, verbose=False)
log_return = np.log(np.array(return_distr) + 1)

#We will use these parameters to solve for call price
mu = np.mean(log_return)

#Override mu.. assume martingality
mu = 0.0
sigma = np.std(log_return)

#Create Surface
c = np.zeros((100,100))

for i in range(100): #Strike subsection
	K = (K_max - K_min)*(i/99) + K_min

	for j in range(100): #Time subsection
		t = T*(j/99)
		c[i][j] = Model.call_price_pless(s0, K, mu, sigma, t)

X = (np.arange(100)/100)*(K_max - K_min) + K_min
Y = (np.arange(100)/100)*T

x, y = np.meshgrid(Y, X)

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, c)
ax.set_title(ticker)
ax.set_ylabel('$K$')
ax.set_xlabel('$t$')
ax.set_zlabel('Price')
plt.show()



#Create Surface
c = np.zeros((100,100))

for i in range(100): #Volatility subsection
	sig = (2*sigma)*(i/99) + 1e-6

	for j in range(100): #Price subsection
		t = T*(j/99)

		c[i][j] = Model.call_price_pless(s0, 555, mu, sig, t)

X = (np.arange(100)/100)*(2*sigma)
Y = (np.arange(100)/100)*(T)

x, y = np.meshgrid(Y, X)

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, c)
ax.set_title(ticker + " $555 Call")
ax.set_ylabel('$\sigma$')
ax.set_xlabel('$t$')
ax.set_zlabel('Price')
plt.show()

print(Model.call_price_pless(s0, 555, mu, sigma, T))


