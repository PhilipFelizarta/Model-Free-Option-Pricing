import numpy as np
from scipy.stats import lognorm
import yfinance as yf
import matplotlib.pyplot as plt
import Model

ticker = 'OCGN'
K_min = 2.5
K_max = 20.0
s0 = 6.89
days = 2
interval = 30*24
T = days*interval

fn, pl, return_distr = Model.return_distr(ticker, verbose=False)

log_return = np.log(np.array(return_distr) + 1)
mu = np.mean(log_return)
sigma = np.std(log_return)

print("Mu: ", mu)
print("Sigma: ", sigma)


s = Model.generate_s_matrix(s0, fn, pl, days=days, num=10000)
p, x = Model.generate_p_x(s, -1, res=1000)


#Generate the estimated lognormal distr from our i.i.d assumptions
p_2 = []
for i in range(len(x)):
	p_2.append(lognorm.pdf(x[i], np.sqrt(T)*sigma, scale=np.exp(T*mu + np.log(s0))))

plt.clf()
plt.plot(x, p)
plt.plot(x, p_2/np.sum(p_2))
plt.title("Models of p(x)")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.show()

print("Adjusted Mu: ", T*mu + np.log(s0))
print("Adjusted Sigma: ", np.sqrt(T)*sigma)


cp = []
Ks = []
c2 = []
for i in range(100):
	K = (K_max - K_min)*(i)/99 + K_min
	Ks.append(K)
	c = Model.call_price(p, np.array(x), K)

	cp.append(c)

	c_pless = Model.call_price_pless(s0, K, mu, sigma, T)
	c2.append(c_pless)

	if i % 10 == 0:
		print("Strike $", K, " MC: $", c, " LN: ", c_pless)

plt.clf()
plt.plot(Ks, cp, c="green")
plt.plot(Ks, c2, c="blue")
plt.title("Call Price for " + ticker)
plt.xlabel("Strike")
plt.ylabel("Premium")
plt.show()


print("Strike $", 7.5, " MC: $", Model.call_price(p, np.array(x), 7.5), " LN: ", Model.call_price_pless(s0, 7.5, mu, sigma, T))