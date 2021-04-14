import numpy as np 
import yfinance as yf
import matplotlib.pyplot as plt
import Model

ticker = 'OCGN'
K_min = 2.5
K_max = 20.0

fn, pl = Model.return_distr(ticker, verbose=False)
s = Model.generate_s_matrix(6.89, fn, pl, days=3, num=10000)
p, x = Model.generate_p_x(s, -1, res=1000)

plt.clf()
plt.plot(x, p)
plt.title("Model of p(x)")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.show()

cp = []
Ks = []
for i in range(100):
	K = (K_max - K_min)*(i)/99 + K_min
	Ks.append(K)
	c = Model.call_price(p, np.array(x), K)
	cp.append(c)

	if i % 10 == 0:
		print("Strike $", K, ": $", c)

plt.clf()
plt.plot(Ks, cp, c="green")
plt.title("Call Price for " + ticker)
plt.xlabel("Strike")
plt.ylabel("Premium")
plt.show()


print("Strike $", 7.5, ": $", Model.call_price(p, np.array(x), 7.5))