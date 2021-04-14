import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def return_distr(ticker, verbose=True):
	stock = yf.Ticker(ticker)
	price_history = stock.history(period="59d", interval="2m")
	close_history = price_history["Close"]

	#Create Return distribution
	daily_return = []
	for i in range(len(close_history)-1):
		d_return = (close_history[i + 1] - close_history[i])/close_history[i]
		daily_return.append(d_return)

	resolution = int(len(daily_return)/2)
	hist = plt.hist(daily_return, bins=resolution)

	if verbose:
		plt.show()

	pl = []
	for i in range(len(list(hist[1])) - 1):
		price_ratio = (hist[1][i + 1] + hist[1][i])/2
		pl.append(price_ratio)

	fn = hist[0]/np.sum(hist[0])

	return fn, pl


def generate_s_matrix(s0, fn, pl, days=30, interval=30*24, num=1000):
	s = np.full((int(days*interval), num), s0)
	random_movements = np.random.choice(pl, p=fn, size=(int(days*interval) - 1, num))

	for t in range(int(days*interval) - 1):
		rm = random_movements[t]
		s[t + 1] = rm*s[t] + s[t]

	return s


def generate_p_x(s, T, res=1000):
	s_T = s[T]

	hist = plt.hist(s_T, bins=res)

	X = []
	for i in range(len(list(hist[1])) - 1):
		x = (hist[1][i+1] + hist[1][i])/2
		X.append(x)

	p = hist[0]/np.sum(hist[0])

	return p, X


def call_price(p, X, K):
	c = np.sum(np.where(X > K, (X - K)*p, 0))
	return c
