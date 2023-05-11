import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.csv', delimiter=',')

x = data[:, 1:]
y = data[1]


mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
x = (x - mean) / std


train = int(0.6 * len(x))
val=int(0.2*len(x))
x_train, y_train = x[:train], y[:train]
x_test, y_test = x[train+val:], y[train+val:]
x_val, y_val = x[train:train+val], y[train:train+val]