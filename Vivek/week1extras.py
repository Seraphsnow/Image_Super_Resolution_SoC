import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import time

def data1():
	x = [np.random.rand() for i in range(1000)]
	y = [x[i] + 0.05*np.random.rand() for i in range(1000)]
	return [x,y]

def data2():
	x = [np.random.rand() for i in range(1000)]
	y = [(x[i])**2 + 0.05*np.random.rand() for i in range(1000)]
	return [x,y]

first=data1()
second=data2()
regr = linear_model.LinearRegression()

# def logic(logr,x):
# 	return logr.coef_ * x + logr.intercept_
firstx=np.array(first[0])
regr.fit(firstx.reshape(-1,1),first[1])

x=np.linspace(0,1,1000)
y=regr.predict(x.reshape(-1,1))
plt.scatter(first[0],first[1])
plt.plot(x,y, '-r')
plt.show()


secondx=np.array(second[0])
regr.fit(secondx.reshape(-1,1),second[1])

x=np.linspace(0,1,1000)
y=regr.predict(x.reshape(-1,1))
plt.scatter(second[0],second[1])
plt.plot(x,y, '-r')
plt.show()