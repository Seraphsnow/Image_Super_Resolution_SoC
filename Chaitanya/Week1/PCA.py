import numpy as np
from matplotlib import pyplot as plt
def data2():
	x = [np.random.rand()*10 for i in range(1000)]
	y = [(x[i])**2 + 0.05*np.random.rand() for i in range(1000)]
	return [x, y]
dataset = data2()
avg = np.mean(dataset, axis =1) #makes a 1-D array that has mean of the input array along axis 1
navg = avg.reshape(2,1) #to make it a 2-D array
stdds = dataset - navg #standardization
c_ds = np.cov(stdds)
w, v = np.linalg.eigh(c_ds) #returns eigenvalues in w with mapping to corresponding eigenvectors in v
m = v[-1, 1]/v[-1,0] 
c = navg[1,0] - m*navg[0,0] #post standardization
x = np.linspace(0,10,1000) #
y = m*x+c
plt.plot(x, y, '-r', label='best fit line')
plt.title('Best fit line')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
x =dataset[0]
y =dataset[1]
plt.scatter(x, y)
plt.show()
