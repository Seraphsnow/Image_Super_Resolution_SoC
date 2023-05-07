import matplotlib.pyplot as plt
import numpy as np
def data1():
        x = [np.random.rand() for i in range(1000)]
        y = [x[i] + 0.05*np.random.rand() for i in range(1000)]
        return [x, y]
arr = data1()
mean = np.mean(arr,axis=1)
mean = mean.reshape(2,1)
arr1 = arr - mean
cov = np.cov(arr1)
w,v = np.linalg.eigh(cov)
e = np.argmax(w)
a = range(2)
b = [v[e][1]/v[e][0]*(i-mean[0][0])+mean[1][0] for i in a]
plt.plot(a, b, color='hotpink')
plt.title('best fit line')
plt.grid()
m = arr[0]
n = arr[1]
plt.plot(m,n,'o',ms = 0.5)
#plt.savefig("data1.jpg")\
plt.sow()

