import numpy as np

# Code to generate n datapoints, returns an np array of n numbers ranging from 0
# to 10000
def data(n):
	x = np.random.randint(10001, size = n)
	return x	

def QuickSort(nparray):
	k = 0
	if len(nparray) == 0:
		return nparray
	for i in range(len(nparray)-1):
		if nparray[i] <= nparray[-1]:
			nparray[i], nparray[k] = nparray[k], nparray[i]
			k+=1
	nparray[-1], nparray[k] = nparray[k], nparray[-1]
	QuickSort(nparray[:k])
	QuickSort(nparray[k+1:])
n = 20 #Enter value of n here
nparray = data(n)
print(nparray)
QuickSort(nparray)
print(nparray)
