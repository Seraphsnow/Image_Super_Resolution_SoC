import numpy as np
from array import *
def data(n):
	x = np.random.randint(10001, size = n)
	return x
n = int(input())
arr = data(n)
print(arr)
def quicksort(a,low = 0, high = n-1):
	if high <= low:
		return
	j = low - 1
	pivot = a[high]
	for i in range(low,high):
		if a[i] < pivot:
			j=j+1
			(a[i],a[j])=(a[j],a[i])
	(a[j+1],a[high])=(a[high],a[j+1])
	quicksort(a,low,j)
	quicksort(a,j+2,high)
quicksort(arr)
print(arr)
