import statistics as st
import numpy as np
n=int(input())
def data(n):
	x = np.random.randint(10001, size = n)
	return x
arr = data(n)
print(arr)
sum=np.sum(arr)
mode=st.mode(arr)
median=np.median(arr)
mean=np.mean(arr)
variance=np.var(arr)
standard_deviation=np.std(arr)
print("sum is : ",sum,"\nmean is : ",mean,"\nmode is : ",mode,"\nmedian is : ",median,"\nvariance is : ",variance,"\nstandard deviation is : ",standard_deviation)
