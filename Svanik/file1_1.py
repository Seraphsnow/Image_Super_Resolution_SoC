
import numpy as np
import statistics
# Code to generate n datapoints, returns an np array of n numbers ranging from 0
# to 100
n= int(input())
x = np.random.randint(10001, size = n)
print(x)
arr1=np.mean(x)
print("mean is :",arr1)
arr2=np.median(x)
print("median is :",arr2)
arr3=statistics.mode(x)
print("mode is :",arr3)
arr4=np.sum(x)
print("sum is  :",arr4)
arr5=np.var(x)
print("variance is :",arr5)
arr6=np.std(x)
print("standard deviation is :",arr6)

    

