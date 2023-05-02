import numpy as np
def data(n):
    x = np.random.randint(10001, size = n)
    return x

#give an input integer to create and give corresponding values for the array

num=int(input())
arr = data(num)
summation = np.sum(arr)
mean = np.mean(arr)
variance = np.var(arr)
std = np.std(arr)
n = len(arr)
median = arr[int(n/2)]
mode = arr[0]
for x in arr:
    if(np.count_nonzero(arr == x) > np.count_nonzero(arr == median)): 
        median = x

print(arr)
print(mean)
print(median)
print(mode)
print(summation)
print(variance)
print(std)








