import numpy as np

# Code to generate n datapoints, returns an np array of n numbers ranging from 0
# to 10000
def data(n):
    x = np.random.randint(10001, size = n)
    y = np.sort(x)
    return y

n=int(input("Number of DataPoints: "))
num = data(n)
mean = np.mean(num)
if(n%2==1):
    median = num[int((n-1)/2)]
    

else:
    median = (num[int(n/2)] + num[int(n/2 -1)])/2

uniqvals,counts = np.unique(num, return_counts=True)
mode = uniqvals[np.argmax(counts)]

standard_deviation= np.std(num)
variance= standard_deviation*standard_deviation


print(num)
print("Mean = ", mean)
print("Median = ", median)
print("Mode = ", mode)
print("Standard Deviation = ", standard_deviation)
print("Variance = ", variance)

