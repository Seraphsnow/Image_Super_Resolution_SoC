import numpy as np

# Code to generate n datapoints, returns an np array of n numbers ranging from 0
# to 10000
def data(n):
    x = np.random.randint(10001, size = n)
    y = np.sort(x)
    return x

def partition(arr, low, high):
    pivot=arr[high]

    i=low-1
    j=low
    while j<high:
        if arr[j] <= pivot:
            i+=1
            (arr[i],arr[j])=(arr[j],arr[i])
        j+=1
    (arr[i+1],arr[high])=(arr[high],arr[i+1])

    return i+1

def qsort(arr, low, high):
    if(high>low):
        pivot=partition(arr,low,high)

        qsort(arr,low,pivot-1)
        qsort(arr,pivot+1,high)

n=int(input("Length of array: "))
x=data(n)
qsort(x,0,n-1)
print(x)

