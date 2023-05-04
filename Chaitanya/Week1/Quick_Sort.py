import numpy as np
def data(n):
    x = np.random.randint(10001, size = n)
    return x
def partition(array, low, high):
    pivot = array[high]
    i = low
    for j in range(low, high):
        if array[j] <= pivot:
            (array[i], array[j]) = (array[j], array[i])
            i = i+1
    (array[i], array[high]) = (array[high], array[i])
    return i

def quicksort(array, low, high):
    if low<high:
        pi = partition(array, low, high)
        quicksort(array, low, pi-1)
        quicksort(array, pi+1, high)

    


num = int(input())
arr = data(num)
print(arr)

quicksort(arr,0, num-1)   

print(arr)