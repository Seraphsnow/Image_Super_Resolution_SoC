# Implementing Quicksort

import numpy as np

def data(n):
  x = np.random.randint(10001,size=n)
  return x
        
def quicksort(arr):
    if(len(arr)==0 or len(arr)==1):
      return(arr)
    i = -1
    for j in range(0,len(arr)):
      if(arr[j] < arr[-1]):
        i += 1
        arr[i], arr[j] = arr[j], arr[i]
  
    arr[i+1], arr[-1] = arr[-1], arr[i+1]
    quicksort(arr[0:i+1])
    quicksort(arr[i+2:])
    return(arr)
    
n = int(input())
x = data(n)
print("Original array:\n",x)
quicksort(x)
print("\nSorted array:\n",x)
