import numpy as np
import array as ar

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        left = [x for x in arr[1:] if x < pivot]
        right = [x for x in arr[1:] if x >= pivot]
        return quicksort(left) + [pivot] + quicksort(right)
 
n= int(input())
x = np.random.randint(10001, size = n)
print(x)
sorted_arr = quicksort(x)
print("Sorted Array in Ascending Order:")
print(sorted_arr)