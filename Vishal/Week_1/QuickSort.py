import numpy as np

# Code to generate n datapoints, returns an np array of n numbers ranging from 0
# to 10000
def data(n):
	x = np.random.randint(10001, size = n)
	return x	


def quicksort(x,low,high):
    if high-low < 1:
        return
    pivot=x[high]
    left=low
    right=high-1
    while left!=right:
        if x[left] < pivot:
            left+=1
        elif x[right] > pivot:
            right-=1
        elif x[right]<=pivot and x[left]>=pivot:
            x[right],x[left]=x[left],x[right]

    

    if x[right] > pivot:
            x[right],x[high]=x[high],x[right]
    quicksort(x,low,right)
    quicksort(x,right+1,high)

n=int(input("Enter the number of elements in the array:\n"))
x=data(n)
print(f"The array before sorting is : {x}\n")
quicksort(x,0,n-1)

print(f"The array after sorting is : {x}")

