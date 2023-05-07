import numpy as np



# Code to generate n datapoints, returns an np array of n numbers ranging from 0

# to 10000

def data(n):

	x = np.random.randint(10001, size = n)

	return x





def sort(x):

    if(x.__len__()==1):

        return x

    if(x.__len__()==0):

        return []

    piv=x[0]

    i=1

    lo=[]

    hi=[]

    while i<x.__len__():

        if(x[i]<piv):

            lo.append(x[i])

        else:

            hi.append(x[i])

        i+=1

    lo=sort(lo)

    hi=sort(hi)

    # print(lo.__len__())

    # print(hi.__len__())

    return lo+[piv]+hi 

size=int(input("give me size of data:"))

n=data(size)

p=sort(n)

print("unsorted Array is :",n)

print("Sorted Array is :",p)

median=0

if(size%2==0):

     median=(p[int(size/2)]+p[int(size/2)-1])/2

else:

     median=p[int(size/2)]

print("Median of given Array is:",median)