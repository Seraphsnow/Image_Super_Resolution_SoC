import numpy as np
import math

# Code to generate n datapoints, returns an np array of n numbers ranging from 0
# to 10000
def data(n):
	x = np.random.randint(10001, size = n)
	return x
n=int(input())
x=data(n)
print("array=",x)
#sum
def add(x):
     return sum(x)
#Mean	
def mean(x):
    mean= sum(x)/len(x)
    return mean
#Quick sort
def split_array(x,low,high):
      pivot=x[high]
      i=low-1
      for j in range(low,high):
            if x[j]<pivot:
                i+=1
                x[j],x[i]=x[i],x[j]
      x[high],x[i+1]=x[i+1],x[high]
      return i+1
def quick_sort(x,low,high):
  if high-low<1:
       return 
  partition=split_array(x,low,high)
  quick_sort(x,low,partition-1)
  quick_sort(x,partition+1,high)
quick_sort(x,0,len(x)-1)
print("sorted array=",x)
#median
def median(x):
     m=len(x)//2
     if len(x)%2==0:
          return (x[m-1]+x[m])/2
     return x[m]
#mode
def mode(x):
     c=[0]*(len(x))
     for i in range(len(x)):
          for j in range(1,len(x)):
               if i!=j and x[j]==x[i]:
                    c[i]+=1
     m=max(c)
     modes=[x[i] for i in range(len(x)) if c[i]==m]
     return modes 
#std deviation
def std_dev(x):
       sd=math.sqrt(sum((x[i]-mean(x))**2 for i in range(len(x))))
       sd=sd/len(x)
       return sd    
def var(x):
      return std_dev(x)**2                     
print("sum=",add(x))
print("mean=",mean(x))
print("median=",median(x))
print("mode=",mode(x))
print("standard deviation=",std_dev(x))
print("variance=",var(x))

      
