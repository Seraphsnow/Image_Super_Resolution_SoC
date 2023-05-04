# To FIND MEAN, MEDIAN, MODE, SUM, VARIANCE, STANDARD DEVIATION OF A NUMPY ARRAY

import numpy as np

def data(n):
  x = np.random.randint(10001,size=n)
  return x
  
def solve(x):
  mean = np.mean(x)
  median = np.median(x)
  
  x_unique, counts = np.unique(x,return_counts=True) 
  i = np.argmax(counts)
  mode = x_unique[i]
  
  _sum = np.sum(x)
  var = np.var(x)
  std = np.std(x)
  
  print("Array:",x)
  print("\nMean =", mean)
  print("Median =", median)
  print("Mode =",mode)
  print("Sum =", _sum)
  print("Variance =", var)
  print("Standard deviation =",std)  
  

n = int(input())
solve(data(n))
