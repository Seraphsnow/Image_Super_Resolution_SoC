
# Perform PCA to identify best-fit line through the following datasets. Utilize the matplotlib library to display the results. 

import numpy as np
import matplotlib.pyplot as plt

def data1():
  x = [np.random.rand() for i in range(1000)]
  y = [x[i] + 0.05*np.random.rand() for i in range(1000)]
  return [x,y]
  
def data2():
  x = [np.random.rand() for i in range(1000)]
  y = [x[i]**2 + 0.05*np.random.rand() for i in range(1000)]
  return [x,y]

def bestfitline(arr):
  x, y = arr
  x_mean, y_mean = np.mean(x), np.mean(y)
  x_std, y_std = np.std(x), np.std(y)
  x = (x - x_mean)/x_std								# Standardization (Shifting origin to mean, and scaling x and y axes)
  y = (y - y_mean)/y_std
  nparr = np.array([x,y])
  
  cov_nparr = np.cov(nparr)								
  eigenvalues, eigenvectors = np.linalg.eigh(cov_nparr)					# Finding the eigenvalues and eigenvectors of the symmetric covariance matrix 
  
  eigenvalue_max = eigenvalues[-1]							# Finding maximum eigenvalue and eigenvector associated with it
  eigenvector_max = eigenvectors[:][-1]
  
  plt.scatter(arr[0], arr[1], color="red")
  plt.plot(arr[0] , y_mean + x*eigenvector_max[1]*y_std/(eigenvector_max[0]))		# Plotting best fit line
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title("PCA - Best Fit Line")
  plt.show()
  
bestfitline(data2())
