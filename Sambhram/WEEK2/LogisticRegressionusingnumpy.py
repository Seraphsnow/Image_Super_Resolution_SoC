import numpy as np
import math 
file_path = 'framingham.csv'
data = np.genfromtxt(file_path, delimiter=',')

theta0 = np.zeros((16,1))

ones = np.ones((4239,1))

data = np.column_stack((data, ones)) #need a np.ones column in the entire array

y = data[:,15] #making the output ready 

data = np.delete(data, 0, axis=0)
data = np.delete(data, 15, axis = 1)
y = np.delete(y, 0, axis = 0)


alpha = 0.00000116342 # learning rate

data = np.nan_to_num(data, nan=0)
error_sum = 0 

for i in range(0,500):
    for k in range(0,500):
        z = np.clip(data[k] @ theta0,-500,500)
        error_sum += 1/4238 * (1/(1+ np.exp(-(data[k] @ theta0))) - y[k] )
        
      
    for j in range(0,16):
        theta0[j] = theta0[j] - alpha * error_sum * data[i,j]
        
       
    
print(1/(1+ np.exp(-(data[9] @ theta0))))
print(y[3])




