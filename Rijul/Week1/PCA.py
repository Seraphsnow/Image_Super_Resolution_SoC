import numpy as np
import matplotlib.pyplot as plt

def data1():
    x = [np.random.rand() for i in range(1000)]
    y = [x[i] + 0.05*np.random.rand() for i in range(1000)]
    return [x, y]

def data2():
    x = [np.random.rand() for i in range(1000)]
    y = [(x[i])**2 + 0.05*np.random.rand() for i in range(1000)]
    return [x, y]

# Standardisation of Data
def std_data(nparray):
    x, y = nparray[0], nparray[1]
    x_std = (x - np.mean(x))/np.std(x)
    y_std = (y - np.mean(y))/np.std(y)
    return np.array([x_std, y_std])

def DimReduction(arr):
    data_set = np.array(arr)
    std_data_set = std_data(data_set)
    
    # Computing the covariance matrix
    cov_matrix = np.cov(std_data_set)
    
    # Computing the eigenvectors and eigenvalues of the covariance matrix to identify the principal components
    eigen_data = np.linalg.eigh(cov_matrix)
    
    # Creating a feature vector to decide which principal components to keep
    u = eigen_data[1][:][-1] # Retaining the eigenvector corresponding to greatest eigenvalue
    
    # Recasting the data along the principal components axes
    m = u[1]/u[0]*np.std(data_set[1])/np.std(data_set[0])
    c = np.mean(data_set[1] - data_set[0]*m)
    
    # Displaying the result using matplotlib
    plt.scatter(data_set[0], data_set[1], color = "red")
    plt.plot(data_set[0], data_set[0]*m + c)
    print("Slope =", m, "Intercept =", c)
    plt.title("Best Fit Line")
    plt.show()

DimReduction(data1())
DimReduction(data2())