import numpy as np
from math import sin
import matplotlib.pyplot as plt

def data1():
    x = [np.random.rand() for i in range(1000)]
    y = [10*x[i] + 0.05*np.random.rand() for i in range(1000)] #Enter your data here
    x_ = np.mean(x)
    y_ = np.mean(y)
    x_std = x - x_
    y_std = y - y_
    return [x_std, y_std], [x, y]

def data2():
    import numpy as np
from math import sin
import matplotlib.pyplot as plt

def data1():
    x = [np.random.rand() for i in range(1000)]
    y = [10*x[i] + 0.05*np.random.rand() for i in range(1000)] #Enter your data here
    x_ = np.mean(x)
    y_ = np.mean(y)
    x_std = x - x_
    y_std = y - y_
    return [x_std, y_std], [x, y]

def data2():
    x = [np.random.rand() for i in range(1000)]
    y = [(x[i])**2 + 0.05*np.random.rand() for i in range(1000)] #Enter your data here
    x_ = np.mean(x)
    y_ = np.mean(y)
    x_std = x - x_
    y_std = y - y_
    return [x_std, y_std], [x, y]

data_set, org_data_set = np.array(data2())
cov_data = np.cov(data_set)
eigen_data = np.linalg.eigh(cov_data)
u = eigen_data[1][:][-1]
c = np.mean(org_data_set[1] - org_data_set[0]*u[1]/u[0])
plt.scatter(org_data_set[0], org_data_set[1], color = "red")
plt.plot(org_data_set[0], org_data_set[0]*u[1]/u[0] + c)
print(u, c)
plt.title("Best Fit Line")
plt.show()


data_set, org_data_set = np.array(data1())
cov_data = np.cov(data_set)
eigen_data = np.linalg.eigh(cov_data)
u = eigen_data[1][:][-1]
c = np.mean(org_data_set[1] - org_data_set[0]*u[1]/u[0])
plt.scatter(org_data_set[0], org_data_set[1], color = "red")
plt.plot(org_data_set[0], org_data_set[0]*u[1]/u[0] + c)
print(u, c)
plt.title("Best Fit Line")
plt.show()
