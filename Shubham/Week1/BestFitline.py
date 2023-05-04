import numpy as np
import matplotlib.pyplot as plt

def sum(list):
    avg = 0
    for i in list:
        avg = avg + i
    return avg
def bfl(x,y):
    xpoints = np.array(x)
    ypoints = np.array(y)
    plt.scatter(x, y, color="red")
    xpoints_mean, ypoints_mean = np.mean(xpoints), np.mean(ypoints)
    x2 = xpoints-xpoints_mean
    y2 = ypoints-ypoints_mean
    x2y2 = x2*y2
    x2x2 = x2*x2
    x2y2_sum = sum(x2y2)
    x2x2_sum = sum(x2x2)
    m = x2y2_sum/x2x2_sum
    b = ypoints_mean - m*xpoints_mean
    x_cd = np.array([np.random.rand() for i in range(1000)])
    y_cd = x_cd*m + b
    plt.plot(x_cd, y_cd)
    plt.title("Best Fit Line")
    plt.show()

x = [np.random.rand() for i in range(1000)]
y = [x[i]**2 + 0.05*np.random.rand() for i in range(1000)]
bfl(x,y)

