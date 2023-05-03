import numpy as np
from mean import mean

# Code to generate n datapoints, returns an np array of n numbers ranging from 0
# to 10000


def data(n):
    x = np.random.randint(10001, size=n)
    return x


x = int(input())
list = data(x)
print(list)
sum_of_x_nos = sum(list)
mean_of_x_nos = mean(sum_of_x_nos, x)
# mode_of_x_nos = mode(list, x)
# variance_of_x_nos = variance(list, x)
# sd_of_x_nos = sd(list, x)
print(sum_of_x_nos)
print(mean_of_x_nos)
