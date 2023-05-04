import numpy as np
import math

# Code to generate n datapoints, returns an np array of n numbers ranging from 0
# to 10000


def data(n):
    x = np.random.randint(9, size=n)
    return x


def sum(list):
    avg = 0
    for i in list:
        avg = avg + i
    return avg


def mean(sum, n):
    avg = sum/n
    return avg


def variance(list, n):
    var = 0
    av = sum(list)
    avg = mean(av, n)
    for i in range(n):
        var += ((list[i] - avg)**2)
    var = var/n
    return var


def partition(list, l, r):
    n = list[r]
    a = l-1
    for i in range(l, r):
        if n >= list[i]:
            a += 1
            (list[a], list[i]) = (list[i], list[a])
    (list[a+1], list[r]) = (list[r], list[a+1])
    return a+1


def quick_sort(list, l, r):
    if r > l:
        pi = partition(list, l, r)
        quick_sort(list, l, pi-1)
        quick_sort(list, pi+1, r)


def median(list, x):
    quick_sort(list, 0, x-1)
    print(list)
    if x % 2 != 0:
        return list[int((x-1)/2)]
    else:
        return (list[int(x/2)] + list[int(x/2) - 1])/2


def mode(list):
    array = np.array(list)
    vals, counts = np.unique(array, return_counts=True)
    index = np.argmax(counts)
    return (vals[index])


x = int(input())
list = data(x)
print(list)
sum_of_x_nos = sum(list)
mean_of_x_nos = mean(sum_of_x_nos, x)
median_of_x_nos = median(list, x)
var_of_x_nos = variance(list, x)
sd_of_x_nos = math.sqrt(var_of_x_nos)
mode_of_x_nos = mode(list)
print(sum_of_x_nos)
print(mean_of_x_nos)
quick_sort(list, 0, x-1)
print(median_of_x_nos)
print(var_of_x_nos)
print(sd_of_x_nos)
print(mode_of_x_nos)
