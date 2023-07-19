import numpy as np
import math
import csv
import pandas as pd

listed_A = []
listed_y = []

part1 = []
y_part1 = []
part2 = []
y_part2 = []
part3 = []
y_part3 = []

with open("framingham.csv", "r") as file:
    csvFile = csv.reader(file)
    flag = False
    for lines in csvFile:
        if flag:    
            listed_y.append(lines[-1])
            list1 = [1] + lines[:-1]
            listed_A.append(list1)
        flag = True

part_size_1 = int(0.7*4239)
part_size_2 =  int(0.85*4239)

for i in range(len(listed_A)):
    if i < part_size_1:
        part1.append(listed_A[i])
        y_part1.append(listed_y[i])
    elif i < part_size_2:
        part2.append(listed_A[i])
        y_part2.append(listed_y[i])
    else:
        part3.append(listed_A[i])
        y_part3.append(listed_y[i])
#DATA SPLITTED

def p(x_k, x):
    z = np.array([x_k])
    z.astype("float64")
    x.astype("float64")
    return 1/(1+math.exp(-1*(z@x)))


def cost(A, x, y):
    ans = 0
    for i in range(len(y)):
        if int(y[i]) == 1:
            ans += math.log(p(A[i], x))
        else:
            ans += math.log(1 - p(A[i], x))
    return math.fabs(ans)

def del_cost(A, x, y):
    ans = 0
    for i in range(len(A)):
       ans += (y[i] - p(A[i], x))*A[i]
    return ans

x = np.array([0.5 for i in range(16)]).T
A = []
B = []
C = []
y = []
y_B = []
y_C = []
for i in range(len(part1)):
    l1 = []
    if "NA" in part1[i]:
        continue
    else:
        y.append(float(y_part1[i]))
        for j in part1[i]:
            l1.append(float(j))
    A.append(l1)
avg1 = np.average(A, axis=0)
std1 = np.std(A, axis=0)
avg1[0] = 1
std1[0] = 1
A -= avg1
A /= std1
for i in A:
    i[0] = 1
for i in B:
    i[0] = 1
for i in C:
    i[0] = 1
for i in range(len(part2)):
    l2 = []
    if "NA" in part2[i]:
        continue
    else:
        y_B.append(float(y_part2[i]))
        for j in part2[i]:
            l2.append(float(j))
    B.append(l2)
avg1 = np.average(B, axis=0)
std1 = np.std(B, axis=0)
avg1[0] = 1
std1[0] = 1
B -= avg1
B /= std1

for i in range(len(part3)):
    l3 = []
    if "NA" in part3[i]:
        continue
    else:
        y_C.append(float(y_part3[i]))
        for j in part3[i]:
            l3.append(float(j))
    C.append(l3)
avg1 = np.average(C, axis=0)
std1 = np.std(C, axis=0)
avg1[0] = 1
std1[0] = 1
C -= avg1
C /= std1

learning_rate = 1e-2
cost_curr = cost(A, x, y)

while True:
    x_new = x + learning_rate*del_cost(A, x, y)
    cost_new = cost(A, x_new, y)
    if cost_new > cost_curr and learning_rate >= 1e-20:
        learning_rate /= 10
    elif cost_new > cost_curr or cost_curr - cost_new < 1e-9:
        break
    else:
        cost_curr = cost_new 
        x = x_new

while True:
    x_new = x + learning_rate*del_cost(B, x, y_B)
    cost_new = cost(B, x_new, y_B)
    if cost_new > cost_curr and learning_rate >= 1e-20:
        learning_rate /= 10
    elif cost_new > cost_curr or cost_curr - cost_new < 1e-9:
        break
    else:
        cost_curr = cost_new 
        x = x_new


count_i = 0
t_count_i = 0

for i in range(len(A)):
    if (np.around(p(A[i], x), 2) > 0.5 and y[i] == 1) or (np.around(p(A[i], x), 2) < 0.5 and y[i] == 0):
        count_i += 1
        
    t_count_i += 1
print(count_i/t_count_i)

count_i = 0
t_count_i = 0

for i in range(len(B)):
    if (np.around(p(B[i], x), 2) > 0.5 and y_B[i] == 1) or (np.around(p(B[i], x), 2) < 0.5 and y_B[i] == 0):
        count_i += 1
        
    t_count_i += 1
print(count_i/t_count_i)

count_i =0
t_count_i = 0

for i in range(len(C)):
    if (np.around(p(C[i], x), 2) > 0.5 and y_C[i] == 1) or (np.around(p(C[i], x), 2) < 0.5 and y_C[i] == 0):
        count_i += 1
    t_count_i += 1
print(count_i/t_count_i)

print(x)
'''
[-1.97759041  0.24023394  0.5088232  -0.05640906  0.01826714  0.264081
 -0.00441572  0.04440023  0.08388741  0.01632737  0.09233665  0.33179291
 -0.02484562  0.08054833 -0.07209893  0.20748172]
 '''