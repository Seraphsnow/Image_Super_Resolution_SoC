import numpy as np
import csv
import math

to = []
with open('framingham.csv', mode = 'r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        if 'NA' in lines:
            continue
        else:
            to.append(lines)
total = []
for i in range(1,3000):
    total.append(list(map(float,np.array(to[i]))))
total = np.array(total)

var = np.var(total, axis=0)
mean = np.mean(total, axis=0)

totals = (total - mean)/var
totals = np.transpose(totals)
total  = np.transpose(total)

onlydata = np.delete(totals, 15, 0)

def sigm(z):
    return 1/(1+math.exp(-z))

def loss(x,y):
    x = max(x, 1e-15)
    x = min(x, 1-1e-15)
    return float(y*(math.log(x)) + (1-y)*(math.log(1-x)))


coeff = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
const = 1

lr = 0.1

cost_d = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
cost_d = np.array(list(map(float,cost_d)))
sum_cost_d = 10
while sum_cost_d > 0.015:
    sum_temp = sum_cost_d
    predn = np.dot(coeff, onlydata)

    for i in range(0,15):
        x = 0
        for j in range(0, 2999):
            x = x + (sigm(predn[j] + const) - total[15][j]) * totals[i][j]

        #print(x)
        cost_d[i] = float(x / 2999)
        #print (cost_d[i])

    y=0
    for i in range(0, 2999):
        y += (sigm(predn[i] + const) - total[15][i])

    const = const - lr * y/2999
    coeff = coeff - lr * cost_d
    sum_cost_d = sum(list(map(abs,cost_d)))
    print(sum_cost_d)
    if sum_temp< sum_cost_d:
        lr = lr/1.1
    

print("The coefficients are: ")
for i in range(15):
    print(coeff[i])
print(const)


        

    

    