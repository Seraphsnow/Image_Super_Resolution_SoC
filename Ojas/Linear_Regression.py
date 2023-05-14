import numpy as np 
import matplotlib.pyplot as plt

temp = np.array([73, 91, 87, 102, 69, 74, 91, 88, 101, 68, 73, 92, 87, 103, 68])
rain = np.array([67, 88, 134, 43, 96, 66, 87, 134, 44, 96, 66, 87, 135, 43, 97])
hmd  = np.array([43, 64, 58, 37, 70, 43, 65, 59, 37, 71, 44, 64, 57, 36, 70])

total = np.array([[73, 91, 87, 102, 69, 74, 91, 88, 101, 68, 73, 92, 87, 103, 68], 
                 [67, 88, 134, 43, 96, 66, 87, 134, 44, 96, 66, 87, 135, 43, 97],
                 [43, 64, 58, 37, 70, 43, 65, 59, 37, 71, 44, 64, 57, 36, 70] ])

mngo = np.array([56, 81, 119, 22, 103, 57, 80, 118, 21, 104, 57, 82, 118, 20, 102])
orng = np.array([70, 101, 133, 37, 119, 69, 102, 132, 38, 118, 69, 100, 134, 38, 120])

coeff = np.array([1, 1, 1])
mc=1
oc=0
lr=0.000001

jd = np.array([10, 10, 10])

while jd[0] > 0.00001 :
    predn = np.dot(coeff, total)

    for i in range(0, 3):
        x=0

        for j in range(0,15):
            x = x + (predn[j] - mngo[j]) * total[i][j]
        
        jd[i] = x/15

    y=0
    for i in range(0,15):
        y += (predn[j] - mngo[j])

    mc = mc - lr * y/15
    coeff = coeff - lr * jd

    

print("The coefficients are: ")
for i in range(3):
    print(coeff[i])
print(mc)
