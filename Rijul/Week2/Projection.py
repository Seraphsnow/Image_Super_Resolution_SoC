import numpy as np
import matplotlib.pyplot as plt
# Assumed linear regression
# Now I think I should assume instead that they are products of functions of temperature, rainfall...
listed_A = [[1, 73, 67, 43], [1, 91, 88, 64], [1, 87, 134, 58], [1, 102, 43, 37], 
            [1, 69, 96, 70], [1, 74,66,43], [1, 91, 87, 65], [1, 88, 134, 59], [1, 101, 44,37],
           [1, 68, 96, 71], [1, 73, 66, 44], [1, 92, 87, 64], [1, 87,135,57]]
for i in listed_A:
    i.append(i[1]*i[2])
    i.append(i[2]*i[3])
    i.append(i[3]*i[1])
    i.append(i[1]*i[2]*i[3])
A = np.array(listed_A)
x = np.array([[1, 1, 1, 1, 1, 1, 1, 1]]).T
listed_b = [i[1]+i[2]+i[3] for i in listed_A]
b = np.array([[56, 81, 119, 22, 103, 57, 80, 118, 21, 104, 57, 82, 118]]).T
learning_rate = 1e-8
cost =  (A@x-b).T @ (A@x-b)
x = np.linalg.inv(A.T@A)@A.T@b
cost =  (A@x-b).T @ (A@x-b)
print(cost)
print(x)
''' [[ 2.7835065 ]
 [-0.43408107]
 [ 0.84057385]
 [ 0.69774067]]'''