import numpy as np
import matplotlib.pyplot as plt
import math

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
        elif int(y[i]) == 0:
            ans += math.log(1 - p(A[i], x))
    return math.fabs(ans)

def del_cost(A, x, y):
    ans = 0
    for i in range(len(A)):
        ans += (y[i] - p(A[i], x))*A[i]
    return ans

y_listed = []
A_listed = []

x = np.array([1])
A = []
y = []
for i in range(len(A_listed)):
    l1 = []
    if "NA" in A_listed[i]:
        continue
    else:
        y.append(float(y_listed[i]))
        for j in A_listed[i]:
            l1.append(float(j))
    A.append(l1)
y = np.array(y)
y.astype("float64")
A = []
for i in range(100):
    A.append([np.random.rand()])
A = [[np.random.rand()] for i in range(100)]
y = [1 if A[i][0] > 0.1 else 0 for i in range(100)]
A = np.array(A)
learning_rate = 4
cost1 = cost(A, x, y)
while True:
    x_new = x + learning_rate*del_cost(A, x, y)
    cost_new = cost(A, x_new, y)
    if cost_new > cost1 and learning_rate >= 1e-15:
        learning_rate /= 1.4
    elif cost_new > cost1 or cost1 - cost_new < 1e-5:
        break
    else:
        cost1 = cost_new 
        x = x_new
print(x)
B = [p(A[i], x) for i in range(100)]
X = [A[i][0] for i in range(100)]
plt.scatter(X, y)
X.sort()
B.sort()
plt.plot(X, B)
plt.show()
'''
 [-0.05460261  0.07450385  0.02600857 -0.15739937 -0.02844317  0.01666762
  0.02914396  0.01334252  0.14713603  0.03033247 -0.00244233  0.02282473
 -0.0256675  -0.05080773 -0.03035339  0.0043388 ]'''
