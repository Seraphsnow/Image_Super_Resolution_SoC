import numpy as np
import matplotlib.pyplot as plt

t = np.array([73, 91, 87, 102, 69, 74, 91, 88, 101, 68, 73])
h = np.array([43, 64, 58, 37, 70, 43, 65, 59, 37, 71, 44])
r = np.array([67, 88, 134, 43, 96, 66, 87, 134, 44, 96, 66])
m = np.array([56, 81, 119, 22, 103, 57, 80, 118, 21, 104, 57])
t1 = [np.mean(t), np.std(t)]
r1 = [np.mean(r), np.std(r)]
h1 = [np.mean(h), np.std(h)]
m1 = [np.mean(m), np.std(m)]
t = (t-np.mean(t))/np.std(t)
r = (r-np.mean(r))/np.std(r)
h = (h-np.mean(h))/np.std(h)
m = (m-np.mean(m))/np.std(m)
# c=np.zeros()
c1 = 0.3
c2 = 0.3
c3 = 0.3
m_p = c1*t + c2*h + c3*r
# print(m)
# print(m_p)
cost = (np.sum((m_p - m)**2))/(11)
for i in range(1000):
    c1n = c1 - 0.001*((np.dot(t, (m_p - m)))/(11))
    c2n = c2 - 0.001*((np.dot(h, (m_p - m)))/(11))
    c3n = c3 - 0.001*((np.dot(r, (m_p - m)))/(11))
    # print(cost)
    # bn = b - 0.01*((np.sum((m_p - m)))/(11))
    c1 = c1n
    c2 = c2n
    c3 = c3n
    # b = bn
    m_p = c1*t + c2*h + c3*r
    cost = (np.sum((m_p - m)**2))/(11)
    # if(abs(prevcost-cost)<0.0000001):
    #   print("i: ",i)
    #   break
    # prevcost=cost
# print(m_p)
print(cost)
plt.plot(m, label="line 2")
plt.plot(m_p, label="line 1")
plt.show()
