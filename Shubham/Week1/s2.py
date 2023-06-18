import numpy as np
import matplotlib

t = np.array([73, 91, 87, 102, 69, 74, 91, 88, 101, 68, 73])
h = np.array([43, 64, 58, 37, 70, 43, 65, 59, 37, 71, 44])
r = np.array([67, 88, 134, 43, 96, 66, 87, 134, 44, 96, 66])
m = np.array([56, 81, 119, 22, 103, 57, 80, 118, 21, 104, 57])
# b = 100
c1 = 1
c2 = 1
c3 = 1
m_p = c1*t + c2*h + c3*r
cost = (np.sum((m_p - m)**2))/(2*11)
while(-100>=cost or cost>=100):
    c1n = c1 - 0.01*c1*((np.sum((m_p - m)))/(11))
    c2n = c2 - 0.01*c2*((np.sum((m_p - m)))/(11))
    c3n = c3 - 0.01*c3*((np.sum((m_p - m)))/(11))
    c1 = c1n
    c2 = c2n
    c3 = c3n
    m_p = c1*t + c2*h + c3*r
    cost = (np.sum((m_p - m)**2))/(2*11)
print(m_p)
print(cost)
