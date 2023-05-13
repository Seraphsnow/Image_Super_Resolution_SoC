import numpy as np
import matplotlib.pyplot as plt


def data1():
    x = [np.random.rand() for i in range(1000)]
    y = [x[i] + 0.05*np.random.rand() for i in range(1000)]
    return [x, y]

def data2():
    x = [np.random.rand() for i in range(1000)]
    y = [(x[i])**2 + 0.05*np.random.rand() for i in range(1000)]
    return [x, y]

inp = data2()
xm=ym=m_den=m_num=0
for i in range(1000):
    xm += inp[0][i]
    ym += inp[1][i]
    
    xm/=1000
    ym/=1000

for i in range(1000):
    m_den += (inp[0][i] - xm)**2

    m_num += (inp[0][i] - xm) * (inp[1][i] - ym)
    
m= m_num/m_den

c= ym - m * xm

inx = inp[0]
iny = inp[1]

outx = np.linspace(min(inx), max(inx), 1000)
outy = m * outx + c 

plt.plot(inx, iny, color = 'b')
plt.plot(outx, outy, ls= ':', color = 'r')
plt.show()




        




