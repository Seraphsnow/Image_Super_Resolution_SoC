
import numpy as np
import matplotlib.pyplot as plt


x = [np.random.rand() for i in range(1000)]
y = [x[i] + 0.05*np.random.rand() for i in range(1000)]



slope,intercept = np.polyfit(x,y,1)
print (slope)
print (intercept)
best_fit_line=np.array([])
for xi in x:
    best_fit_line = np.append(best_fit_line,slope*xi+ intercept)
plt.plot(x,y,'o',ms=0.5,)
plt.plot(x, best_fit_line, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Best_fit_line')
plt.show()
			    
