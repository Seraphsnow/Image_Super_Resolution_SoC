import numpy as np
import matplotlib.pyplot as plt

def data1():
    x=[np.random.rand() for i in range(1000)]
    y=[x[i]+0.05*np.random.rand() for i in range(1000)]

    return [x,y]

def data2():
    x=[np.random.rand() for i in range(1000)]
    y=[x[i]**2 + 0.05*np.random.rand() for i in range(1000)]

    return [x,y]

def line(arr):
    x,y=arr
    xm,ym=np.mean(x),np.mean(y)
    
    std_x,std_y=np.std(x),np.std(y)

    x=(x-xm)/std_x
    y=(y-ym)/std_y

    new_arr=np.array([x,y])  #creating modified array of x and y

    cov_arr=np.cov(new_arr)
    e_val,e_vector=np.linalg.eig(cov_arr)  #for finding the eigen values and eigen vectors of the covariance matrix
    
    e_val_max=e_val[-1]
    e_vector_max=e_vector[:][-1]

    plt.scatter(arr[0],arr[1])   #plotting the initial points

    plt.plot(arr[0],ym+x*std_y*e_vector_max[1]/e_vector_max[0],color='r')  #here e_vector_max is the eigen vector whose eigen value is the maximum and e_vector_max[1]/e_vector_max[0] gives the slope of the line(after scaling and shifting)

    font1 = {'family':'serif','color':'blue','size':20}
    font2 = {'family':'serif','color':'darkred','size':15}

    plt.xlabel('x-axis',fontdict=font2)
    plt.ylabel('y-axis',fontdict=font2)

    plt.title('PCA- Best Fit Line',fontdict=font1)

    plt.show()

line(data1())
line(data2())
