import numpy as np

import math



# Code to generate n datapoints, returns an np array of n numbers ranging from 0

# to 10000

def data(n):

	x = np.random.randint(10001, size = n)

	return x



n=int(input("size of data:"))

x=data(n)

print(x)

count=0

for i in range(n):

	count+=x[i]

print(f"sum : {count}")

print(f"mean : {count/n}")





y=[]

for i in range(n):

	y.append(0)



for i in range(n):

	for j in range(i,n):

		if(x[i]==x[j]):

			y[i]+=1



modeidx=0



for i in range(n):

	if(y[i]>y[modeidx]):

		modeidx=i



print(f"mode : {x[modeidx]}")



mean=count/n

var=0

for i in range(n):

	dff=abs(x[i]-mean)

	var+=dff*dff



print(f"variance : {var/n}")

print(f"standard daviation : {pow(var/n,0.5)}")