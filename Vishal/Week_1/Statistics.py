import numpy as np

# Code to generate n datapoints, returns an np array of n numbers ranging from 0
# to 10000
def data(n):
	x = np.random.randint(10001, size = n)
	return x	

n=int(input("Enter the number of elements in the arrays:\n"))   #the number of elements in the array
a=data(n)  #The array storing the values

s=np.sum(a)

mean=np.mean(a)

median=np.median(a)

variance=np.var(a)

std_dev=np.std(a)

mode_dict={}

for i in a:
    mode_dict.setdefault(i,0)
    mode_dict[i]+=1

max_val=max(mode_dict.values())  #to find the maximum value in the dictionary

max_val_lst=[] #creating an empty list

for i,j in mode_dict.items():
    if j==max_val:
        max_val_lst.append(i)

mode=max_val_lst
print(f"Sum : {s}\n")
print(f"Mean : {mean}\n")
print(f"Median : {median}\n")
print(f"Mode : {mode}\n")
print(f"Variance : {variance}\n")
print(f"Standard Deviation : {std_dev}\n")
