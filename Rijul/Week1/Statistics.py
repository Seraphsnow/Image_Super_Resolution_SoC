import numpy as np

def data(n):
    x = np.random.randint(10001, size = n)
    return x
n = 20 # Value of n goes here
data_set = data(n)
print("Data Set: ", data_set)

# MEAN
print("Mean:", np.mean(data_set))

#MEDIAN
print("Median:", np.median(data_set))

#MODE
vals, counts = np.unique(data_set, return_counts=True)
mode = vals[np.argmax(counts)]
print("Mode:", mode)

#SUM
print("Sum:", np.sum(data_set))

#VARIANCE
print("Var:", np.var(data_set))

#STANDARD DEVIATION
print("Sigma:", np.std(data_set))
