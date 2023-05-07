import pandas as pd
import numpy as np
from sklearn import linear_model

df=pd.read_csv("book.csv")

# print(df)
x = df[['temperature','rainfall','humidity']]
mango = df["mangoes"]
orange = df["oranges"]

# print(x)
# print(mango)
# print(orange)

regr =linear_model.LinearRegression()
regr.fit(x.values,mango)

print(regr.predict([[73,67,43]]))

regr.fit(x.values,orange)
print(regr.predict([[73,67,43]]))