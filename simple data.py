import pandas as pd
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

#sample data
data={'hours':[1,2,3,4,5],'marks':[20,34,56,67,76]}
df=pd.DataFrame(data)

#model

x=df[['hours']]   #features
y=df['marks']     #target

model=LinearRegression()
model.fit(x,y)

#prediction
predicted=model.predict([[7]])
print(f"predicted marks for7studyhours:{predicted[0]:2f}")

#visualization

plt.scatter(df['hours'],df['marks'],color='blue',label='ActualData')
plt.plot(df['hours'],model.predict(x),color='red',label='regressionline')
plt.scatter(7,predicted,
color='green',label='predicted point(7hrs)')
plt.title('study hoursvs marks')
plt.xlabel('hours studied')
plt.ylabel('marks obtained')
plt.legend()
plt.grid(True)
plt.show()