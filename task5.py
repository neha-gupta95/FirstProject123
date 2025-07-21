 

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data 
data = {'Hours': [1, 2, 3, 4, 5],
        'Marks': [20, 34, 56, 67, 76]}
df = pd.DataFrame(data)

# Model
X = df[['Hours']]  # Features
Y = df['Marks']    # Target

model = LinearRegression()
model.fit(X, Y)

# Prediction
predicted = model.predict([[7]])
print(f"Predicted marks for 7 study hours: {predicted[0]:.2f}")

# 🔍 Visualization
plt.scatter(df['Hours'], df['Marks'], color='blue', label='Actual Data')
plt.plot(df['Hours'], model.predict(X), color='red', label='Regression Line')
plt.scatter(7, predicted, color='green', label='Predicted Point (7 hrs)')

plt.title('Study Hours vs Marks')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Obtained')
plt.legend()
plt.grid(True)
plt.show()