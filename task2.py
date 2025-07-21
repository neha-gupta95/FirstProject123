import pandas as pd
import
from sklearn.linear_Model import LinearRegression

#create a data
np.random.seed(0)
hours = np.random.uniform(1,10,50)
# marks = 5*hours+np.random.uniform(-10,10,50)

df=pd.DataFrame({'hours':hours,'marks':marks})
#load the data
X=df[['hours']]
Y=df['marks']

#split the data into trainig and testing 
X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

# create and train the model
model = LinearRegression()
model.fit(X_train,Y_train)

#display the slope and intercept of the line
print(f"slope (coefficient):{model.coef_[0]}")
print(f"intercept:{model.intercept_}")