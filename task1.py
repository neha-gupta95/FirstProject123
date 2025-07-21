#visualize the relationship
#between study hours amd marks
#import necessary libraries
import matplotlib.pyplot as plt
import  seaborn as sns
import pandas as pd
import  numpy as np

#create data
data={
 'hours':[2,4,6,8,10,12,14,16,18,20],
 'marks':[20,30,40,50,60,70,80,90,95,100]   
}
df = pd.DataFrame(data)
#plot a scatter plot using matplotlib

plt.figure(figsize=(10,6))
plt.scatter(df['hours'],df['marks'],color='red',label='matplotlib scatter')
plt.title('study hours vs marks')
plt.xlabel('study hours')
plt.ylabel('marks')
plt.legend()
plt.show()

#plot a scatter plot using seaborn

plt.figure(figsize=(10,6))
sns.scatterplot(x='hours',y='marks',data=df,color= 'green', label='seaborn scatter')
plt.title('study hours vs marks')
plt.xlabel('study hours')
plt.ylabel('marks')
plt.legend()
plt.show()

#Bonus:use sns.regplot () to shaw the regression line

plt.figure(figsize=(10,6))
sns.regplot(x='hours',y='marks', data=df,color='pink')
plt.title('study hours vs marks with regression line')
plt.xlabel('study hours')
plt.ylabel('marks')
plt.show()