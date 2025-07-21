import matplotlib.pyplot as plt
import numpy as np

#define the data

hours_studied = np.array([1,2,3,4,5])
marks_obtained = np.array([20,30,60,70,80])

#create the scatter plot
plt.scatter(hours_studied,marks_obtained,color='blue',label='Actual Data')

#calculate the regression line
z=np.polyfit(hours_studied,marks_obtained,1)
p=np.poly1d(z)

#plot the regression line
plt.scatter(6,p(6),color='green',label='predicted point(6 hrs)')

#set the title and labels
plt.title('study hours vs marks')
plt.xlabel('hours studied')
plt.ylabel('marks obtained')

#add alegend
plt.legend()

#show the plot
plt.show()