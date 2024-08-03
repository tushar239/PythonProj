import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Salaries2.csv')
#display the first 5 rows
#top_5 = df.head()
#print(top_5)
print(df)

'''
      Level   Salary
0        1    45000
1        2    50000
2        3    60000
3        4    80000
4        5   110000
..     ...      ...
105      6   150000
106      7   200000
107      8   300000
108      9   500000
109     10  1000000
'''

#plotting the Scatter plot to check relationship between Sal and Temp
'''
plt.figure(figsize=(7,6))
sns.lmplot(x ="Level", y ="Salary", data = df, order = 2, ci = None)
plt.show()
'''

plt.figure(figsize=(7,6))
sns.regplot(x='Level', y='Salary', scatter=True, fit_reg=True, data=df)
plt.show()

# Converting categorical variables dummy variables
# All machine learning algorithms work only on numeric data
#df_dummies=pd.get_dummies(data=df, drop_first=True)
#print(df_dummies)

# Separating input and output features
x1 = df.drop(['Salary'], axis='columns', inplace=False) # input variables(features)
y1 = df['Salary'] # output variable (feature)



# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state = 3)

regression_model = LinearRegression()

regression_model.fit(X_train, y_train)
print(regression_model.score(X_train, y_train)) # 0.6738072404638304
print(regression_model.score(X_test, y_test)) # 0.6297977850974639

y_pred = regression_model.predict(X_test) # 1.0
print(y_pred)
'''
[ 294916.51130745   40087.94682662  379859.36613439  294916.51130745
  -44854.90800032   40087.94682662 -129797.76282727  -44854.90800032
  125030.80165356   40087.94682662  464802.22096134  294916.51130745
   40087.94682662 -129797.76282727 -129797.76282727 -129797.76282727
   40087.94682662  294916.51130745  294916.51130745  634687.93061522
  464802.22096134  464802.22096134  209973.65648051  549745.07578828
  634687.93061522  209973.65648051  549745.07578828  464802.22096134
  -44854.90800032  209973.65648051  209973.65648051  549745.07578828
   40087.94682662]
'''
print(y_test)
print(X_test)

plt.scatter(X_test['Level'], y_test, color='b')
plt.plot(X_test['Level'], y_pred, color='k')
plt.show()
