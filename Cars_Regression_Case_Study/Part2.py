# To work with dataframes
import pandas as pd

# To perform numerical operations
import numpy as np

# To visualize data
import seaborn as sns
import matplotlib.pyplot as plt  # pyplot means python plot

import gotoDataDir

# =====================================================
# Setting dimensions for plot
# =====================================================
sns.set(rc={'figure.figsize':(11.7, 8.27)})

# =====================================================
# Reading CSV file
# =====================================================
cars_data=pd.read_csv('cars_samples.csv')

# =====================================================
# Creating copy
# =====================================================
cars = cars_data.copy()

# =============================================================
# Working range of data
# =============================================================
cars = cars[
    (cars.yearOfRegistration >= 1950) &
    (cars.yearOfRegistration <= 2018) &
    (cars.price >= 100) &
    (cars.price <= 150000) &
    (cars.powerPS >= 10) &
    (cars.powerPS <= 500)
]
'''
This is also fine
cars = cars[
    (cars['yearOfRegistration'] >= 1950) &
    (cars['yearOfRegistration'] <= 2018) &
    (cars['price'] >= 100) &
    (cars['price'] <= 150000) &
    (cars['powerPS'] >= 10) &
    (cars['powerPS'] <= 500)
]
'''
print("total filtered records")
print(cars.shape[0]) # 43155. ~6700 records are dropped

records_with_month_of_registration_0 = cars[cars['monthOfRegistration'] == 0]
print('Total Records with monthOfRegistration=0 :\n', records_with_month_of_registration_0.shape[0]) # 2578
'''
There are 2578 records with monthOfRegistration=0, which doesn't make sense.
Instead of removing these records, we can combine yearOfRegistration and monthOfRegistration and create new column Age
'''

cars['monthsConvertedToYears'] = cars['monthOfRegistration']/12
info = cars.info()
print('monthsConvertedToYears column added: \n', info)
print(cars['monthsConvertedToYears'])
'''
monthsConvertedToYears column added: 
 None
0        0.250000
1        0.500000
2        0.916667
3        1.000000
4        0.916667
           ...   
49991    0.666667
49992    0.750000
49993    0.666667
49995    0.250000
50000    0.833333
'''
# considering that 2018 is a current year, calculating the Age of the cars
cars['Age'] = 2018 - (cars['yearOfRegistration'] + cars['monthsConvertedToYears'])
cars['Age'] = round(cars['Age'], 2)
print('Age column added: \n', cars['Age'])
'''
Age column added: 
0        14.75
1        12.50
2        14.08
3        11.00
4         9.08
         ...  
49991    13.33
49992    19.25
49993    18.33
49995    16.75
50000    11.17
Name: Age, Length: 43155, dtype: float64
'''

age_summary = cars['Age'].describe()
print('Summary of Age:\n', age_summary)
'''
Summary of Age:
 count    43155.000000
mean        13.864225
std          7.098455
min         -1.000000
25%          9.250000
50%         13.750000
75%         18.170000
max         66.750000
Name: Age, dtype: float64

mean and median are not very far off. So, data is good.
'''

# Dropping yearOfRegistration and monthOfRegistration
# axis = 0 means x-axis(rows), 1 means y-axis(columns)
cars = cars.drop(columns=['yearOfRegistration', 'monthOfRegistration'], axis=1)

# Visualizing parameters

# Age
# kde=True means show a density curve. kde is an algorithm.
plt.figure(figsize=(7,6))
sns.histplot(data=cars, x='Age', kde=True)
plt.show()

plt.figure(figsize=(7,6))
sns.boxplot(y=cars['Age'])
plt.show()

# price
# kde=True means show a density curve. kde is an algorithm.
plt.figure(figsize=(7,6))
sns.histplot(data=cars, x='price', kde=True)
plt.show()

plt.figure(figsize=(7,6))
sns.boxplot(y=cars['price'])
plt.show()

# powerPS
# kde=True means show a density curve. kde is an algorithm.
plt.figure(figsize=(7,6))
sns.histplot(data=cars, x='powerPS', kde=True)
plt.show()

plt.figure(figsize=(7,6))
sns.boxplot(y=cars['powerPS'])
plt.show()

# Visualizing parameters after narrowing working range
# Age vs price
plt.figure(figsize=(7,6))
sns.regplot(data=cars, x='Age', y='price', scatter=True, fit_reg=True,
            marker='*', scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.show()
# Look at the regression line. it is slanted down from left to right.
# It shows that Cars priced higher are newer
# and with increase in age, price decreases
# However, some cars are priced higher with increase in age
