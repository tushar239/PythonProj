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

'''
IMPORTANT:
For categorical variable, following things are useful
- value_count
- crosstab - can be used for a single categorical variable or for two categorical variables 
- countplot - can be used for a single categorical variable or for two categorical variables
- boxplot - can be used for a single categorical variable or for comparing a categorical variable with a numerical variable

For numerical variable, following things are useful
- histplot/displot
- boxplot
- regplot (same as scatterplot with or without regression line) - to compare to numerical variables 
'''

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

# powerPS vs price
plt.figure(figsize=(7,6))
sns.regplot(data=cars, x='powerPS', y='price', scatter=True, fit_reg=True,
            marker='*', scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.show()
# Look at the regression line. it is slanted up from left to right.
# It shows that Cars priced higher are for higher power

###### Now, let's compare the price with categorical variables ######
# Both value_count() and countplot show that highest jobs are private (>22k).
# you can use crosstab with columns='counts' also. crosstab helps you set margins, normalize etc
# you can use countplot to see ths same thing graphically

# crosstab and countplot/barplot can be used for one categorical variable also to see its frequency.
# They can be used to see the relation between two categorical variable also.


# Variable seller
seller_frequency = cars['seller'].value_counts()
print('seller value_count(frequency): \n', seller_frequency)
'''
seller
private       43154
commercial        1
Name: count, dtype: int64
'''
seller_crosstab = pd.crosstab(cars['seller'], columns='count', normalize=True)
print('seller crosstab: \n', seller_crosstab)

'''
seller crosstab: 
 col_0          count
seller              
commercial  0.000023
private     0.999977
'''
plt.figure(figsize=(7,6))
sns.countplot(x='seller', data=cars)
plt.show()
# Fewer cars have 'commercial' => Insignificant

# Variable offerType
offerType_frequency = cars['offerType'].value_counts()
print('offerType value_count(frequency): \n', offerType_frequency)
'''
offerType
offer    43155
Name: count, dtype: int64

There is just one offerType and that is 'offer'
'''

offerType_crosstab = pd.crosstab(cars['offerType'], columns='count', normalize=True)
print('offerType crosstab: \n', offerType_crosstab)
'''
col_0      count
offerType       
offer        1.0
There is just one offerType and that is 'offer'
'''

plt.figure(figsize=(7,6))
sns.countplot(x='offerType', data=cars)
plt.show()
'''
There is just one offerType and that is 'offer'
'''

# All cars have 'offer' => Insignificant

# Variable abtest
abtest_frequency = cars['abtest'].value_counts()
print('abtest value_count(frequency): \n', abtest_frequency)
'''
abtest
test       22337
control    20818
Name: count, dtype: int64

There are only two types of abtest - test and control. Both are equally distributed.
'''
abtest_crosstab = pd.crosstab(cars['abtest'], columns='count', normalize=True)
print('abtest crosstab: \n', abtest_crosstab)
'''
 col_0       count
abtest           
control  0.482401
test     0.517599

There are only two types of abtest - test and control. Both are equally distributed.
'''

plt.figure(figsize=(7,6))
sns.countplot(x='abtest', data=cars)
plt.show()
# There are only two types of abtest - test and control. Both are equally distributed.

plt.figure(figsize=(7,6))
sns.boxplot(x='abtest', y='price', data=cars)
plt.show()
# Box plots are almost same for abtest=test and control. We can't infer anything from box plot also.

# For every price value, there is almost 50-50 distribution
# Does not affect price => Insignificant

# Variable vehicleType

vehicleType_frequency = cars['vehicleType'].value_counts()
print('vehicleType value_count(frequency): \n', vehicleType_frequency)
'''
 vehicleType
limousine        11874
small car         9358
station wagon     8155
bus               3624
cabrio            2810
coupe             2276
suv               1847
others             328
Name: count, dtype: int64
'''

vehicleType_crosstab = pd.crosstab(cars['vehicleType'], columns='count', normalize=True)
print('vehicleType crosstab: \n', vehicleType_crosstab)
'''
vehicleType            
bus            0.089988
cabrio         0.069776
coupe          0.056516
limousine      0.294845
others         0.008145
small car      0.232370
station wagon  0.202498
suv            0.045863
'''

plt.figure(figsize=(7,6))
sns.countplot(x='vehicleType', data=cars)
plt.show()

plt.figure(figsize=(7,6))
sns.boxplot(x='vehicleType', y='price', data=cars)
plt.show()
# 8 vehicle types - limousine, small cars and station wagons max freq
# Price is different for different Vehicle Types.
# So, vehicleType does affect the price

# Variable gearbox

gearbox_frequency = cars['gearbox'].value_counts()
print('gearbox value_count(frequency): \n', gearbox_frequency)
'''
 gearbox
manual       32847
automatic     9512
Name: count, dtype: int64
'''

gearbox_crosstab = pd.crosstab(cars['gearbox'], columns='count', normalize=True)
print('gearbox crosstab: \n', gearbox_crosstab)
'''
 col_0         count
gearbox            
automatic  0.224557
manual     0.775443
'''

plt.figure(figsize=(7,6))
sns.countplot(x='gearbox', data=cars)
plt.show()

plt.figure(figsize=(7,6))
sns.boxplot(x='gearbox', y='price', data=cars)
plt.show()
# It is evident from the boxplot that manual cars price lower than automatic

# gearbox affects the price

# Variable model
model_frequency = cars['model'].value_counts()
print('model value_count(frequency): \n', model_frequency)
'''
  model
golf          3513
others        2911
3er           2510
polo          1513
corsa         1402
              ... 
b_max            1
serie_3          1
elefantino       1
charade          1
rangerover       1
Name: count, Length: 247, dtype: int64
'''

model_crosstab = pd.crosstab(cars['model'], columns='count', normalize=True)
print('model crosstab: \n', model_crosstab)
'''
  col_0       count
model            
100      0.001134
145      0.000096
147      0.001471
156      0.001568
159      0.000458
...           ...
yaris    0.003208
yeti     0.000651
ypsilon  0.000555
z_reihe  0.003015
zafira   0.008007

[247 rows x 1 columns]
'''

plt.figure(figsize=(7,6))
model_countplot = sns.countplot(x='model', data=cars)
model_countplot.set_xticklabels(model_countplot.get_xticklabels(), rotation=45,  horizontalalignment='right')
plt.show()

plt.figure(figsize=(7,6))
model_boxplot = sns.boxplot(x='model', y='price', data=cars)
model_boxplot.set_xticklabels(model_boxplot.get_xticklabels(), rotation=45,  horizontalalignment='right')
plt.show()
# Cars are distributed over many models
# Considered in modelling
