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
cars = cars.drop(columns=['monthsConvertedToYears', 'yearOfRegistration', 'monthOfRegistration'], axis=1)

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
- regplot (same as scatterplot with or without regression line) - to compare two numerical variables 
'''

# Age
# kde=True means show a density curve. kde is an algorithm.
plt.figure(figsize=(7,6))
sns.histplot(data=cars, x='Age', kde=True)
#plt.show()


plt.figure(figsize=(7,6))
sns.boxplot(y=cars['Age'])
#plt.show()

# price
# kde=True means show a density curve. kde is an algorithm.
plt.figure(figsize=(7,6))
sns.histplot(data=cars, x='price', kde=True)
#plt.show()

plt.figure(figsize=(7,6))
sns.boxplot(y=cars['price'])
#plt.show()

# Visualizing parameters after narrowing working range
# Both Age and price are numerical variables, so we should use replot also.
# Age vs price
plt.figure(figsize=(7,6))
sns.regplot(data=cars, x='Age', y='price', scatter=True, fit_reg=True,
            marker='*', scatter_kws={"color": "blue"}, line_kws={"color": "red"})
#plt.show()
# Look at the regression line. it is slanted down from left to right.
# It shows that Cars priced higher are newer
# and with increase in age, price decreases
# However, some cars are priced higher with increase in age

# powerPS
# kde=True means show a density curve. kde is an algorithm.
plt.figure(figsize=(7,6))
sns.histplot(data=cars, x='powerPS', kde=True)
#plt.show()

plt.figure(figsize=(7,6))
sns.boxplot(y=cars['powerPS'])
#plt.show()

# powerPS vs price
# Both powerPS and price are numerical variables, so we should use replot also.
plt.figure(figsize=(7,6))
sns.regplot(data=cars, x='powerPS', y='price', scatter=True, fit_reg=True,
            marker='*', scatter_kws={"color": "blue"}, line_kws={"color": "red"})
#plt.show()
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
#plt.show()
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
#plt.show()
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
#plt.show()
# There are only two types of abtest - test and control. Both are equally distributed.

plt.figure(figsize=(7,6))
sns.boxplot(x='abtest', y='price', data=cars)
#plt.show()
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
#plt.show()

plt.figure(figsize=(7,6))
sns.boxplot(x='vehicleType', y='price', data=cars)
#plt.show()
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
#plt.show()

plt.figure(figsize=(7,6))
sns.boxplot(x='gearbox', y='price', data=cars)
#plt.show()
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
#plt.show()

plt.figure(figsize=(7,6))
model_boxplot = sns.boxplot(x='model', y='price', data=cars)
model_boxplot.set_xticklabels(model_boxplot.get_xticklabels(), rotation=45,  horizontalalignment='right')
#plt.show()
# Cars are distributed over many models
# Considered in modelling

# Variable Kilometer
kilometer_frequency = cars['kilometer'].value_counts().sort_index()
print('kilometer value_count(frequency): \n', kilometer_frequency)
'''
 kilometer
5000        480
10000       209
20000       654
30000       725
40000       807
50000       938
60000      1126
70000      1190
80000      1395
90000      1496
100000     1851
125000     4635
150000    27649
Name: count, dtype: int64
'''
kilometer_crosstab = pd.crosstab(cars['kilometer'], columns='count', normalize=True)
print('kilometer crosstab: \n', kilometer_crosstab)
'''
col_0         count
kilometer          
5000       0.011123
10000      0.004843
20000      0.015155
30000      0.016800
40000      0.018700
50000      0.021736
60000      0.026092
70000      0.027575
80000      0.032325
90000      0.034666
100000     0.042892
125000     0.107404
150000     0.640691
'''

plt.figure(figsize=(7,6))
kilometer_countplot = sns.countplot(x='kilometer', data=cars)
kilometer_countplot.set_xticklabels(kilometer_countplot.get_xticklabels(), rotation=45,  horizontalalignment='right')
#plt.show()

# Regplot is useful between numerical variables. If one is categorical variable, then dots will not be scattered properly.
# Use Boxplot in that case.
'''
plt.figure(figsize=(7,6))
kilometer_regplot = sns.regplot(data=cars, x='kilometer', y='price', scatter=True, fit_reg=True,
            marker='*', scatter_kws={"color": "blue"}, line_kws={"color": "red"})
kilometer_regplot.set_xticklabels(kilometer_regplot.get_xticklabels(), rotation=45,  horizontalalignment='right')
plt.show()
'''

plt.figure(figsize=(7,6))
kilometer_boxplot = sns.boxplot(x='kilometer', y='price', data=cars)
kilometer_boxplot.set_xticklabels(kilometer_boxplot.get_xticklabels(), rotation=45,  horizontalalignment='right')
#plt.show()
# As kilometers goes up, price goes down.
# If kilometer is 5000, its price is the lowest. It seems like an outlier.
# Considered in modelling

# Variable fuelType
fuelType_frequency = cars['fuelType'].value_counts().sort_index()
print('fuelType value_count(frequency): \n', fuelType_frequency)
'''
 fuelType
cng           71
diesel     13028
electro       10
hybrid        36
lpg          697
other          6
petrol     26702

Max cars have fuelType petrol and then diesel
'''

fuelType_crosstab = pd.crosstab(cars['fuelType'], columns='count', normalize=True)
print('fuelType crosstab: \n', fuelType_crosstab)
'''
 col_0        count
fuelType          
cng       0.001751
diesel    0.321282
electro   0.000247
hybrid    0.000888
lpg       0.017189
other     0.000148
petrol    0.658496

Max cars have fuelType petrol and then diesel
'''

plt.figure(figsize=(7,6))
fuelType_countplot = sns.countplot(x='fuelType', data=cars)
fuelType_countplot.set_xticklabels(fuelType_countplot.get_xticklabels(), rotation=45,  horizontalalignment='right')
#plt.show()
# Max cars have fuelType petrol and then diesel

plt.figure(figsize=(7,6))
fuelType_boxplot = sns.boxplot(x='fuelType', y='price', data=cars)
fuelType_boxplot.set_xticklabels(fuelType_boxplot.get_xticklabels(), rotation=45,  horizontalalignment='right')
#plt.show()
# It is very clear that different fuelTypes have different price ranges.
# hybrid cars are more expensive
# Considered in modelling

# Variable brand
brand_frequency = cars['brand'].value_counts()
print('brand value_count(frequency): \n', brand_frequency)
'''
 brand
volkswagen        9229
bmw               4921
opel              4532
mercedes_benz     4182
audi              4025
ford              2840
renault           1954
peugeot           1333
fiat              1001
seat               888
skoda              703
mazda              667
smart              625
nissan             602
citroen            600
toyota             550
volvo              432
mini               429
hyundai            409
mitsubishi         362
sonstige_autos     301
honda              300
kia                279
suzuki             264
porsche            261
alfa_romeo         249
chevrolet          214
chrysler           152
dacia              126
subaru             112
jeep                92
land_rover          81
jaguar              79
daihatsu            67
saab                65
lancia              58
rover               53
daewoo              53
trabant             43
lada                22
Name: count, dtype: int64

Max number of cars are of brand volkswagen and then bmw
'''
brand_crosstab = pd.crosstab(cars['brand'], columns='count', normalize=True)
print('brand crosstab: \n', brand_crosstab)
'''
 col_0              count
brand                   
alfa_romeo      0.005770
audi            0.093268
bmw             0.114031
chevrolet       0.004959
chrysler        0.003522
citroen         0.013903
dacia           0.002920
daewoo          0.001228
daihatsu        0.001553
fiat            0.023195
ford            0.065809
honda           0.006952
hyundai         0.009477
jaguar          0.001831
jeep            0.002132
kia             0.006465
lada            0.000510
lancia          0.001344
land_rover      0.001877
mazda           0.015456
mercedes_benz   0.096906
mini            0.009941
mitsubishi      0.008388
nissan          0.013950
opel            0.105017
peugeot         0.030889
porsche         0.006048
renault         0.045279
rover           0.001228
saab            0.001506
seat            0.020577
skoda           0.016290
smart           0.014483
sonstige_autos  0.006975
subaru          0.002595
suzuki          0.006117
toyota          0.012745
trabant         0.000996
volkswagen      0.213857
volvo           0.010010

Max number of cars are of brand volkswagen and then bmw
'''
plt.figure(figsize=(7,6))
brand_countplot = sns.countplot(x='brand', data=cars)
brand_countplot.set_xticklabels(brand_countplot.get_xticklabels(), rotation=45,  horizontalalignment='right')
#plt.show()
# Max number of cars are of brand volkswagen and then bmw

plt.figure(figsize=(7,6))
brand_boxplot = sns.boxplot(x='brand', y='price', data=cars)
brand_boxplot.set_xticklabels(brand_boxplot.get_xticklabels(), rotation=45,  horizontalalignment='right')
#plt.show()
# proche has highest price
# brands definitely has effect on the price
# Considering brand in modelling

# Variable notRepairedDamage
# yes- car was damaged but not rectified
# no- car was damaged but has been rectified
notRepairedDamage_frequency = cars['notRepairedDamage'].value_counts()
print('notRepairedDamage value_count(frequency): \n', notRepairedDamage_frequency)
'''
 notRepairedDamage
no     32835
yes     4018

Max damaged cars were repaired 
'''
notRepairedDamage_crosstab = pd.crosstab(cars['notRepairedDamage'], columns='count', normalize=True)
print('notRepairedDamage crosstab: \n', notRepairedDamage_crosstab)
'''
notRepairedDamage          
no                 0.890972
yes                0.109028

Max damaged cars were repaired 
'''

plt.figure(figsize=(7,6))
notRepairedDamage_countplot = sns.countplot(x='notRepairedDamage', data=cars)
notRepairedDamage_countplot.set_xticklabels(notRepairedDamage_countplot.get_xticklabels(), rotation=45,  horizontalalignment='right')
#plt.show()
# Max damaged cars were repaired

plt.figure(figsize=(7,6))
notRepairedDamage_boxplot = sns.boxplot(x='notRepairedDamage', y='price', data=cars)
notRepairedDamage_boxplot.set_xticklabels(notRepairedDamage_boxplot.get_xticklabels(), rotation=45,  horizontalalignment='right')
#plt.show()
# As expected, the cars that require the damages to be repaired fall under lower price ranges

# ============================================================
# Removing insignificant variables
# ============================================================
cols_to_drop=['seller', 'offerType', 'abtest', 'postalCode']
cars=cars.drop(columns=cols_to_drop, axis=1) # axis=1 means cols
cars_copy = cars.copy()

# Using corr(), just checking whether there is a heavy correlation between price and other numeric variables
# Pearson method can calculate correlation only between two numeric variables.
cars_select1=cars.select_dtypes(exclude=[object])
correlation = cars_select1.corr(method="pearson")
print(correlation)
'''
              price   powerPS  kilometer       Age
price      1.000000  0.576038  -0.440266 -0.340663
powerPS    0.576038  1.000000  -0.017510 -0.156198
kilometer -0.440266 -0.017510   1.000000  0.294685
Age       -0.340663 -0.156198   0.294685  1.000000

It clearly shows that age and kilometer have inverse correlation with price.
As Age or kilometer increases, price decreases.

There is no heavy correlation between price and other numeric variables.
There is a reasonable correlation between the price and powerPS
'''
# Just more filtering
# select price column
just_price_col = correlation['price']
# select all rows from row# 1
just_price_col=just_price_col[1:]
print(just_price_col)
# OR
#just_price_col_without_first_row = correlation.loc[:, 'price'][1:]
#print(just_price_col_without_first_row)
'''
             price   
powerPS      0.576038
kilometer   -0.440266
Age         -0.340663

It clearly shows that age and kilometer have inverse correlation with price.
As Age or kilometer increases, price decreases.

There is no heavy correlation between price and other numeric variables.
There is a reasonable correlation between the price and powerPS
'''


'''
# ========================================================
We are going to build a Liner Regression and Random Forest model
on two sets of data.
1. Data obtained by omitting rows with any missing value
2. Data obtained by imputing the missing values
# ========================================================
'''
# ========================================================
# OMITTING MISSING VALUES
# ========================================================
cars_omit=cars.dropna(axis=0)
print('Number of rows after the rows with empty cells are omitted: ', cars_omit.shape[0]) # 33227

# Converting categorical variables dummy variables
# All machine learning algorithms work only on numeric data
cars_omit=pd.get_dummies(data=cars_omit, drop_first=True)

# ======================================================
# IMPORTING NECESSARY LIBRARIES
# ======================================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Separating input and output features
x1 = cars_omit.drop(['price'], axis='columns', inplace=False) # input variables(features)
y1 = cars_omit['price'] # output variable (feature)

prices = pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
plt.figure(figsize=(7,6))
prices_histogram = prices.hist()
# plt.show()
# compared to the histogram of y1, the histogram of log(y1) is giving a nice bell curve

# Transforming price as a logarithmic value
y1 = np.log(y1)

# Splitting data into train and test data

# https://www.geeksforgeeks.org/how-to-split-the-dataset-with-scikit-learns-train_test_split-function/
# random_state = 0. random_state is a seed used by random number generator.
# If you set random seed, same set of records will be chosen every time, you run this code,
# otherwise different set of records will be chosen.
# test_size=0.3 means 30% data will go in test set and 70% data will go to train set.
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state = 3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) # (23258, 44050) (9969, 44050) (23258,) (9969,)

# ============================================================
# BASELINE MODEL FOR OMITTED DATA
# ============================================================

"""
Let' find out RMSE -Root Mean Square Error
What is Error?
difference between actual observation and predicted observation

MAE, MSE, RMSE 
Watch 'MAE MSE RMSE.mp4'
https://www.youtube.com/watch?v=XifOXgdl7AI

We are making a base model by using test data mean value
This is to set a benchmark and to compare with our regression model
"""
# finding the mean for test data value
base_pred = np.mean(y_test)
print(base_pred)

# Repeating the same value till length of test data
base_pred = np.repeat(base_pred, len(y_test)) #  [8.24607899 8.24607899 8.24607899 ... 8.24607899 8.24607899 8.24607899]
print('base_pred : \n', base_pred)

# finding the RMSE
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
print('base_root_mean_square_error (RMSE) : \n', base_root_mean_square_error) # 1.127432049631379

# Setting intercept as true
lgr=LinearRegression(fit_intercept=True)

# Model
model_lin1 = lgr.fit(X_train, y_train)

# Predicting model on test set
cars_predictions_lin1 = lgr.predict(X_test)

# Computing MSE and RMSE
lin_mse1 = mean_squared_error(y_test, cars_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print("RMSE of test data: \n", lin_rmse1)

residuals1=y_test-cars_predictions_lin1
plt.figure(figsize=(7,6))
sns.regplot(x=cars_predictions_lin1, y=residuals1, scatter=True, fit_reg=False, data=cars)
#plt.show()
residuals1.describe()