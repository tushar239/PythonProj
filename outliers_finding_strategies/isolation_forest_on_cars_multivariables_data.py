from sklearn.ensemble import IsolationForest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

############ Trying Isolation Forest strategy to find outliers in multiple features of cars_samples ###############

cars_data=pd.read_csv('cars_samples.csv')
cars = cars_data.copy()
info = cars.info()
print(info)


def boxplots(df):
    # First subplot (left)
    plt.subplot(1, 3, 1)
    sns.boxplot(y=df['price'])
    plt.title("Price")

    # Second subplot (center)
    plt.subplot(1, 3, 2)
    sns.boxplot(y=df['powerPS'])
    plt.title("PowerPS")

    # Third subplot (right)
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df['yearOfRegistration'])
    plt.title("yearOfRegistration")

    plt.tight_layout()  # Adjust spacing
    plt.show()

    '''
    # all variables in one box plot
    data = [df['price'], df['powerPS'], df['yearOfRegistration']]
    plt.boxplot(data, labels=["price", "powerPS", "yearOfRegistration"])
    plt.title("Box Plot Example")
    plt.ylabel("Values")
    plt.grid(True)
    plt.show()
    '''


# taking 3 variables having outliers from cars samples
cars_limited_variables = pd.DataFrame()
cars_limited_variables['price'] = cars['price']
cars_limited_variables['powerPS'] = cars['powerPS']
cars_limited_variables['yearOfRegistration'] = cars['yearOfRegistration']

# box plots before dropping outliers
boxplots(df=cars_limited_variables)

iso = IsolationForest(contamination=0.15) # contamination=0.15 means we expect ~15% of points to be outliers.
cars_limited_variables['outlier'] = iso.fit_predict(cars_limited_variables)

# -1 = outlier, 1 = inlier
print(cars_limited_variables[cars_limited_variables['outlier'] == -1])
'''
        price  powerPS  yearOfRegistration  outlier
4       18750      185                2008       -1
19        698        0                2017       -1
38       9300       63                1979       -1
69     205000      116                1952       -1
70      22800      241                2011       -1
...       ...      ...                 ...      ...
49961   36399      230                2015       -1
49980       0       90                2017       -1
49984   17000      204                2013       -1
49994  175000      286                1998       -1
49997   19999        0                1990       -1
'''

###### Dropping outliers dataframe ##########
# Condition: Drop rows where 'Score' is less than 80
condition = cars_limited_variables['outlier'] == -1

# Get the index of rows to drop
indices_to_drop = cars_limited_variables[condition].index
print(type(indices_to_drop)) # <class 'pandas.core.indexes.base.Index'>
print(len(indices_to_drop)) # total 7500 rows needs to be removed
print(len(cars_limited_variables)) # total records 50001

# Drop the rows (in-place)
cars_limited_variables.drop(indices_to_drop, inplace=True)

# box plots after dropping outliers
boxplots(df=cars_limited_variables)