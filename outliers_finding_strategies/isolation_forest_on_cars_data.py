from sklearn.ensemble import IsolationForest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

############ Trying Isolation Forest strategy to find outliers in powerPS feature of cars_samples ###############

cars_data=pd.read_csv('cars_samples.csv')
cars = cars_data.copy()
info = cars.info()
print(info)

# creating a new dataframe that contains only powerPS variable
just_powerPS_df = cars['powerPS'].to_frame()
print(just_powerPS_df)
print(len(just_powerPS_df)) # 50001

# applying IsolationForest strategy to find outliers
# contamination=0.039 means we expect ~3.9% of points to be outliers.
# By plotting box plot with different contamination values, I could figure out that 0.039 can be the right value
iso = IsolationForest(contamination=0.039)
just_powerPS_df['outlier'] = iso.fit_predict(just_powerPS_df[['powerPS']])

# -1 = outlier, 1 = inlier
print(just_powerPS_df[just_powerPS_df['outlier'] == -1])

####### box plot before dropping outlier rows ##################
sns.boxplot(y=cars['powerPS'])
plt.title("Box Plot of powerPS before dropping outlier rows")
plt.show()

###### Dropping outliers from cars dataframe ##########
# Condition: Drop rows where 'Score' is less than 80
condition = just_powerPS_df['outlier'] == -1

# Get the index of rows to drop
indices_to_drop = just_powerPS_df[condition].index
print(type(indices_to_drop)) # <class 'pandas.core.indexes.base.Index'>
print(len(indices_to_drop)) # total 1948 rows needs to be removed

# Drop the rows (in-place)
cars.drop(indices_to_drop, inplace=True)

######## box plot after dropping outlier rows ##################
sns.boxplot(y=cars['powerPS'])
plt.title("Box Plot of powerPS after dropping outlier rows")
plt.show()