# DBSCAN works the best with clustered data.
# For clustered data, you can find a good eps using elbow method.
# for cars['powerPS'], it doesn't work as it might not be a clustered data.

from sklearn.ensemble import IsolationForest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

############ Trying DBSCAN strategy to find outliers in powerPS feature of cars_samples ###############

cars_data=pd.read_csv('cars_samples.csv')
cars = cars_data.copy()
info = cars.info()
print(info)

# creating a new dataframe that contains only powerPS variable
just_powerPS_df = cars['powerPS'].to_frame()
print(just_powerPS_df)
print(len(just_powerPS_df)) # 50001


# elbow method strategy from chatgpt
def find_eps(min_samples):
    from sklearn.datasets import make_blobs
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import numpy as np

    # Create sample data
    #X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)

    # Step 1: Choose min_samples and compute k-distance
    k = min_samples  # min_samples
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(just_powerPS_df[['powerPS']])
    distances, _ = nn.kneighbors(just_powerPS_df[['powerPS']])
    print(distances)
    '''
    
    '''
    # Step 2: Get the distances to the k-th nearest neighbor
    k_distances = np.sort(distances[:, k - 1])  # kth column (0-indexed)
    print(k_distances)

    # Step 3: Plot the k-distance graph
    plt.figure(figsize=(8, 5))
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot
    plt.plot(k_distances) # plot y using x as index array 0..N-1, using default line style and color

    plt.ylabel(f"{k}th Nearest Neighbor Distance")
    plt.xlabel("Points sorted by distance")
    plt.title("K-distance Graph to Estimate eps for DBSCAN")
    plt.grid(True)
    plt.show()

####### box plot before dropping outlier rows ##################
sns.boxplot(y=cars['powerPS'])
plt.title("Box Plot of powerPS before dropping outlier rows")
plt.show()



# eps - Radius to search for neighboring points
# min_samples - Minimum number of points required to form a dense region (cluster)
# Usually, min_samples = 2 * number_of_features, If unsure, try between 4 to 10.
min_samples = 2
find_eps(min_samples)
#eps = 50000
eps = 5
#dbscan = DBSCAN(eps=5, min_samples=2)
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
just_powerPS_df['outlier'] = dbscan.fit_predict(just_powerPS_df[['powerPS']]) # When you use df[['column_name']], you are providing a list containing a single column name

print(just_powerPS_df)

# Outliers are labeled as -1
print(just_powerPS_df[just_powerPS_df['outlier'] == -1])


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