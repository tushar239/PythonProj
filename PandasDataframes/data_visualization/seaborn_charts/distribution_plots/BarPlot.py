# https://www.geeksforgeeks.org/seaborn-barplot-method-in-python/
# https://www.geeksforgeeks.org/pandas-groupby/

# importing the required library
import seaborn as sns
import matplotlib.pyplot as plt

# read a titanic.csv file
# from seaborn library
df = sns.load_dataset('titanic')
df.to_csv('titanic.csv')


gk = df.groupby('who')
print(gk.first())
print(gk.get_group('child'))

gkk = df.groupby(['who', 'class'])
print(gkk.first())
print(gkk.get_group(('child','First')))

# who v/s fare barplot
sns.barplot(x='who',
            y='fare',
            data=df)
plt.show()


# who v/s fare barplot
sns.barplot(x='who',
            y='fare',
            hue='class',
            data=df)
plt.show()

# who v/s fare barplot
sns.barplot(x='who',
            y='fare',
            hue='class',
            data=df,
            palette="Blues")
plt.show()