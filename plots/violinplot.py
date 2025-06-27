# violin plot: Combines box plot and KDE to show distribution shape.

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = sns.load_dataset("iris")
sns.violinplot(data=df, x="species", y="sepal_length")
plt.title("Violin Plot")
plt.show()
