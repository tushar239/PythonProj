# strip plot: Like a swarm plot, but allows overlapping — better for large sample sizes.

import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.stripplot(x="day", y="total_bill", data=df, jitter=True)
plt.title("Strip Plot")
plt.show()
