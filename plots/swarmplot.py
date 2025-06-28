# swarm plot: Similar to a scatter plot, but avoids overlapping points — useful for small datasets.

import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")
sns.swarmplot(x="day", y="total_bill", data=df)
plt.title("Swarm Plot")
plt.show()
