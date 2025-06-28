# Jointplot: Combines a scatter plot with histograms (or KDE) on the axes.

import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.jointplot(data=df, x="total_bill", y="tip", kind="scatter")
plt.title("Joint Plot")
plt.show()