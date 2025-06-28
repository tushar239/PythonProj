# facetgrid/cat plot : Multiple plots split by categories (great for grouped visualization).

import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.catplot(x="day", y="total_bill", col="sex", data=df, kind="box")
plt.suptitle("Box Plots by Gender", y=1.05)
plt.show()

sns.catplot(x="day", y="total_bill", col="sex", data=df, kind="bar")
plt.suptitle("Box Plots by Gender", y=1.05)
plt.show()

sns.catplot(x="day", y="total_bill", col="sex", data=df, kind="swarm")
plt.suptitle("Box Plots by Gender", y=1.05)
plt.show()