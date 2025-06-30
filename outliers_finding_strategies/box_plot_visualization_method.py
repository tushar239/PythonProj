import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'age': [15, 16, 14, 17, 18, 16, 100]  # 100 is an outlier
})

sns.boxplot(y=df['age'])
plt.title("Box Plot of Age")
plt.show()
