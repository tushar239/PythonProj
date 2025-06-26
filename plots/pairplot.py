# code is taken from chatgpt
'''
A pairplot:

Plots a scatter plot for every pair of numerical features.
Plots histograms (or KDE plots) on the diagonal.
Can color by a categorical label (using hue).

Optional Parameters:
Parameter	    Description
hue	            Categorical column to color the points
diag_kind	    'hist' or 'kde' (default)
markers	        Customize marker styles for different hues
corner=True	    Show only lower triangle (less clutter)

KDE vs Histogram:
Feature	        Histogram	            KDE
Shape	        Blocky (bins)	        Smooth curve
Bin size	    Affects appearance	    Bandwidth affects smoothness
Output	        Frequency bars	        Probability density curve
Use case	    Quick distribution view	More precise density estimation
'''

import seaborn as sns
import matplotlib.pyplot as plt

# Load the built-in Iris dataset
iris = sns.load_dataset("iris")

# Create pairplot
#sns.pairplot(iris, hue="species")  # hue colors by species (target class)
sns.pairplot(iris, hue="species", diag_kind="hist") # default is kde
#sns.pairplot(iris, hue="species", diag_kind="kde", corner=True)
'''
KDE stands for Kernel Density Estimation, which is a non-parametric way to estimate the probability distribution of a continuous variable.
In simple terms: KDE smooths out the data to draw a smooth curve (like a histogram, but better) showing where the data is concentrated.
'''

plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.show()
'''
What You’ll See:
A grid of scatter plots comparing:
sepal length, sepal width, petal length, petal width
Histograms (or KDEs) along the diagonal
Color-coded points based on flower species (setosa, versicolor, virginica)
'''
