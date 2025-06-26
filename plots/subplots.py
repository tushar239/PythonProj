import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

print(x)
print(y1)
print(y2)

# Create a figure with 1 row, 2 columns
# figure is the overall window
plt.figure(figsize=(10, 4))

'''
plt.subplot(nrows, ncols, index)

nrows	Number of rows in the grid
ncols	Number of columns in the grid
index	Position of the current plot (1-based)
'''

# First subplot (left)
plt.subplot(1, 2, 1)
plt.plot(x, y1)
plt.title("Sine Wave")

# Second subplot (right)
plt.subplot(1, 2, 2)
plt.plot(x, y2, color='orange')
plt.title("Cosine Wave")

plt.tight_layout()  # Adjust spacing
plt.show()
