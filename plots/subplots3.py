import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# Flatten the 2D array to make indexing easier
axes = ax.flatten()

for i in range(4):
    axes[i].plot(x, np.sin(x + i))  # Slightly shift each plot
    axes[i].set_title(f"Plot {i+1}")

plt.tight_layout()
plt.show()
