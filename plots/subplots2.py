import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create 1 row and 2 columns of subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Plot on the first subplot
ax[0].plot(x, y1)
ax[0].set_title("Sine")

# Plot on the second subplot
ax[1].plot(x, y2, color='orange')
ax[1].set_title("Cosine")

plt.tight_layout()
plt.show()
