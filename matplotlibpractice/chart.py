import numpy as np
import matplotlib.pyplot as plt

gfg = np.random.choice(10, 15)
print(gfg) # [5 8 9 1 6 7 9 3 7 7 3 6 9 2 9]

count, bins, ignored = plt.hist(gfg, 25, density=True)
plt.show()