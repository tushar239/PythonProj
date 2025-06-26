# code is taken from chatgpt

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Sample data
x = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]).T # same as transpose()
print(x)
'''
[[ 1  2  2  8  8 25]
 [ 2  2  3  7  8 80]]
'''
# Apply fuzzy c-means clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    x, c=2, m=2.0, error=0.005, maxiter=1000, init=None)

# Membership values for each data point
print("Membership matrix:\n", u)
'''
Membership matrix:
 [[9.97615405e-01 9.98409518e-01 9.98956255e-01 9.96223566e-01
  9.94998583e-01 2.16990998e-09]
 [2.38459541e-03 1.59048153e-03 1.04374541e-03 3.77643394e-03
  5.00141688e-03 9.99999998e-01]]
'''

# Predict cluster of new data
newdata = np.array([[3], [3]])
u_new, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
    test_data=newdata, cntr_trained=cntr, m=2.0, error=0.005, maxiter=1000)

print("New point's cluster memberships:", u_new)
# u[i][j] → Degree of membership of point j in cluster i
'''
New point's cluster memberships: [[5.24309801e-04] [9.99475690e-01]]
'''