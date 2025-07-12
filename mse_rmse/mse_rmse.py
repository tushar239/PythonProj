'''
MSE (Mean Squared Error)
It measures the average of the squares of the prediction errors.
Good when you want to penalize large errors more (because of squaring).

What is RMSE (Root Mean Squared Error)?
It's the square root of MSE, bringing the error back to original units.
Easier to interpret than MSE because it’s in the same unit as the data.

MSE and RMSE are used to compare two models.
Whichever model has low MSE and RMSE, go for that model.
'''

from sklearn.metrics import mean_squared_error
import numpy as np

# True vs Predicted values
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# MSE
mse = mean_squared_error(y_true, y_pred)

# RMSE
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)

'''
MSE: 0.375
RMSE: 0.6123724356957945
'''