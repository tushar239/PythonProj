import numpy
import numpy as np

preds_val= [[0.07557997],
 [0.07594149],
 [0.07798918]]

preds_val_numpy_array = numpy.array(preds_val)
print(preds_val_numpy_array.shape) # (3, 1)


preds_val_3D = np.reshape(preds_val,(1, 1, len(preds_val))) # (1,1,3)
print(preds_val_3D)
'''
[
    [
        0.07557997,0.07594149,0.07798918
    ]
]
'''

something = np.reshape(preds_val,(preds_val_numpy_array.shape[0], preds_val_numpy_array.shape[1], 1))
print(something)

'''
[[[0.07557997]],
 [[0.07594149]],
 [[0.07798918]]]
'''