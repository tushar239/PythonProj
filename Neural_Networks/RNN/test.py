import numpy as np

preds_val= [[0.07557997],
 [0.07594149],
 [0.07798918]]

'''
[
    [
        0.07557997,0.07594149,0.07798918
    ]
]
'''

preds_val_3D = np.reshape(preds_val,(1, 1, len(preds_val)))
print(preds_val_3D)