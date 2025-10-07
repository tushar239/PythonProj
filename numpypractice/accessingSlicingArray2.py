import numpy as np

inputs = [[0.3691268],
          [0.37620682],
          [0.35528856],
          [0.451298],
          [0.4201888],
          [0.39294143],
          [0.39755417],
          [0.37770865],
          [0.46406351],
          [0.57552027],
          [0.62980047],
          [0.54076378]]
inputs = np.array(inputs)

# extracting 0 to 4th rows
subset_2D = input[0:5]
'''
[[0.3691268], [0.37620682], [0.35528856], [0.451298], [0.4201888] ]
'''

# extracting 0 to 4th rows and 0th column in those rows
subset = inputs[0:5, 0]
print(subset)
'''
[0.3691268  0.37620682 0.35528856 0.451298   0.4201888 ]
'''

subset_2D = subset.reshape(-1,1)
print(subset_2D)
'''
[[0.3691268 ]
 [0.37620682]
 [0.35528856]
 [0.451298  ]
 [0.4201888 ]]
'''

subset_3D = subset_2D.reshape(1, subset_2D.shape[0], subset_2D.shape[1])
print(subset_3D)
'''
[[[0.3691268 ]
  [0.37620682]
  [0.35528856]
  [0.451298  ]
  [0.4201888 ]]]
'''


