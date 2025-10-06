import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # all scaled stock prices will be in between 0 and 1

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv') # creating a dataframe of Google's stock prices

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
print(real_stock_price)
'''
[[778.81], [788.36], [786.08], [795.26], [806.4 ], [807.86], [805.  ], [807.14], [807.48], [807.08], [805.81], [805.12], [806.91], [807.25], [822.3 ], [829.62], [837.81], [834.71], [814.66], [796.86]]
'''
# Getting the predicted stock price of 2017
dataset_train_open_series = dataset_train['Open'] # 1258 records
dataset_test_open_series = dataset_test['Open'] # 20 records
print(len(dataset_train_open_series))

# extract last 60 records
dataset_train_open_series_last_60 = dataset_train_open_series[len(dataset_train_open_series)-60:]
print(len(dataset_train_open_series_last_60)) # 60
print(dataset_train_open_series_last_60)

dataset_total = pd.concat((dataset_train_open_series_last_60, dataset_test_open_series), axis=0)
print(len(dataset_total)) # 80
print(dataset_total)

inputs = dataset_total.values
'''
[779.   779.66 777.71 786.66 783.76 781.22 781.65 779.8  787.85 798.24
 803.3  795.   804.9  816.68 806.34 801.   808.35 795.47 782.89 778.2
 767.25 750.66 774.5  783.4  779.94 791.17 756.54 755.6  746.97 755.2
 766.92 771.37 762.61 772.63 767.73 764.26 760.   771.53 770.07 757.44
 744.59 757.71 764.73 761.   772.48 780.   785.04 793.9  797.4  797.34
 800.4  790.22 796.76 795.84 792.36 790.9  790.68 793.7  783.33 782.75
 778.81 788.36 786.08 795.26 806.4  807.86 805.   807.14 807.48 807.08
 805.81 805.12 806.91 807.25 822.3  829.62 837.81 834.71 814.66 796.86]
'''
inputs = inputs.reshape(-1,1)
print(inputs)
'''
[[779.  ]
 [779.66]
 [777.71]
 [786.66]
 [783.76]
 [781.22]
 [781.65]
 [779.8 ]
 [787.85]
 [798.24]
 [803.3 ]
 [795.  ]
 [804.9 ]
 [816.68]
 [806.34]
 [801.  ]
 [808.35]
 [795.47]
 [782.89]
 [778.2 ]
 [767.25]
 [750.66]
 [774.5 ]
 [783.4 ]
 [779.94]
 [791.17]
 [756.54]
 [755.6 ]
 [746.97]
 [755.2 ]
 [766.92]
 [771.37]
 [762.61]
 [772.63]
 [767.73]
 [764.26]
 [760.  ]
 [771.53]
 [770.07]
 [757.44]
 [744.59]
 [757.71]
 [764.73]
 [761.  ]
 [772.48]
 [780.  ]
 [785.04]
 [793.9 ]
 [797.4 ]
 [797.34]
 [800.4 ]
 [790.22]
 [796.76]
 [795.84]
 [792.36]
 [790.9 ]
 [790.68]
 [793.7 ]
 [783.33]
 [782.75]
 [778.81]
 [788.36]
 [786.08]
 [795.26]
 [806.4 ]
 [807.86]
 [805.  ]
 [807.14]
 [807.48]
 [807.08]
 [805.81]
 [805.12]
 [806.91]
 [807.25]
 [822.3 ]
 [829.62]
 [837.81]
 [834.71]
'''

inputs = sc.fit_transform(inputs)

# creating the samples of 60 records each
# for the first sample(0-59 records), 60th record will the actual output
# for second sample (1-60 records), 61st record will be the actual output
# for third sample (2-60 records), 62nd record will be the actual output
# and so on
# so, 60th to 79th records will be the actual outputs of samples. This is same as real_stock_price.
X_test = []
for i in range(0, 20):
    X_test.append(inputs[i:60+i, 0])
X_test = np.array(X_test) # converting python list to numpy array
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test)
'''
[[[0.3691268 ]
  [0.37620682]
  [0.35528856]
  ...
  [0.52681828]
  [0.41557606]
  [0.40935422]]

 [[0.37620682]
  [0.35528856]
  [0.451298  ]
  ...
  [0.41557606]
  [0.40935422]
  [0.36708861]]

 [[0.35528856]
  [0.451298  ]
  [0.4201888 ]
  ...
  [0.40935422]
  [0.36708861]
  [0.46953443]]

 ...

 [[0.54580562]
  [0.41085604]
  [0.36054495]
  ...
  [0.83361939]
  [0.91214332]
  [1.        ]]

 [[0.41085604]
  [0.36054495]
  [0.24308088]
  ...
  [0.91214332]
  [1.        ]
  [0.96674533]]

 [[0.36054495]
  [0.24308088]
  [0.06511478]
  ...
  [1.        ]
  [0.96674533]
  [0.75166273]]]
'''
