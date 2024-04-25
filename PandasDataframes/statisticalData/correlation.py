import PandasDataframes.gotoDataDir
import pandas as pd
import numpy as np

# to see entire dataframe in the output
pd.set_option('display.max_columns', None)

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])

cars_data_copy = cars_data.copy()

'''
It shows the strength of relation between two variables. 
Variables do not have to be numeric, but for our example, we will use numeric variables.

Correlation is bound between -1 and +1. 0 means no correlation at all. 

Closer to +1 represents there is a strong correlation between two variables positively. 
Theoretically, above 0.7, you can say there is a fair correlation between two numerical variables.

Closer to -1 represents weaker correlation between two numerical variables. E.g. relation between price and age of the car. As car gets older, price goes down.
'''

# Pearson method can calculate correlation only between two numeric variables.
#cars_data_copy = cars_data_copy.select_dtypes(exclude=['object'])
result = cars_data_copy.corr(method="pearson", numeric_only=True)
print(result)
'''
              Price       Age        KM  ...  Automatic        CC    Weight
Price      1.000000 -0.878407 -0.574720  ...   0.033081  0.165067  0.581198
Age       -0.878407  1.000000  0.512735  ...   0.032573 -0.120706 -0.464299
KM        -0.574720  0.512735  1.000000  ...  -0.081248  0.299993 -0.026271
MetColor   0.112041 -0.099659 -0.093825  ...  -0.013973  0.029189  0.057142
Automatic  0.033081  0.032573 -0.081248  ...   1.000000 -0.069321  0.057249
CC         0.165067 -0.120706  0.299993  ...  -0.069321  1.000000  0.651450
Weight     0.581198 -0.464299 -0.026271  ...   0.057249  0.651450  1.000000

[7 rows x 7 columns]
'''

# You have to install scipy package for kendall method to work
result = cars_data_copy.corr(method="kendall", numeric_only=True)
print(result)
'''
              Price       Age        KM  ...  Automatic        CC    Weight
Price      1.000000 -0.676975 -0.451828  ...   0.035114  0.094611  0.313279
Age       -0.676975  1.000000  0.391403  ...   0.039274 -0.081276 -0.275119
KM        -0.451828  0.391403  1.000000  ...  -0.062915  0.139621 -0.047783
MetColor   0.084657 -0.070473 -0.056332  ...  -0.013973  0.037644  0.055747
Automatic  0.035114  0.039274 -0.062915  ...   1.000000 -0.055211  0.075632
CC         0.094611 -0.081276  0.139621  ...  -0.055211  1.000000  0.568306
Weight     0.313279 -0.275119 -0.047783  ...   0.075632  0.568306  1.000000

[7 rows x 7 columns]

It shows strong (positive) correlation between CC and Weight. 
It means as CC is increasing in data, Weight is also increasing.
There is a negative correlation between Price and Age. As age of the car increases, Price reduces.
'''


result = cars_data_copy.corr(method="spearman", numeric_only=True)
print(result)
'''
             Price       Age        KM  ...  Automatic        CC    Weight
Price      1.000000 -0.846022 -0.620391  ...   0.042542  0.118579  0.415687
Age       -0.846022  1.000000  0.548892  ...   0.047612 -0.103895 -0.386541
KM        -0.620391  0.548892  1.000000  ...  -0.077015  0.181954 -0.064825
MetColor   0.102567 -0.085439 -0.068953  ...  -0.013973  0.040320  0.066393
Automatic  0.042542  0.047612 -0.077015  ...   1.000000 -0.059122  0.090042
CC         0.118579 -0.103895  0.181954  ...  -0.059122  1.000000  0.683772
Weight     0.415687 -0.386541 -0.064825  ...   0.090042  0.683772  1.000000

[7 rows x 7 columns]
'''