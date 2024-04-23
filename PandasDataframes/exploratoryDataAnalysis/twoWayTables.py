import PandasDataframes.gotoDataDir
import pandas as pd
import numpy as np

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])

cars_data_copy = cars_data.copy()

'''
Two-way frequency table
-----------------------
It finds out the frequency cross table between two variables.
Here the values of Automatic and the values of FuelType variables(columns).
'''
result = pd.crosstab(index=cars_data_copy['Automatic'],
                     columns=cars_data_copy['FuelType'],
                     dropna=False)
print(result)
'''
FuelType   CNG  Diesel  Petrol  NaN
Automatic                          
0           15     144    1104   93
1            0       0      73    7

This relationship shows that Automatic cars (value=1) has only Petrol cars.
'''

'''
Two-way Joint Probability
-------------------------
Joint probability: focuses on the probability of multiple events occurring simultaneously.

By setting normalize=True in crosstab(), you can convert the values to percentage (probability).
'''
result = pd.crosstab(index=cars_data_copy['Automatic'],
                     columns=cars_data_copy['FuelType'],
                     normalize=True,
                     dropna=True)
print(result)
'''
FuelType        CNG    Diesel    Petrol
Automatic                              
0          0.011228  0.107784  0.826347
1          0.000000  0.000000  0.054641

This means the probability of Automatic cars having petrol engine is really high.
'''

'''
Two-Way table - marginal probability
------------------------------------
You will get row sums and column sums by setting margins=True
'''
result = pd.crosstab(index=cars_data_copy['Automatic'],
                     columns=cars_data_copy['FuelType'],
                     normalize=True,
                     margins=True,
                     dropna=True)
print(result)
'''
FuelType        CNG    Diesel    Petrol       All
Automatic                                        
0          0.011228  0.107784  0.826347  0.945359
1          0.000000  0.000000  0.054641  0.054641
All        0.011228  0.107784  0.880988  1.000000
'''

'''
Two-way table- conditional probability
--------------------------------------
https://www.youtube.com/watch?v=-bijcTgRVBQ - watch this video to understand what is conditional probability
Conditional probability: focuses on the probability of one event given that another event has already occurred.

In above data with margins, 
    Conditional probability P(Petrol | Automatic) = probability of petrol with given probability of Automatic
                                                  = 0.826347/0.945359
                                                  = 0.874109  
'''
result = pd.crosstab(index=cars_data_copy['Automatic'],
                     columns=cars_data_copy['FuelType'],
                     normalize='index',
                     margins=True,
                     dropna=True)
print(result)
'''
FuelType        CNG    Diesel    Petrol
Automatic                              
0          0.011876  0.114014  0.874109
1          0.000000  0.000000  1.000000
All        0.011228  0.107784  0.880988


'''