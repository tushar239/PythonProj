import pandas as pd

car_colors = pd.Series(['Blue', 'Red', 'Green'], dtype='category')
car_data = pd.Series(pd.Categorical(['Yellow', 'Green', 'Red', 'Blue', 'Purple'],
                    categories=car_colors, ordered=False))
find_entries = pd.isnull(car_data)
print(find_entries)
'''
0     True
1    False
2    False
3    False
4     True
'''
print(car_colors)
print()
print(car_data)
print()
print(find_entries[find_entries == True])
isNull = car_data.isnull()
print(type(isNull)) # <class 'pandas.core.series.Series'>
print(isNull)
'''
0     True
1    False
2    False
3    False
4     True
dtype: bool
'''