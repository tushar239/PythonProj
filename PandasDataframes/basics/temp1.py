import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]
        }
df = pd.DataFrame(data)

# Using .loc[] to select rows by label
row_by_label = df.loc[1:2]  # Selects the row with index label 1 (Bob's data)
print(row_by_label)
'''
      Name  Age
1      Bob   30
2  Charlie   35
'''

# Using .iloc[] to select rows by position
row_by_position = df.iloc[1:2]  # Selects the second row (Bob's data)
print(row_by_position)
'''
  Name  Age
1  Bob   30
'''

df = df.set_index('Name')
print(df)

print("index values:")
indexValues = df.index.values
print(indexValues)

# Using .loc[] to select rows by label
row_by_label = df.loc[['Alice', 'Bob']]  # Selects the row with index label 1 (Bob's data)
print(row_by_label)