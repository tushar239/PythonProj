# lambda arguments: expression
'''
Use Case	                    Method
--------                        ------
Apply to one column	            df['col'].apply(lambda x: ...)
Apply across rows	            df.apply(lambda row: ..., axis=1)
Conditional value assignment	Inside the lambda
Group-wise transformation	    groupby().transform(lambda x: ...)
'''
import pandas as pd

# 1. Apply a Function to Each Value in a Column
df = pd.DataFrame({'price': [100, 200, 300]})
df['price_with_tax'] = df['price'].apply(lambda pr : pr * 1.10)
print(df)
'''
   price  price_with_tax
0    100           110.0
1    200           220.0
2    300           330.0
'''

# 2. Apply a Function Row-Wise
df = pd.DataFrame({
    'math': [80, 90, 70],
    'science': [85, 95, 60]
})

# Calculate average marks for each student
df['average'] = df.apply(lambda row: (row['math'] + row['science']) / 2, axis=1) # axis=1 means column
print(df)
'''
    math  science  average
0    80       85     82.5
1    90       95     92.5
2    70       60     65.0
'''

# 3. Create a New Column Based on a Condition
df = pd.DataFrame({'marks': [45, 67, 89, 38]})

# Pass or Fail
df['result'] = df['marks'].apply(lambda mark: 'Pass' if mark >= 50 else 'Fail') # comprehension pattern
print(df)
'''
   marks result
0     45   Fail
1     67   Pass
2     89   Pass
3     38   Fail
'''

# 4. Apply lambda to Multiple Columns Conditionally
df = pd.DataFrame({
    'product': ['A', 'B', 'C'],
    'price': [100, 250, 400]
})

# Label products as "Cheap", "Medium", "Expensive"
df['label'] = df['price'].apply(
    lambda x: 'Cheap' if x < 150 else ('Medium' if x <= 300 else 'Expensive') # comprehension pattern using if-else if-else
)
print(df)
'''
  product  price      label
0       A    100      Cheap
1       B    250     Medium
2       C    400  Expensive
'''

# 5. Use with groupby().apply()
df = pd.DataFrame({
    'department': ['IT', 'HR', 'IT', 'HR'],
    'salary': [60000, 50000, 65000, 55000]
})

# Calculate % difference from average salary in each department
df['salary_diff'] = df.groupby('department')['salary'].transform(
    lambda x: x - x.mean()
)
print(df)
'''
  department  salary  salary_diff
0         IT   60000      -2500.0
1         HR   50000      -2500.0
2         IT   65000       2500.0
3         HR   55000       2500.0
'''




