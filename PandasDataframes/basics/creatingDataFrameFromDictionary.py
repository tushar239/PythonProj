# https://www.geeksforgeeks.org/how-to-replace-values-in-column-based-on-condition-in-pandas/

import pandas as pd

# Data
student = {
    'Name': ['John', 'Jay', 'sachin', 'Geetha', 'Amutha', 'ganesh'],
    'gender': ['male', 'male', 'male', 'female', 'female', 'male'],
    'math score': [50, 100, 70, 80, 75, 40],
    'test preparation': ['none', 'completed', 'none', 'completed',
                         'completed', 'none'],
}

# Creating a DataFrame object
df = pd.DataFrame(student)

print(df)