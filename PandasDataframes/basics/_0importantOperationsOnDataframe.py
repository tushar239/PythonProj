# https://www.geeksforgeeks.org/how-to-replace-values-in-column-based-on-condition-in-pandas/

import pandas as pd
import numpy as np


# Data
student = {
    'Name': ['John', 'Jay', 'sachin', 'Geetha', 'Amutha', 'ganesh', None],
    'gender': ['male', 'male', 'male', 'female', 'female', 'male', None],
    'math score': [50, 100, 70, 80, 75, 40, 60],
    'test preparation': ['none', 'completed', 'none', 'completed',
                         'completed', 'none', 'none'],
}

# Creating a DataFrame object
df = pd.DataFrame(student)
print(df)

# checking entire df (all columns) for null. It returns data frame of True and False
nullDf = df.isnull()
print(nullDf)

'''
  Name  gender  math score  test preparation
0  False   False       False             False
1  False   False       False             False
2  False   False       False             False
3  False   False       False             False
4  False   False       False             False
5  False   False       False             False
6   True    True       False             False
'''

# checking a particular column for null. It returns a series of True and False
nullTestPreparationSeries = df['test preparation'].isnull()
print(nullTestPreparationSeries)
'''
0    False
1    False
2    False
3    False
4    False
5    False
6    False
'''
nullTestPrparationNotOd = df[nullTestPreparationSeries == False]
print(nullTestPrparationNotOd)
'''
     Name  gender  math score test preparation
0    John    male          50             none
1     Jay    male         100        completed
2  sachin    male          70             none
3  Geetha  female          80        completed
4  Amutha  female          75        completed
5  ganesh    male          40             none
6    None    None          60             none
'''
############ dataframe related operations ##################
# summaryOfDf       = df.info()
# df1               = df[[col1, col2]]
# summaryOfCol      = df[col].info()/astype(dtype='int64')/mean()/median()/mode()
# df2               = df[condition]  ---- just like df.loc[(condition)]
# df3               = df[filter1 & ~filter2] ---- just like df.loc[(filter1 & ~filter2)]
# df[col][row num] = value --- assigning value to a particular cell

dfSummary = df.info()
'''
 #   Column            Non-Null Count  Dtype 
---  ------            --------------  ----- 
 0   Name              6 non-null      object
 1   gender            6 non-null      object
 2   math score        7 non-null      int64 
 3   test preparation  7 non-null      object
dtypes: int64(1), object(3)
memory usage: 352.0+ bytes
'''
nameGenderDf = df[['Name', 'gender']]
print(nameGenderDf)

'''
  Name  gender
0    John    male
1     Jay    male
2  sachin    male
3  Geetha  female
4  Amutha  female
5  ganesh    male
6    None    None
<class 'pandas.core.series.Series'>
RangeIndex: 7 entries, 0 to 6
Series name: Name
'''

nameSummary = df['Name'].info()
'''
Non-Null Count  Dtype 
--------------  ----- 
6 non-null      object
'''

# just like df.loc('math score' > 50)
mathGreaterDataframe = df[df['math score'] > 50]
print(mathGreaterDataframe)
'''
     Name  gender  math score test preparation
1     Jay    male         100        completed
2  sachin    male          70             none
3  Geetha  female          80        completed
4  Amutha  female          75        completed
5    None    None          60             none
'''
for i in range(0, len(df['Name']), 1):
    aCellFromDf = df['Name'][i]
    print(aCellFromDf)
'''
John
Jay
sachin
Geetha
Amutha
ganesh
None
'''

filter1 = df['Name'].isnull()
filter2 = df['gender'].isnull()
dfWithFilters = df[filter1 & filter2]
print(dfWithFilters)
'''
   Name gender  math score test preparation
6  None   None          60             none
'''

############## df.loc[] #########################
# it can be used to filter out the records OR to assign the values to particular cells.
mathGreaterDataframe = df.loc[(df['math score'] > 50)]
print(mathGreaterDataframe)
'''
     Name  gender  math score test preparation
1     Jay    male         100        completed
2  sachin    male          70             none
3  Geetha  female          80        completed
4  Amutha  female          75        completed
5    None    None          60             none
'''

df.loc[(df['math score'] == 70), ['math score']] = 100
print(df)
'''
    Name  gender  math score test preparation
0    John    male          50             none
1     Jay    male         100        completed
2  sachin    male         100             none --- changes math score=100
3  Geetha  female          80        completed
4  Amutha  female          75        completed
5  ganesh    male          40             none
6    None    None          60             none
'''

################# np.where() ######################
# It can change the column values
# or
# returns a series with changed values(it doesn't change df in this case)
df['Name'] = np.where(df['Name'] == 'John',
                      'Josh',
                      df['Name'])
print(df['Name'])
'''
0      Josh
1       Jay
2    sachin
3    Geetha
4    Amutha
5    ganesh
6      None
Name: Name, dtype: object
'''

dfWithJaymin = np.where(df['Name'] == 'Jay',
                      'Jaymin',
                      df['Name'])
print(dfWithJaymin) # ['Josh' 'Jaymin' 'sachin' 'Geetha' 'Amutha' 'ganesh' None]

alteredMathScores = np.where(df['math score'] > 70,
         df['math score'],
         np.nan) # nan is used for missing data
print(alteredMathScores) # [ nan 100. 100.  80.  75.  nan  nan]

print(df)
'''
  Name  gender  math score test preparation
0    Josh    male          50             none
1     Jay    male         100        completed
2  sachin    male         100             none
3  Geetha  female          80        completed
4  Amutha  female          75        completed
5  ganesh    male          40             none
6    None    None          60             none

Process finished with exit code 0
'''

#########  changing index column ##########
print("------- changing index column ---------")
'''
subsetDf = df.loc[2:4, ['Name', 'math score']]
print(subsetDf)
'''

df = df.set_index("Name")
print(df)
for i in range(0, len(df['math score']), 1):
    aCellFromDf = df['math score'][i]
    print(aCellFromDf)
    #df['math score'][i] = df['math score'][i] * 10 # not allowed. use df.loc[]

subsetDf = df.loc[2:4, ['Name', 'math score']]
print(subsetDf)
'''
     Name  math score
2  sachin         100
3  Geetha          80
4  Amutha          75
'''

subsetDf = df.iloc[2:4, 1:3]
print(subsetDf)
'''
   gender  math score
2    male         100
3  female          80
'''

nameSeries = df['Name']
print(type(nameSeries)) # <class 'pandas.core.series.Series'>