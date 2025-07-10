import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Step 1: Read CSV as list of lists
# Each row is a transaction: ['milk', 'bread', 'butter']
df = pd.read_csv('transactions.csv', header=None)

print(df)
'''
   0       1       2
0   milk   bread  butter
1  bread  butter     NaN
2   milk   bread     NaN
3   milk   bread  butter
4  bread     NaN     NaN
'''

print(df[0])
#transactions = df.values.tolist()
print(df.shape[1]) # 3
print(df.values)
'''
[['milk' 'bread' 'butter']
 ['bread' 'butter' nan]
 ['milk' 'bread' nan]
 ['milk' 'bread' 'butter']
 ['bread' nan nan]]
'''
print(df.values.tolist())
'''
[['milk', 'bread', 'butter'], 
['bread', 'butter', nan], 
['milk', 'bread', nan], 
['milk', 'bread', 'butter'], 
['bread', nan, nan]]

'''
'''
if df.shape[1] == 1:
    transactions = df[0].apply(lambda x: x.split(',')).tolist()
else:
    transactions = df.values.tolist()
'''
# or
#transactions = df[0].apply(lambda x: x.split(',')).tolist() if df.shape[1] == 1 else df.values.tolist()
#print(type(transactions)) # <class 'list'>
#print(transactions)

# Step 2: Convert rows to lists and drop NaNs
transactions = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist() # axis=1 means column
print(type(transactions)) # <class 'list'>
print(transactions)

'''
[['milk', 'bread', 'butter'], 
['bread', 'butter'], 
['milk', 'bread'], 
['milk', 'bread', 'butter'], 
['bread']]
'''

# Step 2: Convert to one-hot encoded DataFrame
'''
The TransactionEncoder is designed to convert transactional data 
(typically a list of lists, where each inner list represents a transaction 
and contains items) into a one-hot encoded format. 
This format is crucial for applying frequent pattern mining algorithms like Apriori or FP-Growth.

Transaction Data Format: 
The TransactionEncoder expects your data to be a Python list of lists, where each inner list represents a transaction (a collection of items purchased together).
For example: [['Apple', 'Beer', 'Rice', 'Chicken'], ['Apple', 'Beer', 'Rice'], ['Apple', 'Beer']].

Learning Unique Labels: 
When you call te.fit(dataset), the TransactionEncoder iterates through all the transactions in your dataset and identifies all the distinct items that appear across all transactions.
For example, in the dataset above, it would identify 'Apple', 'Beer', 'Rice', and 'Chicken' as the unique items.

Internal Storage: 
This learned information (the list of unique items) is stored internally within the TransactionEncoder object (accessed via the columns_ attribute). 
This is crucial for the subsequent transformation step.

Internal Storage: 
This learned information (the list of unique items) is stored internally within the TransactionEncoder object (accessed via the columns_ attribute). 
This is crucial for the subsequent transformation step.

In summary:
The fit method essentially "teaches" the TransactionEncoder about the vocabulary of items within your transaction data, preparing it for 
the transformation into a one-hot encoded format suitable for market basket analysis and other machine learning tasks.

One-hot encoding:
It is a method used in machine learning and data science to convert categorical (non-numerical) data into a numerical format that algorithms can understand.
-   For each unique category in a categorical variable, a new binary (0 or 1) column (also known as a dummy variable) is created.
-   In each row, only one of these new columns will have a value of 1, indicating the presence of that particular category, while all other columns for that row will be 0. 

Imagine a dataset with a "Color" column containing the values "Red," "Blue," and "Green." 
One-hot encoding would transform this into three new binary columns: 
    "Color_Red," "Color_Blue," and "Color_Green." 
If an original row has "Red" as the color, the "Color_Red" column would be 1, and "Color_Blue" and "Color_Green" would be 0.
If an original row has "Blue" as the color, the "Color_Blue" column would be 1, and "Color_Red" and "Color_Green" would be 0.

'''
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
onehot_df = pd.DataFrame(te_array, columns=te.columns_)
print(onehot_df)

'''
 bread  butter   milk
0   True    True   True
1   True    True  False
2   True   False   True
3   True    True   True
4   True   False  False
'''

# Step 3: Run FP-Growth algorithm
frequent_itemsets = fpgrowth(onehot_df, min_support=0.4, use_colnames=True)

# Step 4: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Step 5: Display results
print("📦 Frequent Itemsets:")
print(frequent_itemsets)

'''
Frequent Itemsets:
   support               itemsets
0      1.0                (bread)
1      0.6                 (milk)
2      0.6               (butter)
3      0.6          (milk, bread)
4      0.6        (butter, bread)
5      0.4         (milk, butter)
6      0.4  (butter, milk, bread)
'''

print("\n🔗 Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
'''
antecedents consequents  support  confidence  lift
0          (milk)     (bread)      0.6         1.0   1.0
1        (butter)     (bread)      0.6         1.0   1.0
2  (milk, butter)     (bread)      0.4         1.0   1.0
'''
