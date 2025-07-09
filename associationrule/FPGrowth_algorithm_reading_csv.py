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

#transactions = df.values.tolist()


#transactions = df[0].apply(lambda x: x.split(',')).tolist() if df.shape[1] == 1 else df.values.tolist()
#print(type(transactions)) # <class 'list'>
#print(transactions)

# Step 2: Convert rows to lists and drop NaNs
transactions = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
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
