import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Step 1: Read CSV as list of lists
# Each row is a transaction: ['milk', 'bread', 'butter']
df = pd.read_csv('transactions.csv', header=None)

print(df)

transactions = df.values.tolist()
print(type(transactions)) # <class 'list'>
print(transactions)

#transactions = df[0].apply(lambda x: x.split(',')).tolist() if df.shape[1] == 1 else df.values.tolist()
#print(type(transactions)) # <class 'list'>
#print(transactions)

'''
[['milk', 'bread', 'butter'], 
['bread', 'butter', nan], 
['milk', 'bread', nan], 
['milk', 'bread', 'butter'], 
['bread', nan, nan]]
'''

# Step 2: Convert to one-hot encoded DataFrame
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
onehot_df = pd.DataFrame(te_array, columns=te.columns_)

# Step 3: Run FP-Growth algorithm
frequent_itemsets = fpgrowth(onehot_df, min_support=0.4, use_colnames=True)

# Step 4: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Step 5: Display results
print("📦 Frequent Itemsets:")
print(frequent_itemsets)

print("\n🔗 Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
