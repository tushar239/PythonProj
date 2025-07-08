'''
https://www.youtube.com/watch?v=q4J1YupRFC0&list=PLm5IZukmvRNmj2eqq9Y0_7GRP_4dozIGZ&index=5
https://www.youtube.com/watch?v=LTLNiL6EWt8&list=PLm5IZukmvRNmj2eqq9Y0_7GRP_4dozIGZ&index=6
https://www.youtube.com/watch?v=bury6r8_rYs&list=PLm5IZukmvRNmj2eqq9Y0_7GRP_4dozIGZ&index=7
'''

# code is taken from chatgpt

'''
Support	    P(A ∩ B)	How often A and B appear together
Confidence	P(B) when A appears = Support(A ∩ B) / Support(A)
Lift	    Confidence / Support(B)	    If Lift > 1, A and B are positively correlated
'''
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Step 1: Sample transaction data (one-hot encoded)
df = pd.DataFrame({
    'milk':   [1, 0, 1, 1, 0],
    'bread':  [1, 1, 1, 1, 0],
    'butter': [0, 1, 1, 1, 1],
    'jam':    [0, 0, 1, 0, 1]
})

# Step 2: Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Step 3: Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

'''
antecedents consequents  support  confidence    lift
0         (bread)      (milk)      0.6        0.75  1.2500
1          (milk)     (bread)      0.6        1.00  1.2500
2         (bread)    (butter)      0.6        0.75  0.9375
3        (butter)     (bread)      0.6        0.75  0.9375
4           (jam)    (butter)      0.4        1.00  1.2500
5  (milk, butter)     (bread)      0.4        1.00  1.2500
'''