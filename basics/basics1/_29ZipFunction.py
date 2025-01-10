# zip function yields tuples until an input is exhausted.
state = ['Gujarat', 'Maharashtra', 'Rajasthan']
capital = ['Gandhinagar', 'Mumbai', 'Jaipur', 'Vadodara']
zipped = list(zip(state, capital))
print(zipped) # [('Gujarat', 'Gandhinagar'), ('Maharashtra', 'Mumbai'), ('Rajasthan', 'Jaipur')]

# looping zipped values
output_dict = {}
for (key, value) in zip(state, capital):
    output_dict[key] = value