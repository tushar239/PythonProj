# https://www.askpython.com/python/built-in-methods/for-loop-two-variables


'''
To loop over two or more sequences at the same time, the entries can be paired with the zip() function.
'''

names = ['Dhushi', 'Praj', 'Lee']
languages = ['Python', 'JavaScript', 'Java']
for name, language in zip(names, languages):
    print('My name is {0}. My favourite language is {1}.'.format(name, language))
'''
My name is Dhushi. My favourite language is Python.
My name is Praj. My favourite language is JavaScript.
My name is Lee. My favourite language is Java.
'''

'''
enumerate() the function is used to retrieve the position index and corresponding value while looping through the list. 
'''
languages = ['Python', 'C', 'Java', 'JavaScript']
for index, value in enumerate(languages):
    print(f'Index:{index}, Value:{value}')
'''
Index:0, Value:Python
Index:1, Value:C
Index:2, Value:Java
Index:3, Value:JavaScript
'''

'''
Looping dictionary items
'''
person = {"Dhushi": 6, "Lee": 32, "Marc": 30}

for key, value in person.items():
    print(f'Name: {key}, Age: {value}.')
'''
Name: Dhushi, Age: 6.
Name: Lee, Age: 32.
Name: Marc, Age: 30.
'''
