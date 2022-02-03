"""
Dictionary in python is like a HashMap in Java.
Duplicate keys are not allowed.

dictionaryname = {key1: value1, key2: value2}   --- It is in the form of Json

dictionary used to be unordered before 3.7 version, but now it is ordered.
"""

thisdict = {
    "brand": "Ford",
    "model": "Mustang",
    "year": 1964,
    "colors": ["red", "white", "blue"]
}
print(type(thisdict))  # <class 'dict'>
print(thisdict)  # {'brand': 'Ford', 'model': 'Mustang', 'year': 1964, 'colors': ['red', 'white', 'blue']}
print(len(thisdict))  # 4
print(thisdict["brand"])  # Ford
print(thisdict.get("brand"))  # Ford

keys = thisdict.keys()
print(type(keys))  # <class 'dict_keys'>
print(keys)  # dict_keys(['brand', 'model', 'year', 'colors'])

values = thisdict.values()
print(type(values))  # <class 'dict_values'>
print(values)  # dict_values(['Ford', 'Mustang', 1964, ['red', 'white', 'blue']])

# Change Items - two ways to change the values of the keys
thisdict["brand"] = "Toyota"
thisdict.update({"model": "scion", "year": 2020})
thisdict.get("colors").append("purple")
print(thisdict)  # {'brand': 'Toyota', 'model': 'scion', 'year': 2020, 'colors': ['red', 'white', 'blue', 'purple']}

# Add Items
thisdict = {"brand": "Ford", "model": "Mustang", "year": 1964, "color": "red"}
print(thisdict)
thisdict["city"] = "Sacramento"
thisdict.update({"state": "CA"})  # {'brand': 'Ford', 'model': 'Mustang', 'year': 1964, 'color': 'red'}
print(thisdict)  # {'brand': 'Ford', 'model': 'Mustang', 'year': 1964, 'color': 'red', 'city': 'Sacramento', 'state': 'CA'}

# Loop (iterate) dictionaries
for key in thisdict:
    value = thisdict[key]
    print(key, ":", value, end=", ")  # brand : Ford,model : Mustang,year : 1964,color : red,city : Sacramento,state : CA,

print()

for key, value in thisdict.items():
    print(key, ":", value, end=", ")  # brand : Ford, model : Mustang, year : 1964, color : red, city : Sacramento, state : CA,

print()

for key in thisdict.keys():
    print(key, end=",")  # brand,model,year,color,city,state,

print()

for value in thisdict.values():
    print(value, end=",")  # Ford,Mustang,1964,red,Sacramento,CA,

print()

# Copy dictionary
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
mydict = thisdict.copy()
print(mydict)  # {'brand': 'Ford', 'model': 'Mustang', 'year': 1964}
# or
mydict = dict(thisdict)
print(mydict)  # {'brand': 'Ford', 'model': 'Mustang', 'year': 1964}

# Nested dictionaries
myfamily = [
  {
    "name" : "Emil",
    "year" : 2004
  },
  {
    "name" : "Tobias",
    "year" : 2007
  },
  {
    "name" : "Linus",
    "year" : 2011
  }
]
print(type(myfamily))  # <class 'list'>
print(myfamily)  # [{'name': 'Emil', 'year': 2004}, {'name': 'Tobias', 'year': 2007}, {'name': 'Linus', 'year': 2011}]

# add 3 dictionaries into a new dictionary

child1 = {
  "name" : "Emil",
  "year" : 2004
}
child2 = {
  "name" : "Tobias",
  "year" : 2007
}
child3 = {
  "name" : "Linus",
  "year" : 2011
}

myfamily = {
  "child1" : child1,
  "child2" : child2,
  "child3" : child3
}

print(type(myfamily))  # <class 'dict'>
print(myfamily)  # {'child1': {'name': 'Emil', 'year': 2004}, 'child2': {'name': 'Tobias', 'year': 2007}, 'child3': {'name': 'Linus', 'year': 2011}}

# All other methods
# https://www.w3schools.com/python/python_dictionaries_methods.asp
