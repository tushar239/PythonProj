"""
Dictionary in python is like a HashMap in Java.
Duplicate keys are not allowed.

dictionaryname = {key1: value1, key2: value2}
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

# Loop dictionaries
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
