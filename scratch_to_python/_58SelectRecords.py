import _54MySqlInitialConfig as initial

connection = initial.connection
cursor = connection.cursor()

cursor.execute("SELECT * FROM customers")

"""
result = cursor.fetchall()  # result is tuple

for x in result:
    print(x)
"""
# OR
"""
for (name, address) in cursor:
    print(name, address)
"""
# OR
for row in cursor:
    # print(type(row))  # Tuple
    print(row)

