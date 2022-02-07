import _54MySqlInitialConfig as initial

connection = initial.connection
cursor = connection.cursor()

sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
val = ("John", "Highway 21")
cursor.execute(sql, val)

connection.commit()

print(cursor.rowcount, "record inserted.")
# This is how you access rowid of inserted record
print("1 record inserted, ID:", cursor.lastrowid)


cursor.close()
connection.close()
