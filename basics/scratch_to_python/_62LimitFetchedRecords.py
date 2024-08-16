import _54MySqlInitialConfig as initial

connection = initial.connection
cursor = connection.cursor()

cursor.execute("SELECT * FROM customers LIMIT 5")
# From some position
# cursor.execute("SELECT * FROM customers LIMIT 5 OFFSET 2")

myresult = cursor.fetchall()

for x in myresult:
    print(x)

cursor.close()
connection.close()
