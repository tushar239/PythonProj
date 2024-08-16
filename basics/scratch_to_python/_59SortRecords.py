import _54MySqlInitialConfig as initial

connection = initial.connection
cursor = connection.cursor()

sql = "SELECT * FROM customers ORDER BY name"

cursor.execute(sql)

myresult = cursor.fetchall()

for x in myresult:
    print(x)

print("########################################################")
# Sort in descending order
sql = "SELECT * FROM customers ORDER BY name DESC"

cursor.execute(sql)

myresult = cursor.fetchall()

for x in myresult:
    print(x)

print("########################################################")

cursor.close()
connection.close()
