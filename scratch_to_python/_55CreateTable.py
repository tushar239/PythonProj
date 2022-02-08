import _54MySqlInitialConfig as initial

cursor = initial.cursor
connection = initial.connection

cursor.execute("CREATE TABLE customers (name VARCHAR(255), address VARCHAR(255))")

cursor.execute("SHOW TABLES")

for x in cursor:
    print(x)

cursor.close()
connection.close()