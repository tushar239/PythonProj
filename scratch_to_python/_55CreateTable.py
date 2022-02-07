import _54MySqlInitialConfig as initial

mycursor = initial.cursor
connection = initial.connection

mycursor.execute("CREATE TABLE customers (name VARCHAR(255), address VARCHAR(255))")

mycursor.execute("SHOW TABLES")

for x in mycursor:
    print(x)

mycursor.close()
connection.close()