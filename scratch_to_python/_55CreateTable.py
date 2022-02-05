import _54MySqlInitialConfig as initial

mycursor = initial.mycursor

mycursor.execute("CREATE TABLE customers (name VARCHAR(255), address VARCHAR(255))")

mycursor.execute("SHOW TABLES")

for x in mycursor:
    print(x)
