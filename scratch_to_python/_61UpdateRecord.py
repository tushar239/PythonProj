import _54MySqlInitialConfig as initial

connection = initial.connection
cursor = connection.cursor()

sql = "UPDATE customers SET address = 'Canyon 123' WHERE address = 'Valley 345'"

cursor.execute(sql)

connection.commit()

print(cursor.rowcount, "record(s) affected")
