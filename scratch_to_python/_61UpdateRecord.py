import _54MySqlInitialConfig as initial

connection = initial.connection
cursor = connection.cursor()

sql = "UPDATE customers SET address = 'Canyon 123' WHERE address = 'Valley 345'"
# OR
"""
sql = "UPDATE customers SET address = %s WHERE address = %s"
val = ("Valley 345", "Canyon 123")

cursor.execute(sql, val)
"""

cursor.execute(sql)

connection.commit()

print(cursor.rowcount, "record(s) affected")

cursor.close()
connection.close()
