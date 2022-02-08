import _54MySqlInitialConfig as initial

connection = initial.connection
cursor = connection.cursor()

# The WHERE clause specifies which record(s) that should be deleted. If you omit the WHERE clause, all records will be deleted!
sql = "DELETE FROM customers WHERE address = 'Mountain 21'"

cursor.execute(sql)

# OR
"""
# It is considered a good practice to escape the values of any query
# This is to prevent SQL injections, which is a common web hacking technique to destroy or misuse your database.
# The mysql.connector module uses the placeholder %s to escape values

sql = "DELETE FROM customers WHERE address = %s"
adr = ("Mountain 21", )

cursor.execute(sql, adr)
"""

# connection.commit(). It is required to make the changes, otherwise no changes are made to the table.
connection.commit()

print(cursor.rowcount, "record(s) deleted")

cursor.close()
connection.close()
