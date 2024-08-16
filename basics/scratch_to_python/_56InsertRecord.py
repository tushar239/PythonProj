import _54MySqlInitialConfig as initial

connection = initial.connection
cursor = connection.cursor()

# It is considered a good practice to escape the values of any query
# This is to prevent SQL injections, which is a common web hacking technique to destroy or misuse your database.
# The mysql.connector module uses the placeholder %s to escape values
sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"  # User %s for even int column also. Don't use %d.
val = ("John", "Highway 21")
cursor.execute(sql, val)

# It is required to make the changes, otherwise no changes are made to the table.
connection.commit()

print(cursor.rowcount, "record inserted.")
# This is how you access rowid of inserted record
print("1 record inserted, ID:", cursor.lastrowid)


cursor.close()
connection.close()
