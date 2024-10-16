import _54MySqlInitialConfig as initial

connection = initial.connection
cursor = connection.cursor()

# sql = "DROP TABLE customers"
# To avoid any error, use 'DROP TABLE IF EXISTS ...'
sql = "DROP TABLE IF EXISTS customers"

cursor.execute(sql)

cursor.close()
connection.close()
