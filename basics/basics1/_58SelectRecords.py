import _54MySqlInitialConfig as initial

connection = initial.connection
cursor = connection.cursor()

cursor.execute("SELECT * FROM customers")

"""
result = cursor.fetchall()  # result is tuple

for x in result:
    print(x)
"""
# OR
"""
for (name, address) in cursor:
    print(name, address)
"""
# OR
for row in cursor:
    # print(type(row))  # Tuple
    print(row)

print("#############################")
#######################################################################
# Select only the name and address columns:
cursor.execute("SELECT name, address FROM customers")

myresult = cursor.fetchall()

for x in myresult:
    print(x)

print("#############################")
##############################################################################################
# fetchone() - If you are only interested in one row, you can use the fetchone() method.
# When you do this, records are fetched lazily one by one through the connection. If there are records in the cursor which are not fetched and if you try to reuse or close the cursor, database will complain.
# To stop this happening, you have multiple options, in which putting 'limit' in sql query is the best
# https://stackoverflow.com/questions/29772337/python-mysql-connector-unread-result-found-when-using-fetchone
# 1. connection.cursor(buffered=True)
    # The reason is that without a buffered cursor, the results are "lazily" loaded, meaning that "fetchone" actually only fetches one row from the full result set of the query. When you will use the same cursor again, it will complain that you still have n-1 results (where n is the result set amount) waiting to be fetched. However, when you use a buffered cursor the connector fetches ALL rows behind the scenes and you just take one from the connector so the mysql db won't complain.
# 2. put 'LIMIT' in sql query

# cursor.execute("SELECT * FROM customers")
cursor.execute("SELECT * FROM customers LIMIT 0, 1")

myresult = cursor.fetchone()

print(myresult)

print("#############################")
# Select With a Filter
sql = "SELECT * FROM customers WHERE address ='Park Lane 38'"

cursor.execute(sql)

myresult = cursor.fetchall()

for x in myresult:
    print(x)

print("#############################")

sql = "SELECT * FROM customers WHERE address LIKE '%way%'"

cursor.execute(sql)

myresult = cursor.fetchall()

for x in myresult:
    print(x)

print("#############################")
sql = "SELECT * FROM customers WHERE address = %s"
adr = ("Yellow Garden 2", )

cursor.execute(sql, adr)

myresult = cursor.fetchall()

for x in myresult:
    print(x)

print("#############################")

cursor.close()
connection.close()
