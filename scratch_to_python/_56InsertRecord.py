import _54MySqlInitialConfig as initial

sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
val = ("John", "Highway 21")
initial.mycursor.execute(sql, val)

initial.mydb.commit()

print(initial.mycursor.rowcount, "record inserted.")
