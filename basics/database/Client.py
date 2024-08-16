import MySQLdb

# Open database connection
db = MySQLdb.connect("localhost", "root", "admin", "myschema")

# prepare a cursor object using cursor() method
cursor = db.cursor()

# execute SQL query using execute() method.
cursor.execute("SELECT VERSION()")

# Fetch a single row using fetchone() method.
data = cursor.fetchone()
print("Database version : %s " % data)

try:
    # Execute the SQL command
    cursor.execute("select * from employee")
    # Fetch all the rows in a list of lists.
    results = cursor.fetchall()
    for row in results:
        #print(row)
        id = row[0]
        name = row[1]
        # Now print fetched result
        print("id=%d,name=%s" % \
              (id, name))

except Exception as e:
    print("Error: unable to fecth data: " + e)  # disconnect from server

db.close()
