import _54MySqlInitialConfig as initial

cursor = initial.cursor
connection = initial.connection

sql = "DROP TABLE IF EXISTS users"
cursor.execute(sql)

sql = "DROP TABLE IF EXISTS products"
cursor.execute(sql)

sql = """CREATE TABLE IF NOT EXISTS products (
            id int AUTO_INCREMENT, 
            name VARCHAR(255), 
            PRIMARY KEY(id)
        )
        """
cursor.execute(sql)

sql = """CREATE TABLE IF NOT EXISTS users (
            id int AUTO_INCREMENT, 
            name VARCHAR(255), 
            fav int,
            PRIMARY KEY(id),
            FOREIGN KEY (fav) REFERENCES products(id)
        )
        """

cursor.execute(sql)

connection.commit()

sql = "INSERT INTO products (name) VALUES (%s)"
val = [
  ('Chocolate Heaven',),
  ('Tasty Lemons',),
  ('Vanilla Dreams',)
]
cursor.executemany(sql, val)

sql = "INSERT INTO users (name, fav) VALUES (%s, %s)"  # User %s for even int column also. Don't use %d.
val = [
  ('John', 1),
  ('Peter', 2),
  ('Amy', 3),
  ('Hannah', None),  # Use None to insert null in db
  ("Michael", None)
]
cursor.executemany(sql, val)

connection.commit()


sql = "SELECT \
  users.name AS user, \
  products.name AS favorite \
  FROM users \
  INNER JOIN products ON users.fav = products.id"  # You can use JOIN instead of INNER JOIN. They will both give you the same result.

cursor.execute(sql)

myresult = cursor.fetchall()

for x in myresult:
  print(x)

cursor.close()
connection.close()

