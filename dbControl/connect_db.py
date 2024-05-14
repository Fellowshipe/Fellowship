import psycopg2

def connectDB():
    try:
        conn = psycopg2.connect(
            dbname="JungoNara", 
            user="postgres", 
            password="admin", 
            host="localhost", 
            port="5432"
        )
        return conn
    except Exception as e:
        print("Database connection failed due to {}".format(e))