import psycopg2

from dotenv import load_dotenv
import os

def connectDB():
    try:
        load_dotenv()

        dbname=os.getenv('DB_NAME')
        user=os.getenv('DB_USER')
        password=os.getenv('DB_PW') 
        host=os.getenv('DB_HOST') 
        port=os.getenv('DB_PORT')

        conn = psycopg2.connect(
            dbname=dbname, 
            user=user, 
            password=password, 
            host=host, 
            port=port
        )
        return conn
    except Exception as e:
        print("Database connection failed due to {}".format(e))