import psycopg2
import sys

def insert_product(conn, 
                   title, 
                   price, 
                   member_level, 
                   post_date, 
                   product_status, 
                   payment_method, 
                   shipping_method, 
                   transaction_region, 
                   description):
    try:
        cur = conn.cursor()
        query = '''
        INSERT INTO cellphone (title, price, member_level, post_date, product_status, 
        payment_method, shipping_method, transaction_region, description) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''
        data = (title, price, member_level, post_date, product_status, 
                payment_method, shipping_method, transaction_region, description)
        cur.execute(query, data)


        conn.commit()
        cur.close()
    except psycopg2.DatabaseError as e:
        print(f"Error {e}")
        sys.stderr.write(f"Error {e}\n")
        conn.rollback()
        cur.close()
        return False
