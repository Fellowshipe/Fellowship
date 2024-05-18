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
        RETURNING id
        '''
        data = (title, price, member_level, post_date, product_status, 
                payment_method, shipping_method, transaction_region, description)
        cur.execute(query, data)
        try:
            last_id = cur.fetchone()[0]
            #print("id값", last_id)
        except:
            print("id값 추출 불가")
        conn.commit()
        cur.close()
    except psycopg2.DatabaseError as e:
        print(f"Error {e}")
        sys.stderr.write(f"Error {e}\n")
        conn.rollback()
        cur.close()
        return False
    return last_id