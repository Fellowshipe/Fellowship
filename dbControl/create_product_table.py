def create_product_table(conn):
    cur = conn.cursor()
    create_product_table_query = '''
    CREATE TABLE products (
        id SERIAL PRIMARY KEY,
        title VARCHAR(64) NOT NULL,
        price VARCHAR(15) NOT NULL,
        member_level VARCHAR(10) NOT NULL,
        post_date TIMESTAMP WITH TIME ZONE NOT NULL,
        product_status VARCHAR(100),
        payment_method VARCHAR(100),
        shipping_method VARCHAR(100),
        transaction_region VARCHAR(100),
        description TEXT
    );
    '''
    cur.excute(create_product_table_query)
    conn.commit()
    cur.close()