
# db_utils.py
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def run_bin_query():
    query = """
        SELECT DISTINCT LEFT(card_no, 6) AS bin
        FROM public.altitude_transaction t
        LEFT JOIN public.altitude_project p ON t.project_id = p.project_id
        LEFT JOIN public.altitude_customers c ON t.txid = c.txid
        WHERE EXTRACT(YEAR FROM t.created_date) = EXTRACT(YEAR FROM CURRENT_DATE - INTERVAL '1 month')
          AND EXTRACT(MONTH FROM t.created_date) = EXTRACT(MONTH FROM CURRENT_DATE - INTERVAL '1 month')
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        df = pd.read_sql_query(query, conn)
        conn.close()
        bin_list = df['bin'].tolist()
        return bin_list
    except Exception as e:
        return []

