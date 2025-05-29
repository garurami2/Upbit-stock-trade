import sqlite3
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="trading.sqlite"):
        self.conn = sqlite3.connect(db_path)
        self.setup_database()

    def setup_database(self):
        cursor = self.conn.cursor()
        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS trading_history(
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                timestamp DATETIME NOT NULL,
                                decision TEXT NOT NULL,
                                percentage REAL NOT NULL,
                                reason TEXT NOT NULL,
                                btc_balance REAL NOT NULL,
                                krw_balance REAL NOT NULL,
                                btc_avg_buy_price REAL NOT NULL,
                                btc_krw_price REAL NOT NULL
                            )
                       """)
        self.conn.commit()

    def record_trade(self, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price):
        cursor = self.conn.cursor()
        cursor.execute("""
                            INSERT INTO trading_history(
                                timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       """, (
            datetime.now(),
            decision,
            percentage,
            reason,
            btc_balance,
            krw_balance,
            btc_avg_buy_price,
            btc_krw_price
            )
        )
        self.conn.commit()