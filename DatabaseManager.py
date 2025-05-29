import sqlite3
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="trading.sqlite"):
        self.conn = sqlite3.connect(db_path)
        self.setup_database()

    # DB 생성
    def setup_database(self):
        cursor = self.conn.cursor()
        # 기존 거래 기록 테이블
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

        # 거래 반성 일기 테이블 추가
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_reflection(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trading_id INTEGER NOT NULL,
                reflection_date DATETIME NOT NULL,
                market_condition TEXT NOT NULL,
                decision_analysis TEXT NOT NULL,
                improvement_point TEXT NOT NULL,
                success_rate REAL NOT NULL,
                learning_points TEXT NOT NULL,
                FOREIGN KEY (trading_id) REFERENCES trading_history(id)
            )
        """)
        self.conn.commit()

    # 최근 거래 내역 조회
    def get_recent_trades(self, limit=10):
        cursor = self.conn.cursor()
        cursor.execute("""
                       SELECT *
                       FROM trading_history
                       ORDER BY timestamp DESC
                           LIMIT ?
                       """, (limit,))
        return cursor.fetchall()

    # 최근 방성 일기 조회
    def get_reflection_history(self, limit=10):
        cursor = self.conn.cursor()
        cursor.execute("""
                       SELECT r.*, h.decision, h.percentage, h.btc_krw_price
                       FROM trading_reflection r
                                JOIN trading_history h ON r.trading_id = h.id
                       ORDER BY r.reflection_date DESC LIMIT ?
                       """, (limit,))
        return cursor.fetchall()

    # 거래 내역 저장
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
        return cursor.lastrowid

    # 반성 일기 추가
    def add_reflection(self, reflection_data):
        cursor = self.conn.cursor()
        cursor.execute("""
                       INSERT INTO trading_reflection(
                           trading_id, reflection_date, market_condition,
                           decision_analysis, improvement_point, success_rate,
                           learning_points) VALUES (?, ?, ?, ?, ?, ?, ?)
                       """,(
            reflection_data['trading_id'],
            reflection_data['reflection_date'],
            reflection_data['market_condition'],
            reflection_data['decision_analysis'],
            reflection_data['improvement_points'],
            reflection_data['success_rate'],
            reflection_data['learning_points']
        ))
        self.conn.commit()

    #