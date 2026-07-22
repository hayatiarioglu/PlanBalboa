import os
import sqlite3
import logging
import queue
import threading
from typing import Optional, Dict, Any, List

logger = logging.getLogger("DatabaseVault")

class DatabaseVault:
    """
    THREAD-SAFE ASYNC WRITE-QUEUE SQLITE DATABASE VAULT.
    SQLite WAL Mode + Async Queue Worker.
    Sürüm 13.1 DSS Sinyalleri ve Kullanıcı Özel Takip Listesi (Watchlist) Tabloları.
    """

    def __init__(self, db_path: str = "data/financial_ai.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.write_queue = queue.Queue()
        self.is_running = True

        self._init_db()

        self.worker_thread = threading.Thread(target=self._write_worker, daemon=True)
        self.worker_thread.start()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                    signal_code INTEGER NOT NULL,
                    p_success REAL NOT NULL,
                    p_success_prev REAL,
                    stop_loss_price REAL NOT NULL,
                    target_price_low REAL NOT NULL,
                    target_price_high REAL NOT NULL,
                    engine_a_signal INTEGER NOT NULL,
                    engine_b_signal INTEGER NOT NULL,
                    revision_reason TEXT NOT NULL,
                    days_held INTEGER NOT NULL DEFAULT 0
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL UNIQUE,
                    added_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_ticker_id ON signals(ticker, id);")
            conn.commit()
        logger.info("SQLite WAL Veritabanı ve Watchlist Tablosu Başarıyla İlklendirildi.")

    def _write_worker(self):
        while self.is_running or not self.write_queue.empty():
            try:
                task = self.write_queue.get(timeout=1.0)
                if task is None:
                    break

                sql, params = task
                try:
                    with self._get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(sql, params)
                        conn.commit()
                except Exception as e:
                    logger.error(f"Async DB Yazma Hatası: {e} | SQL: {sql}")
                finally:
                    self.write_queue.task_done()
            except queue.Empty:
                continue

    def execute_write_async(self, sql: str, params: tuple = ()):
        self.write_queue.put((sql, params))

    def get_last_signal(self, ticker: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM signals WHERE ticker = ? ORDER BY id DESC LIMIT 1;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (ticker,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def add_to_watchlist(self, ticker: str) -> bool:
        """Kullanıcının özel takip listesine hisse ekler."""
        sql = "INSERT OR IGNORE INTO watchlist (ticker) VALUES (?);"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (ticker.upper(),))
            conn.commit()
            return cursor.rowcount > 0

    def remove_from_watchlist(self, ticker: str) -> bool:
        """Kullanıcının özel takip listesinden hisse çıkarır."""
        sql = "DELETE FROM watchlist WHERE ticker = ?;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (ticker.upper(),))
            conn.commit()
            return cursor.rowcount > 0

    def get_watchlist(self) -> List[str]:
        """Kullanıcının tüm özel takip hisselerini döner."""
        query = "SELECT ticker FROM watchlist ORDER BY id ASC;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [row["ticker"] for row in cursor.fetchall()]

    def close(self):
        self.is_running = False
        self.write_queue.put(None)
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)
        logger.info("DatabaseVault Güvenle Kapatıldı.")
