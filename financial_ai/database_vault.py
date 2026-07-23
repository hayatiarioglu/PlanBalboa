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
    SQLite WAL Mode + Persistent Worker + Busy Timeout.
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
        conn = sqlite3.connect(self.db_path, timeout=60.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                    current_price REAL NOT NULL DEFAULT 0.0,
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

            try:
                cursor.execute("ALTER TABLE signals ADD COLUMN current_price REAL DEFAULT 0.0;")
            except sqlite3.OperationalError:
                pass

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL UNIQUE,
                    added_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                );
            """)
            conn.commit()

    def _write_worker(self):
        """Tek bir kalıcı bağlantı ile kuyruktaki yazma işlemlerini işler."""
        conn = self._get_connection()
        while self.is_running:
            try:
                task = self.write_queue.get(timeout=1.0)
                if task is None:
                    break

                sql, params = task
                try:
                    cursor = conn.cursor()
                    cursor.execute(sql, params)
                    conn.commit()
                except Exception as e:
                    logger.error(f"Async DB Yazma Hatası: {e} | SQL: {sql}")
                finally:
                    self.write_queue.task_done()
            except queue.Empty:
                continue

        try:
            conn.close()
        except Exception:
            pass

    def execute_write_async(self, sql: str, params: tuple = ()):
        self.write_queue.put((sql, params))

    def get_last_signal(self, ticker: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM signals WHERE ticker = ? ORDER BY id DESC LIMIT 1;"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (ticker.upper(),))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.OperationalError:
            return None

    def add_to_watchlist(self, ticker: str) -> bool:
        sql = "INSERT OR IGNORE INTO watchlist (ticker) VALUES (?);"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (ticker.upper(),))
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            return False

    def remove_from_watchlist(self, ticker: str) -> bool:
        sql = "DELETE FROM watchlist WHERE ticker = ?;"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (ticker.upper(),))
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            return False

    def get_watchlist(self) -> List[str]:
        query = "SELECT ticker FROM watchlist ORDER BY id ASC;"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                return [row["ticker"] for row in cursor.fetchall()]
        except Exception:
            return []

    def close(self):
        self.is_running = False
        self.write_queue.put(None)
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)
        logger.info("DatabaseVault Güvenle Kapatıldı.")
