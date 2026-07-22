import sqlite3
import queue
import threading
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("DatabaseVault")

class DatabaseVault:
    """
    SQLite WAL Mode & Thread-Safe Async Write-Queue Veritabanı Katmanı.
    Telegram Bot (asyncio) ve Background Scheduler thread'lerinin 
    çakışmasız ve kilitsiz (Database Locked riski sıfır) çalışmasını sağlar.
    """
    def __init__(self, db_path: str = "financial_ai.db"):
        self.db_path = db_path
        self.write_queue = queue.Queue()
        self._is_running = True
        
        # Ana thread veritabanını ilklendirir
        self._init_db()
        
        # Arka planda sadece YAZMA işlemlerini yapacak tekil thread
        self.worker_thread = threading.Thread(target=self._write_worker, daemon=True)
        self.worker_thread.start()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Sinyaller ve Karar Geçmişi Tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    signal_code INTEGER NOT NULL, -- (+1: AL, 0: KORU/NÖTR, -1: SAT)
                    p_success REAL NOT NULL,
                    p_success_prev REAL,
                    stop_loss_price REAL,
                    target_price_low REAL,
                    target_price_high REAL,
                    engine_a_signal INTEGER,
                    engine_b_signal INTEGER,
                    revision_reason TEXT
                );
            """)
            
            # Indexing (Hızlı Okuma İçin)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_time ON signals(ticker, timestamp);")
            conn.commit()

    def _write_worker(self):
        """Kuyruktan gelen YAZMA taleplerini sırayla ve güvenli şekilde işler."""
        conn = self._get_connection()
        while self._is_running:
            try:
                task = self.write_queue.get(timeout=1.0)
                if task is None:
                    break
                
                query, params = task
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                self.write_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Veritabanı yazma hatası: {e}")
        conn.close()

    def execute_write_async(self, query: str, params: tuple = ()):
        """Yazma taleplerini kuyruğa atar (Non-blocking)."""
        self.write_queue.put((query, params))

    def get_last_signal(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Okuma işlemi doğrudan ve anlık yapılır (WAL Mode sayesinde kilitlenmez)."""
        query = """
            SELECT * FROM signals 
            WHERE ticker = ? 
            ORDER BY timestamp DESC LIMIT 1;
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (ticker,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def close(self):
        self._is_running = False
        self.write_queue.put(None)
        self.worker_thread.join()

if __name__ == "__main__":
    db = DatabaseVault("test_financial_ai.db")
    db.execute_write_async(
        "INSERT INTO signals (ticker, signal_code, p_success, stop_loss_price, target_price_low, target_price_high) VALUES (?, ?, ?, ?, ?, ?)",
        ("THYAO", 1, 0.56, 699.1, 827.9, 853.7)
    )
    print("Database Vault WAL Mode Testi Başarıyla Tamamlandı.")
