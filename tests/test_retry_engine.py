import unittest
import datetime
from unittest.mock import MagicMock
from aether.trainer.retry_engine import (
    RetryEngine,
    is_within_nightly_window,
    RetryWindowExpiredError,
    MaxRetriesExceededError
)

class TestRetryEngine(unittest.TestCase):
    def test_nightly_window_validation(self):
        """
        Validates time window logic: Friday 18:10 to Saturday 06:00.
        """
        # Friday 18:09:59 (Too early -> False)
        fri_early = datetime.datetime(2026, 7, 24, 18, 9, 59)
        self.assertFalse(is_within_nightly_window(fri_early))
        
        # Friday 18:10:00 (Valid -> True)
        fri_start = datetime.datetime(2026, 7, 24, 18, 10, 0)
        self.assertTrue(is_within_nightly_window(fri_start))
        
        # Friday 23:59:59 (Valid -> True)
        fri_late = datetime.datetime(2026, 7, 24, 23, 59, 59)
        self.assertTrue(is_within_nightly_window(fri_late))
        
        # Saturday 05:59:59 (Valid -> True)
        sat_end = datetime.datetime(2026, 7, 25, 5, 59, 59)
        self.assertTrue(is_within_nightly_window(sat_end))
        
        # Saturday 06:00:00 (Too late -> False)
        sat_over = datetime.datetime(2026, 7, 25, 6, 0, 0)
        self.assertFalse(is_within_nightly_window(sat_over))
        
        # Sunday 12:00:00 (False)
        sun_noon = datetime.datetime(2026, 7, 26, 12, 0, 0)
        self.assertFalse(is_within_nightly_window(sun_noon))

    def test_successful_execution_first_try(self):
        """
        Verifies task returning result on first attempt.
        """
        mock_sleep = MagicMock()
        engine = RetryEngine(max_retries=10, base_delay=1.0, enforce_time_window=False, sleep_fn=mock_sleep)
        
        task = MagicMock(return_value="SUCCESS")
        result = engine.execute(task)
        
        self.assertEqual(result, "SUCCESS")
        self.assertEqual(task.call_count, 1)
        mock_sleep.assert_not_called()

    def test_retry_success_after_failures(self):
        """
        Verifies retry with exponential backoff on intermittent errors.
        """
        mock_sleep = MagicMock()
        engine = RetryEngine(max_retries=10, base_delay=1.0, enforce_time_window=False, sleep_fn=mock_sleep)
        
        # Fails twice, succeeds on 3rd attempt
        task = MagicMock(side_effect=[ValueError("Error 1"), ValueError("Error 2"), "OK"])
        result = engine.execute(task)
        
        self.assertEqual(result, "OK")
        self.assertEqual(task.call_count, 3)
        # Delays should be 1.0 (for attempt 1 failure) and 2.0 (for attempt 2 failure)
        mock_sleep.assert_has_calls([unittest.mock.call(1.0), unittest.mock.call(2.0)])

    def test_max_retries_exceeded(self):
        """
        Verifies MaxRetriesExceededError is raised after 10 failures.
        """
        mock_sleep = MagicMock()
        engine = RetryEngine(max_retries=10, base_delay=1.0, enforce_time_window=False, sleep_fn=mock_sleep)
        
        task = MagicMock(side_effect=RuntimeError("Persistent failure"))
        
        with self.assertRaises(MaxRetriesExceededError):
            engine.execute(task)
            
        self.assertEqual(task.call_count, 10)
        self.assertEqual(mock_sleep.call_count, 9)

if __name__ == "__main__":
    unittest.main()
