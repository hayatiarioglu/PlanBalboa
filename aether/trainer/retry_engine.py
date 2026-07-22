import time
import datetime
from typing import Callable, Any, Optional

class RetryWindowExpiredError(Exception):
    """Raised when the execution time leaves the allowed Friday 18:10 - Saturday 06:00 window."""
    pass

class MaxRetriesExceededError(Exception):
    """Raised when the maximum number of retries is reached without success."""
    pass

def is_within_nightly_window(now: Optional[datetime.datetime] = None) -> bool:
    """
    Checks if a given datetime is within the allowed nightly training window:
    Friday 18:10:00 to Saturday 06:00:00.
    """
    if now is None:
        now = datetime.datetime.now()
        
    weekday = now.weekday()  # Monday=0, ..., Friday=4, Saturday=5, Sunday=6
    
    # Friday 18:10 - 23:59:59
    if weekday == 4:
        if now.hour > 18 or (now.hour == 18 and now.minute >= 10):
            return True
        return False
        
    # Saturday 00:00 - 05:59:59
    if weekday == 5:
        if now.hour < 6:
            return True
        return False
        
    return False

class RetryEngine:
    """
    Retry Engine with Exponential Backoff and Time Window Enforcement (Armor 1).
    - Max 10 attempts.
    - Exponential delay (base_delay * 2^(attempt-1)).
    - Aborts if execution time falls outside Friday 18:10 - Saturday 06:00.
    """
    
    def __init__(
        self,
        max_retries: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        enforce_time_window: bool = True,
        sleep_fn: Callable[[float], None] = time.sleep
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.enforce_time_window = enforce_time_window
        self.sleep_fn = sleep_fn
        
    def execute(self, task_fn: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Executes task_fn with retry armor.
        """
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            # Check time window constraint before each attempt
            if self.enforce_time_window:
                if not is_within_nightly_window():
                    raise RetryWindowExpiredError(
                        f"Attempt {attempt} aborted: Current time is outside the allowed window (Friday 18:10 - Saturday 06:00)."
                    )
            
            try:
                # Try executing the task
                result = task_fn(*args, **kwargs)
                return result
            except Exception as e:
                last_exception = e
                if attempt == self.max_retries:
                    break
                
                # Compute exponential backoff delay
                delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                self.sleep_fn(delay)
                
        raise MaxRetriesExceededError(
            f"Failed after {self.max_retries} attempts. Last error: {last_exception}"
        ) from last_exception
