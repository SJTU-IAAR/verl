# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fcntl
import json
import logging
import os
import tempfile
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class GlobalRateLimiter:
    """
    Cross-process global rate limiter using file locks.
    
    This implementation ensures rate limiting across multiple processes/nodes,
    which is useful in distributed training environments where multiple GPUs
    or nodes need to coordinate tool execution rates.
    """
    
    def __init__(self, rate_limit_per_minute: float = 60.0, lock_file: Optional[str] = None):
        """
        Initialize the global rate limiter.
        
        Args:
            rate_limit_per_minute: Maximum requests per minute across all processes
            lock_file: Custom lock file path. If None, uses system temp directory
        """
        self.rate = rate_limit_per_minute / 60.0  # requests per second
        self.lock_file = lock_file or os.path.join(
            tempfile.gettempdir(), 
            "verl_global_rate_limiter.lock"
        )
        self.state_file = self.lock_file + ".state"
        
        logger.info(f"GlobalRateLimiter initialized: {rate_limit_per_minute} RPM, "
                   f"lock file: {self.lock_file}")
        
    def get_token(self) -> float:
        """
        Attempt to get a token from the global bucket.
        
        Returns:
            float: Time to wait in seconds before making the request.
                  0.0 means the request can proceed immediately.
        """
        max_wait = 10.0  # Maximum wait time to prevent deadlocks
        
        try:
            with open(self.lock_file, 'w') as f:
                # Try to acquire file lock (non-blocking)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Read current state
                now = time.time()
                last_request_time = now
                
                if os.path.exists(self.state_file):
                    try:
                        with open(self.state_file, 'r') as state_f:
                            state = json.load(state_f)
                            last_request_time = state.get('last_request_time', now)
                    except (json.JSONDecodeError, IOError):
                        # If state file is corrupted, start fresh
                        pass
                
                # Calculate wait time based on rate limit
                time_since_last = now - last_request_time
                min_interval = 1.0 / self.rate
                
                if time_since_last >= min_interval:
                    # Can proceed immediately
                    wait_time = 0.0
                    new_request_time = now
                else:
                    # Need to wait
                    wait_time = min_interval - time_since_last
                    new_request_time = last_request_time + min_interval
                
                # Update state file with new request time
                try:
                    with open(self.state_file, 'w') as state_f:
                        json.dump({
                            'last_request_time': new_request_time,
                            'updated_at': now
                        }, state_f)
                except IOError as e:
                    logger.warning(f"Failed to update state file: {e}")
                
                return min(wait_time, max_wait)
                
        except (IOError, OSError) as e:
            # If file lock fails, return conservative wait time
            logger.warning(f"Failed to acquire file lock: {e}, using fallback wait time")
            return 1.0
    
    def cleanup(self):
        """Clean up lock and state files."""
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
            logger.info("GlobalRateLimiter cleanup completed")
        except OSError as e:
            logger.warning(f"Failed to cleanup rate limiter files: {e}")


# Global rate limiter singleton
_global_rate_limiter = None
_global_rate_limiter_lock = threading.Lock()


def get_global_rate_limiter(rate_limit_per_minute: float = 60.0) -> GlobalRateLimiter:
    """
    Get or create the global rate limiter singleton.
    
    Args:
        rate_limit_per_minute: Rate limit for new instances
        
    Returns:
        GlobalRateLimiter instance
    """
    global _global_rate_limiter
    
    with _global_rate_limiter_lock:
        if _global_rate_limiter is None:
            _global_rate_limiter = GlobalRateLimiter(
                rate_limit_per_minute=rate_limit_per_minute
            )
            logger.info(f"Created global rate limiter singleton: {rate_limit_per_minute} RPM")
    
    return _global_rate_limiter 