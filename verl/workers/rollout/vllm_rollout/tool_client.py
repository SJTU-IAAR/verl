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
"""
Tool Client for executing code via an HTTP API.
"""
import time
import json
import logging
import requests
import threading
import concurrent.futures
from typing import List, Dict, Any, Optional
import ray
import ray.util.scheduling_strategies
import os

logger = logging.getLogger(__name__)

# Add a flag to control verbose logging
VERBOSE_CLIENT_LOGGING = True  # Set to False to disable detailed logging

# Helper function for full code and output logging
def log_detailed(message, content=None):
    if VERBOSE_CLIENT_LOGGING:
        log_entry = f"[TOOL_CLIENT] {message}"  # Start with prefix
        if content is not None:
            # Add content, using markers for clarity
            log_entry += f":\n---\n{content}\n---"
        log_entry += " [/TOOL_CLIENT]"  # Add suffix
        # Use logger.info for consistency with existing logging
        logger.info(log_entry)

class TokenBucket:
    """
    Token bucket rate limiter to control request rate.
    This helps ensure that requests are sent at a consistent rate, even with multiple threads.
    """
    def __init__(self, rate_limit_per_minute: float = 120.0, burst_size: int = 10):
        """
        Initialize the token bucket rate limiter.
        
        Args:
            rate_limit_per_minute: Maximum number of requests per minute
            burst_size: Maximum number of tokens that can be accumulated (controls burst capacity)
        """
        self.rate = rate_limit_per_minute / 60.0  # Tokens per second
        self.max_tokens = burst_size
        self.tokens = burst_size  # Start with full bucket
        self.last_update = time.time()
        self.lock = threading.Lock()
        
        logger.info(f"Rate limiter initialized: {rate_limit_per_minute} requests/minute "
                   f"({self.rate:.2f} requests/second), burst size: {burst_size}")
    
    def get_token(self) -> float:
        """
        Attempt to get a token from the bucket.
        
        Returns:
            Time to wait in seconds if no token is immediately available,
            or 0 if a token was acquired
        """
        with self.lock:
            # Update tokens based on time elapsed
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1.0:
                # Token available, consume it
                self.tokens -= 1.0
                return 0.0
            else:
                # No token available, calculate wait time
                wait_time = (1.0 - self.tokens) / self.rate
                return wait_time

@ray.remote
class MasterForwarder:
    """Ray actor that runs on master node to forward tool requests."""
    
    def __init__(self, server_url: str, timeout: float = 30.0):
        # Use localhost when running on master node since actor is scheduled there
        # Replace any IP with localhost since we're running on the master node
        if server_url.startswith('http://127.0.0.1:') or server_url.startswith('http://localhost:'):
            self.server_url = server_url
        else:
            # Extract port from server_url and use localhost
            import re
            port_match = re.search(r':(\d+)', server_url)
            port = port_match.group(1) if port_match else '30003'
            self.server_url = f"http://127.0.0.1:{port}"
        
        self.timeout = timeout
        logger.info(f"MasterForwarder initialized with URL: {self.server_url}")
        
    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute code on the master node and return results."""
        try:
            payload = {"code": code}
            response = requests.post(
                f"{self.server_url}/execute",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Ensure success field exists
            if 'success' not in result:
                result['success'] = 'error' not in result or not result['error']
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Master forwarding failed: {str(e)}",
                "output": f"Error in master forwarding: {str(e)}"
            }

class ToolClient:
    """
    Client for executing code via an external HTTP service.
    Handles parallel execution, retries, and error handling.
    """
    
    def __init__(self, 
                 server_url: str,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: float = 30.0,
                 request_interval: float = 0.1,
                 max_workers: int = 10,
                 rate_limit_per_minute: float = 120.0,
                 master_ip: str = None):
        """
        Initialize the ToolClient.
        
        Args:
            server_url: URL of the tool execution server
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            timeout: Timeout for HTTP requests in seconds
            request_interval: Interval between concurrent requests to avoid overloading the server
            max_workers: Maximum number of concurrent requests
            rate_limit_per_minute: Maximum number of requests per minute
            master_ip: IP address of the master node (if None, auto-detect)
        """
        self.server_url = server_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.request_interval = request_interval
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self.master_ip = "192.168.154.83"  # Master node IP
        
        # Initialize rate limiter
        self.rate_limiter = TokenBucket(
            rate_limit_per_minute=rate_limit_per_minute,
            burst_size=min(20, max(5, int(rate_limit_per_minute/30.0)))  # Dynamic burst size
        )
        
        # Check if we're running on a Ray worker node and need to use master forwarding
        self.use_master_forwarding = self._should_use_master_forwarding()
        if self.use_master_forwarding:
            logger.info(f"Running on Ray worker node - will forward requests through master node ({self.master_ip})")
        
        logger.info(f"ToolClient initialized with server URL: {server_url}")
        logger.info(f"Request limiting: {rate_limit_per_minute} requests/minute "
                   f"with max {max_workers} concurrent workers")
        
        # Test server connectivity
        self._test_server_connection()
    
    def _should_use_master_forwarding(self) -> bool:
        """Check if we should use Ray master node forwarding."""
        try:
            # Check if Ray is initialized
            if not ray.is_initialized():
                return False
            
            # Simple IP-based detection
            import socket
            import subprocess
            
            # Try multiple methods to get current IP
            current_ip = None
            try:
                # Method 1: socket.gethostbyname
                current_ip = socket.gethostbyname(socket.gethostname())
            except:
                try:
                    # Method 2: hostname -I command
                    result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and result.stdout.strip():
                        current_ip = result.stdout.strip().split()[0]
                except:
                    pass
            
            # Get master IP from environment variable or use default
            master_ip = os.environ.get('MASTER_ADDR', '192.168.154.83')
            
            if current_ip == master_ip:
                logger.info(f"Running on master node ({current_ip}) - direct connection")
                return False
            else:
                logger.info(f"Running on worker node ({current_ip}) - using master forwarding")
                return True
            
        except Exception as e:
            logger.warning(f"Could not determine node type: {e}, using direct connection")
            return False
    
    def _get_master_forwarder(self):
        """Get or create the master forwarder actor."""
        if not hasattr(self, '_master_forwarder'):
            try:
                # Get master IP from environment
                master_ip = os.environ.get('MASTER_ADDR', '192.168.154.83')
                
                # Try to schedule actor on master node using node affinity
                try:
                    # Get all nodes and find master node ID
                    nodes = ray.nodes()
                    master_node_id = None
                    
                    for node in nodes:
                        if node.get('Alive', False) and node.get('NodeManagerAddress') == master_ip:
                            master_node_id = node['NodeID']
                            break
                    
                    if master_node_id:
                        # Create actor with node affinity to master
                        self._master_forwarder = MasterForwarder.options(
                            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                                node_id=master_node_id,
                                soft=False
                            )
                        ).remote(self.server_url, self.timeout)
                        logger.info(f"Created master forwarder actor on master node {master_ip} (node_id: {master_node_id})")
                    else:
                        # Fallback: create without specific scheduling
                        logger.warning(f"Could not find master node {master_ip}, creating actor without node affinity")
                        self._master_forwarder = MasterForwarder.remote(self.server_url, self.timeout)
                        logger.info("Created master forwarder actor (fallback)")
                        
                except Exception as scheduling_error:
                    logger.warning(f"Node affinity scheduling failed: {scheduling_error}, using default scheduling")
                    # Fallback: create without specific scheduling
                    self._master_forwarder = MasterForwarder.remote(self.server_url, self.timeout)
                    logger.info("Created master forwarder actor (fallback)")
                    
            except Exception as e:
                logger.error(f"Failed to create master forwarder: {e}")
                raise
        return self._master_forwarder
    
    def _test_server_connection(self):
        """Test if the tool server is available."""
        try:
            test_code = "print('Hello, Tool!')"
            log_detailed("Testing server connection with code", test_code)
            
            # Apply rate limiting even for test connections
            wait_time = self.rate_limiter.get_token()
            if wait_time > 0:
                logger.info(f"Rate limit applied for server test: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
            
            # Use the same logic as _execute_single for consistency
            if self.use_master_forwarding:
                # Test through master forwarding
                forwarder = self._get_master_forwarder()
                result = ray.get(forwarder.execute_code.remote(test_code))
                if result.get('success', False):
                    logger.info("Tool server connection test successful (via master forwarding)")
                else:
                    logger.warning(f"Tool server test failed via master forwarding: {result.get('error', 'Unknown error')}")
            else:
                # Direct HTTP request
                response = requests.post(
                    f"{self.server_url}/execute",
                    json={"code": test_code},
                    timeout=self.timeout
                )
                response.raise_for_status()
                logger.info("Tool server connection test successful (direct)")
                
                if VERBOSE_CLIENT_LOGGING:
                    try:
                        result = response.json()
                        log_detailed("Server test response", result.get('output', ''))
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Tool server may not be available: {str(e)}")
    
    def _execute_single(self, code: str) -> Dict[str, Any]:
        """
        Execute a single code snippet with retries.
        
        Args:
            code: The code to execute
            
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        
        log_detailed("Executing code", code)
        
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting - wait for a token if necessary
                wait_time = self.rate_limiter.get_token()
                if wait_time > 0:
                    logger.debug(f"Rate limit applied: waiting {wait_time:.2f}s before next request")
                    time.sleep(wait_time)
                
                # Add slight delay between requests as a backup mechanism
                with self._lock:
                    time.sleep(self.request_interval)
                
                # Choose execution method based on whether we need master forwarding
                if self.use_master_forwarding:
                    # Use Ray master node forwarding
                    forwarder = self._get_master_forwarder()
                    result = ray.get(forwarder.execute_code.remote(code))
                else:
                    # Direct HTTP request
                    payload = {"code": code}
                    response = requests.post(
                        f"{self.server_url}/execute",
                        json=payload,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    result = response.json()
                
                result['execution_time'] = time.time() - start_time
                
                # Ensure success field exists
                if 'success' not in result:
                    result['success'] = 'error' not in result or not result['error']
                
                if result.get('success', False):
                    log_detailed("Code execution successful", result.get('output', ''))
                else:
                    log_detailed("Code execution failed", result.get('error', 'Unknown error'))
                
                return result
            
            except (requests.RequestException, json.JSONDecodeError, ray.exceptions.RayError) as e:
                logger.warning(f"Tool execution failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    # Use exponential backoff for retries
                    backoff_time = self.retry_delay * (2 ** attempt)
                    time.sleep(backoff_time)
                else:
                    error_msg = f"Tool execution failed after {self.max_retries} attempts: {str(e)}"
                    logger.error(error_msg)
                    log_detailed("Tool execution error", error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "output": f"Error executing tool: {str(e)}",
                        "execution_time": time.time() - start_time
                    }
        
        # Should not be reached if max_retries >= 1
        return {
            "success": False,
            "error": "Max retries reached but loop finished unexpectedly",
            "output": "Error: Unexpected state in tool execution",
            "execution_time": time.time() - start_time
        }
    
    def batch_execute(self, codes: List[str]) -> List[Dict[str, Any]]:
        """
        Execute multiple code snippets with proper rate limiting.
        
        Args:
            codes: List of code snippets to execute
            
        Returns:
            List of dictionaries with execution results
        """
        if not codes:
            return []
        
        # Filter out empty codes
        valid_codes = [(i, code) for i, code in enumerate(codes) if code and code.strip()]
        indices, valid_code_snippets = zip(*valid_codes) if valid_codes else ([], [])
        
        log_detailed(f"Executing {len(valid_code_snippets)} code snippets with rate limiting")
        
        # Calculate estimated time based on rate limiting
        estimated_time = len(valid_code_snippets) / (self.rate_limiter.rate if self.rate_limiter.rate > 0 else 1)
        logger.info(f"Executing {len(valid_code_snippets)} code snippets")
        logger.info(f"Rate limiting: {self.rate_limiter.rate*60:.1f} requests/minute, estimated time: {estimated_time:.1f}s")
        
        # Execute codes serially to ensure proper rate limiting
        # This is necessary because parallel execution can overwhelm the rate limiter
        results_map = {}
        start_time = time.time()
        
        for i, (idx, code) in enumerate(zip(indices, valid_code_snippets)):
            logger.debug(f"Executing code snippet {i+1}/{len(valid_code_snippets)} (index {idx})")
            results_map[idx] = self._execute_single(code)
            
            # Progress logging
            if (i + 1) % 5 == 0 or i + 1 == len(valid_code_snippets):
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = (len(valid_code_snippets) - i - 1) * avg_time
                logger.info(f"Progress: {i+1}/{len(valid_code_snippets)} complete, "
                          f"elapsed: {elapsed:.1f}s, estimated remaining: {remaining:.1f}s")
        
        total_time = time.time() - start_time
        actual_rate = len(valid_code_snippets) / total_time * 60  # requests per minute
        logger.info(f"Batch execution completed in {total_time:.1f}s, "
                   f"actual rate: {actual_rate:.1f} requests/minute")
        
        # Create full results list with placeholders for empty codes
        results = []
        for i in range(len(codes)):
            if i in results_map:
                results.append(results_map[i])
            else:
                # Empty code placeholder
                results.append({
                    "success": False,
                    "error": "No code provided",
                    "output": "",
                    "execution_time": 0.0
                })
        
        if VERBOSE_CLIENT_LOGGING:
            for i, result in enumerate(results):
                status = "Success" if result.get('success', False) else "Failed"
                log_detailed(f"Batch result {i} ({status})", 
                            result.get('output') if result.get('success', False) else result.get('error', 'Unknown error'))
        
        return results