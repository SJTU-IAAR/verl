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
                 max_workers: int = 10):
        """
        Initialize the ToolClient.
        
        Args:
            server_url: URL of the tool execution server
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            timeout: Timeout for HTTP requests in seconds
            request_interval: Interval between concurrent requests to avoid overloading the server
            max_workers: Maximum number of concurrent requests
        """
        self.server_url = server_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.request_interval = request_interval
        self.max_workers = max_workers
        self._lock = threading.Lock()
        
        logger.info(f"ToolClient initialized with server URL: {server_url}")
        
        # Test server connectivity
        self._test_server_connection()
    
    def _test_server_connection(self):
        """Test if the tool server is available."""
        try:
            test_code = "print('Hello, Tool!')"
            log_detailed("Testing server connection with code", test_code)
            
            response = requests.post(
                f"{self.server_url}/execute",
                json={"code": test_code},
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info("Tool server connection test successful")
            
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
        payload = {"code": code}
        start_time = time.time()
        
        log_detailed("Executing code", code)
        
        for attempt in range(self.max_retries):
            try:
                with self._lock:
                    # Add slight delay between requests to avoid overwhelming the server
                    time.sleep(self.request_interval)
                
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
            
            except (requests.RequestException, json.JSONDecodeError) as e:
                logger.warning(f"Tool execution failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
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
        Execute multiple code snippets in parallel.
        
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
        
        log_detailed(f"Executing {len(valid_code_snippets)} code snippets in parallel")
        
        # Execute valid codes in parallel
        results_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._execute_single, code): i for i, code in zip(indices, valid_code_snippets)}
            
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results_map[idx] = future.result()
                except Exception as e:
                    error_msg = f"Unexpected error in thread for code execution: {str(e)}"
                    logger.error(error_msg)
                    log_detailed(f"Thread error for code {idx}", error_msg)
                    results_map[idx] = {
                        "success": False,
                        "error": error_msg,
                        "output": f"Error: {str(e)}",
                        "execution_time": 0.0
                    }
        
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