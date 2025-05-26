#!/usr/bin/env python3
"""
Simple rate limit test script to verify our rate limiting implementation.
"""

import time
import sys
import os

# Add the parent directory to the path to import ToolClient
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from verl.workers.rollout.vllm_rollout.tool_client import ToolClient

def test_rate_limiting():
    """Test rate limiting with various configurations"""
    
    # Test configuration
    server_url = "http://127.0.0.1:30003"
    rate_limit = 100  # 60 requests per minute = 1 request per second
    max_workers = 5
    
    print(f"=== Rate Limiting Test ===")
    print(f"Server URL: {server_url}")
    print(f"Rate Limit: {rate_limit} requests/minute")
    print(f"Max Workers: {max_workers}")
    print(f"Expected rate: {rate_limit/60:.2f} requests/second")
    print("-" * 40)
    
    # Initialize ToolClient with strict rate limiting
    tool_client = ToolClient(
        server_url=server_url,
        max_workers=max_workers,
        rate_limit_per_minute=rate_limit,
        request_interval=0.5,  # Additional 0.5s between requests
        timeout=30.0
    )
    
    # Test 1: Single requests
    print("\n=== Test 1: Single Requests ===")
    start_time = time.time()
    
    for i in range(5):
        print(f"Sending request {i+1}/5...")
        result = tool_client._execute_single(f'print("Test request {i+1}")')
        elapsed = time.time() - start_time
        print(f"  Request {i+1} completed in {elapsed:.2f}s total, success: {result.get('success', False)}")
    
    total_time = time.time() - start_time
    actual_rate = 5 / total_time * 60  # requests per minute
    print(f"\nSingle requests test:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Actual rate: {actual_rate:.1f} requests/minute")
    print(f"  Expected rate: {rate_limit} requests/minute")
    print(f"  Rate control effective: {actual_rate <= rate_limit * 1.1}")  # Allow 10% tolerance
    
    # Test 2: Batch requests
    print("\n=== Test 2: Batch Requests ===")
    batch_codes = [f'print("Batch request {i+1}")' for i in range(10)]
    
    start_time = time.time()
    results = tool_client.batch_execute(batch_codes)
    total_time = time.time() - start_time
    
    successful_requests = sum(1 for r in results if r.get('success', False))
    actual_rate = len(batch_codes) / total_time * 60  # requests per minute
    
    print(f"\nBatch requests test:")
    print(f"  Total requests: {len(batch_codes)}")
    print(f"  Successful requests: {successful_requests}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Actual rate: {actual_rate:.1f} requests/minute")
    print(f"  Expected rate: {rate_limit} requests/minute")
    print(f"  Rate control effective: {actual_rate <= rate_limit * 1.1}")  # Allow 10% tolerance
    
    print(f"\n=== Test Summary ===")
    print(f"Rate limiting appears to be {'WORKING' if actual_rate <= rate_limit * 1.1 else 'NOT WORKING'}")
    
    return actual_rate <= rate_limit * 1.1

if __name__ == "__main__":
    try:
        success = test_rate_limiting()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1) 