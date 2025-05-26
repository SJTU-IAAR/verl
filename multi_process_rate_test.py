#!/usr/bin/env python3
"""
Multi-process rate limit test script to verify cross-process global rate limiting.
"""

import time
import sys
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import tempfile

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the GlobalRateLimiter from our implementation
sys.path.append(os.path.join(os.path.dirname(__file__), 'verl', 'workers', 'rollout', 'vllm_rollout'))

def worker_function(worker_id, num_requests, rate_limit, lock_file):
    """Worker function that simulates tool requests from a single GPU process"""
    
    # Import here to avoid issues with multiprocessing
    from verl.workers.rollout.vllm_rollout.tool_vllm_rollout_spmd import GlobalRateLimiter
    
    global_limiter = GlobalRateLimiter(rate_limit_per_minute=rate_limit, lock_file=lock_file)
    
    results = []
    print(f"[Worker {worker_id}] Starting with {num_requests} requests")
    
    start_time = time.time()
    
    for i in range(num_requests):
        request_start = time.time()
        
        # Apply global rate limiting
        wait_time = global_limiter.get_token()
        if wait_time > 0:
            print(f"[Worker {worker_id}] Request {i+1}: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        # Simulate actual request (just a short delay)
        time.sleep(0.1)
        
        elapsed = time.time() - request_start
        total_elapsed = time.time() - start_time
        
        result = {
            'worker_id': worker_id,
            'request_id': i+1,
            'wait_time': wait_time,
            'total_elapsed': total_elapsed,
            'request_time': elapsed
        }
        results.append(result)
        
        print(f"[Worker {worker_id}] Request {i+1} completed in {elapsed:.2f}s (total: {total_elapsed:.2f}s)")
    
    total_time = time.time() - start_time
    actual_rate = num_requests / total_time * 60  # requests per minute
    
    print(f"[Worker {worker_id}] Completed {num_requests} requests in {total_time:.2f}s (rate: {actual_rate:.1f} RPM)")
    
    return {
        'worker_id': worker_id,
        'results': results,
        'total_time': total_time,
        'actual_rate': actual_rate
    }

def test_multi_process_rate_limiting():
    """Test global rate limiting across multiple processes"""
    
    # Test configuration
    rate_limit = 100  # 60 requests per minute = 1 request per second
    num_workers = 5  # Simulate 4 GPU processes
    requests_per_worker = 100  # 3 requests per worker
    total_requests = num_workers * requests_per_worker
    
    # Create temporary lock file
    lock_file = os.path.join(tempfile.gettempdir(), f"test_rate_limiter_{int(time.time())}.lock")
    
    print(f"=== Multi-Process Rate Limiting Test ===")
    print(f"Rate Limit: {rate_limit} requests/minute")
    print(f"Number of Workers (simulated GPUs): {num_workers}")
    print(f"Requests per Worker: {requests_per_worker}")
    print(f"Total Requests: {total_requests}")
    print(f"Expected minimum time: {total_requests / (rate_limit/60):.1f}s")
    print(f"Lock file: {lock_file}")
    print("-" * 60)
    
    # Run workers in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        # Submit all workers
        for worker_id in range(num_workers):
            future = executor.submit(worker_function, worker_id, requests_per_worker, rate_limit, lock_file)
            futures.append(future)
        
        # Collect results
        all_results = []
        for future in futures:
            try:
                result = future.result(timeout=120)  # 2 minute timeout
                all_results.append(result)
            except Exception as e:
                print(f"Worker failed: {e}")
    
    total_test_time = time.time() - start_time
    
    # Analyze results
    print(f"\n=== Results Analysis ===")
    print(f"Total test time: {total_test_time:.2f}s")
    
    all_request_times = []
    for worker_result in all_results:
        worker_id = worker_result['worker_id']
        worker_time = worker_result['total_time']
        worker_rate = worker_result['actual_rate']
        
        print(f"Worker {worker_id}: {worker_time:.2f}s, rate: {worker_rate:.1f} RPM")
        
        for request in worker_result['results']:
            all_request_times.append(request['total_elapsed'])
    
    # Calculate overall rate
    overall_rate = total_requests / total_test_time * 60
    expected_min_time = total_requests / (rate_limit / 60)
    
    print(f"\nOverall Results:")
    print(f"  Total requests: {total_requests}")
    print(f"  Total time: {total_test_time:.2f}s")
    print(f"  Overall rate: {overall_rate:.1f} requests/minute")
    print(f"  Rate limit: {rate_limit} requests/minute")
    print(f"  Expected minimum time: {expected_min_time:.1f}s")
    print(f"  Rate control effective: {overall_rate <= rate_limit * 1.2}")  # 20% tolerance
    
    # Timeline analysis
    if all_request_times:
        all_request_times.sort()
        print(f"\nRequest Timeline:")
        for i, req_time in enumerate(all_request_times):
            print(f"  Request {i+1}: {req_time:.2f}s")
    
    # Cleanup
    try:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        if os.path.exists(lock_file + ".state"):
            os.remove(lock_file + ".state")
    except:
        pass
    
    # Return success if rate is controlled
    return overall_rate <= rate_limit * 1.2

if __name__ == "__main__":
    try:
        success = test_multi_process_rate_limiting()
        print(f"\n=== Test {'PASSED' if success else 'FAILED'} ===")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 