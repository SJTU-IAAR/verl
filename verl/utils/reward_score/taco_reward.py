import json
from typing import Dict, Any

def compute_score(solution_str: str, ground_truth: Dict[str, Any] = None, extra_info=None) -> float:
    """
    Calculates the reward for a TACO dataset item based on the pre-executed
    results returned by the tool server. This function adheres to the Verl
    framework's 'compute_score' interface.

    The reward is the fraction of test cases that passed. A test case is considered
    passed if its execution status is 'completed' and its actual output matches
    the expected output.

    :param solution_str: The JSON string returned from the tool server, which
                         contains a list of test case result dictionaries.
    :param ground_truth: The original dataset item. This is optional and not used here.
    :param extra_info: Extra information, not used.
    :return: A float between 0.0 and 1.0 representing the pass rate.
    """
    try:
        # The solution_str is expected to be a JSON string representing a list of results.
        execution_results = json.loads(solution_str)
    except (json.JSONDecodeError, TypeError):
        # If parsing fails, it means the script execution failed or returned
        # a malformed result. Award no points.
        return 0.0

    if not isinstance(execution_results, list) or not execution_results:
        return 0.0

    passed_count = 0
    total_cases = len(execution_results)

    for result in execution_results:
        # A test case passes only if the server confirms it completed successfully
        # AND the actual output matches the expected one.
        if isinstance(result, dict) and result.get("status") == "completed":
            # Cast to string for robustness, in case the output is a number (e.g., 4)
            # instead of a string ('4\n').
            actual_output = str(result.get("actual_output", "")).strip()
            expected_output = str(result.get("expected_output", "")).strip()
            if actual_output == expected_output:
                passed_count += 1

    return float(passed_count) / total_cases if total_cases > 0 else 0.0

if __name__ == '__main__':
    # This block tests the new compute_score function's logic.
    # It simulates different JSON outputs that would come from the server.

    # Testing the compute_score function

    # Test Case 1: All tests pass, including one with an integer output
    server_output_pass = json.dumps([
        {"status": "completed", "actual_output": "4\\n", "expected_output": "4\\n"},
        {"status": "completed", "actual_output": -1, "expected_output": "  -1  \n"}
    ])
    reward = compute_score(server_output_pass)
    print(f"Calculated reward for 'all pass' scenario: {reward}")
    assert reward == 1.0, f"Expected 1.0, but got {reward}"

    # Test Case 2: One test passes, one fails (wrong answer)
    server_output_partial = json.dumps([
        {"status": "completed", "actual_output": "4\\n", "expected_output": "4\\n"},
        {"status": "completed", "actual_output": 5, "expected_output": "-1\\n"}
    ])
    reward = compute_score(server_output_partial)
    print(f"Calculated reward for 'partial pass' scenario: {reward}")
    assert reward == 0.5, f"Expected 0.5, but got {reward}"

    # Test Case 3: One test passes, one has an execution error
    server_output_error = json.dumps([
        {"status": "completed", "actual_output": "4\\n", "expected_output": "4\\n"},
        {"status": "error", "traceback": "..."}
    ])
    reward = compute_score(server_output_error)
    print(f"Calculated reward for 'execution error' scenario: {reward}")
    assert reward == 0.5, f"Expected 0.5, but got {reward}"

    # Test Case 4: Malformed JSON output from server
    server_output_malformed = "This is not json"
    reward = compute_score(server_output_malformed)
    print(f"Calculated reward for 'malformed output' scenario: {reward}")
    assert reward == 0.0, f"Expected 0.0, but got {reward}"
    
    # Test Case 5: Empty result list
    server_output_empty = "[]"
    reward = compute_score(server_output_empty)
    print(f"Calculated reward for 'empty result' scenario: {reward}")
    assert reward == 0.0, f"Expected 0.0, but got {reward}"

    print("\\nAll tests passed!") 