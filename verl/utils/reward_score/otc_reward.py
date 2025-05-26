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
OTC (Optimal Tool Call) Reward Implementation

This module implements the OTC reward mechanism from the paper, which aims to optimize
tool usage efficiency by encouraging models to use the minimal but sufficient number
of tool calls to solve problems.

Two implementations are provided:
1. OTC-PPO: Simple version that penalizes tool usage directly
2. OTC-GRPO: Group-based optimization that finds optimal tool call counts
"""

import re
import math
import random
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict


def count_tool_calls(solution_str: str, tool_patterns: Optional[List[str]] = None) -> int:
    """
    Count the number of tool calls in a solution string.
    
    Args:
        solution_str: The solution text to analyze
        tool_patterns: List of regex patterns to match tool calls
                      If None, uses default patterns for common tools
    
    Returns:
        Number of tool calls found
    """
    if tool_patterns is None:
        # Default patterns for common tool usage
        tool_patterns = [
            r'<code>.*?</code>',           # Code execution blocks
            r'<search>.*?</search>',       # Search queries
            r'web_search\s*\(',            # Web search function calls
            r'search_r1\s*\(',             # Search R1 function calls
            r'calculator\s*\(',            # Calculator function calls
            r'python\s*\(',                # Python function calls
            r'tool_call\s*\(',             # Generic tool calls
        ]
    
    total_calls = 0
    for pattern in tool_patterns:
        matches = re.findall(pattern, solution_str, re.DOTALL | re.IGNORECASE)
        total_calls += len(matches)
    
    return total_calls


def compute_otc_ppo_reward(m: int, c: float = 1.0) -> float:
    """
    Compute OTC-PPO reward based on tool call count.
    
    Formula: r_tool = cos(m * π / (2m + c))
    
    Args:
        m: Number of tool calls in current trajectory
        c: Smooth constant controlling reward decay rate
           - Smaller c: faster punishment, encourages less tool use
           - Larger c: more tolerant
    
    Returns:
        Tool usage reward (0 to 1)
    """
    if m == 0:
        return 1.0  # Perfect score for no tool usage
    
    # Compute the cosine reward
    denominator = 2 * m + c
    if denominator == 0:
        return 0.0
    
    angle = m * math.pi / denominator
    reward = math.cos(angle)
    
    # Ensure reward is in [0, 1] range
    return max(0.0, reward)


def compute_mapping_function(m: int, n: int) -> float:
    """
    Compute the mapping function f(m, n) for OTC-GRPO.
    
    Formula:
    f(m, n) = {
        0,           if m = 0 and n = 0
        m,           if n = 0  
        2nm/(m+n),   otherwise (harmonic mean variant)
    }
    
    Args:
        m: Current trajectory tool call count
        n: Estimated optimal tool call count
        
    Returns:
        Mapped value
    """
    if m == 0 and n == 0:
        return 0.0
    elif n == 0:
        return float(m)
    else:
        # Harmonic mean variant: 2nm/(m+n)
        denominator = m + n
        if denominator == 0:
            return 0.0
        return 2.0 * n * m / denominator


def compute_otc_grpo_reward(m: int, n: int, c: float = 1.0) -> float:
    """
    Compute OTC-GRPO reward based on current and optimal tool call counts.
    
    Formula:
    r_tool = {
        1,                           if f(m,n) = n = 0
        cos(m * π / (2m + c)),      if n = 0
        sin(f(m,n) * π / (2n)),     otherwise
    }
    
    Args:
        m: Number of tool calls in current trajectory
        n: Estimated optimal number of tool calls
        c: Smooth constant for cosine reward when n=0
        
    Returns:
        Tool usage reward (0 to 1)
    """
    f_mn = compute_mapping_function(m, n)
    
    # Case 1: f(m,n) = n = 0 (optimal solution requires no tools and none used)
    if f_mn == n and n == 0:
        return 1.0
    
    # Case 2: n = 0 (optimal solution requires no tools but tools were used)
    elif n == 0:
        return compute_otc_ppo_reward(m, c)
    
    # Case 3: General case (optimal solution requires tools)
    else:
        if n == 0:
            return 0.0  # Avoid division by zero
        
        angle = f_mn * math.pi / (2.0 * n)
        reward = math.sin(angle)
        
        # Ensure reward is in [0, 1] range
        return max(0.0, reward)


def find_optimal_tool_calls(correct_trajectories: List[str], 
                          tool_patterns: Optional[List[str]] = None) -> int:
    """
    Find the optimal (minimum) number of tool calls from correct trajectories.
    
    This is used in OTC-GRPO to estimate the optimal tool call count.
    
    Args:
        correct_trajectories: List of solution strings that led to correct answers
        tool_patterns: Patterns to match tool calls
        
    Returns:
        Minimum number of tool calls among correct trajectories
    """
    if not correct_trajectories:
        return 0
    
    tool_counts = []
    for trajectory in correct_trajectories:
        count = count_tool_calls(trajectory, tool_patterns)
        tool_counts.append(count)
    
    return min(tool_counts)


def compute_otc_reward(solution_str: str, 
                      ground_truth: Any,
                      method: str = "ppo",
                      base_reward_fn: Optional[callable] = None,
                      correct_trajectories: Optional[List[str]] = None,
                      tool_patterns: Optional[List[str]] = None,
                      c: float = 1.0,
                      alpha: float = 0.7,
                      beta: float = 0.3,
                      return_dict: bool = False) -> Union[float, Dict[str, Any]]:
    """
    Compute OTC (Optimal Tool Call) reward combining base reward with tool efficiency.
    
    Args:
        solution_str: The solution text to evaluate
        ground_truth: Ground truth for base reward calculation
        method: "ppo" for OTC-PPO or "grpo" for OTC-GRPO
        base_reward_fn: Function to compute base reward (e.g., accuracy)
                       If None, assumes the answer is correct (reward=1.0)
        correct_trajectories: List of correct solution strings (needed for GRPO)
        tool_patterns: Patterns to match tool calls
        c: Smooth constant for cosine reward
        alpha: Weight for base reward (default: 0.7)
        beta: Weight for tool efficiency reward (default: 0.3)
        return_dict: Whether to return detailed information
        
    Returns:
        Combined reward or dictionary with detailed information
    """
    # 10% random debug printing
    do_print = random.random() < 0.1
    
    # Count tool calls in current solution
    m = count_tool_calls(solution_str, tool_patterns)
    
    # Compute base reward (accuracy)
    if base_reward_fn is not None:
        base_reward = base_reward_fn(solution_str, ground_truth)
        if isinstance(base_reward, dict):
            base_score = base_reward.get("score", 0.0)
        else:
            base_score = float(base_reward)
    else:
        # Assume correct answer if no base reward function provided
        base_score = 1.0
    
    # Compute tool efficiency reward
    if method.lower() == "ppo":
        tool_reward = compute_otc_ppo_reward(m, c)
        n = None  # Not used in PPO method
    elif method.lower() == "grpo":
        if correct_trajectories is None:
            # Fallback to PPO method if no group information available
            tool_reward = compute_otc_ppo_reward(m, c)
            n = None
        else:
            n = find_optimal_tool_calls(correct_trajectories, tool_patterns)
            tool_reward = compute_otc_grpo_reward(m, n, c)
    else:
        raise ValueError(f"Unknown OTC method: {method}. Use 'ppo' or 'grpo'.")
    
    # Combine rewards
    final_reward = alpha * base_score + beta * tool_reward
    
    if do_print:
        print(f"======== OTC REWARD CALCULATION DEBUG (10% sample) ========")
        print(f"Method: {method.upper()}")
        print(f"Tool calls found (m): {m}")
        if n is not None:
            print(f"Optimal tool calls (n): {n}")
        print(f"Base reward: {base_score:.4f}")
        print(f"Tool efficiency reward: {tool_reward:.4f}")
        print(f"Alpha (base weight): {alpha}, Beta (tool weight): {beta}")
        print(f"Final combined reward: {final_reward:.4f}")
        print(f"Ground truth: {ground_truth}")
        print(f"Solution string (first 300 chars): {solution_str[:300]}...")
        if len(solution_str) > 300:
            print(f"Solution string (last 100 chars): ...{solution_str[-100:]}")
        print(f"============================================================")
    
    if return_dict:
        result = {
            "score": final_reward,
            "base_score": base_score,
            "tool_reward": tool_reward,
            "tool_calls": m,
            "method": method,
            "alpha": alpha,
            "beta": beta
        }
        if n is not None:
            result["optimal_tool_calls"] = n
        return result
    
    return final_reward


def compute_score_otc_ppo(solution_str: str, 
                         ground_truth: Any,
                         base_reward_fn: Optional[callable] = None,
                         c: float = 1.0,
                         alpha: float = 0.7,
                         beta: float = 0.3) -> Dict[str, Any]:
    """
    Convenience function for OTC-PPO reward calculation.
    
    Args:
        solution_str: Solution text
        ground_truth: Ground truth
        base_reward_fn: Base reward function
        c: Smooth constant
        alpha: Base reward weight
        beta: Tool reward weight
        
    Returns:
        Dictionary with detailed reward information
    """
    return compute_otc_reward(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method="ppo",
        base_reward_fn=base_reward_fn,
        c=c,
        alpha=alpha,
        beta=beta,
        return_dict=True
    )


def compute_score_otc_grpo(solution_str: str,
                          ground_truth: Any,
                          correct_trajectories: List[str],
                          base_reward_fn: Optional[callable] = None,
                          c: float = 1.0,
                          alpha: float = 0.7,
                          beta: float = 0.3) -> Dict[str, Any]:
    """
    Convenience function for OTC-GRPO reward calculation.
    
    Args:
        solution_str: Solution text
        ground_truth: Ground truth
        correct_trajectories: List of correct solution strings
        base_reward_fn: Base reward function
        c: Smooth constant
        alpha: Base reward weight
        beta: Tool reward weight
        
    Returns:
        Dictionary with detailed reward information
    """
    return compute_otc_reward(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method="grpo",
        base_reward_fn=base_reward_fn,
        correct_trajectories=correct_trajectories,
        c=c,
        alpha=alpha,
        beta=beta,
        return_dict=True
    )


# Example usage and testing
if __name__ == "__main__":
    # Test tool call counting
    test_solution = """
    I need to search for information first.
    <search>What is the capital of France?</search>
    
    Let me also calculate something.
    <code>
    result = 2 + 2
    print(result)
    </code>
    
    The answer is Paris.
    """
    
    tool_count = count_tool_calls(test_solution)
    print(f"Tool calls found: {tool_count}")
    
    # Test OTC-PPO
    ppo_reward = compute_otc_ppo_reward(tool_count, c=1.0)
    print(f"OTC-PPO reward: {ppo_reward:.4f}")
    
    # Test OTC-GRPO
    grpo_reward = compute_otc_grpo_reward(tool_count, n=1, c=1.0)
    print(f"OTC-GRPO reward: {grpo_reward:.4f}")
    
    # Test full OTC reward
    result = compute_otc_reward(
        solution_str=test_solution,
        ground_truth="Paris",
        method="ppo",
        return_dict=True
    )
    print(f"Full OTC result: {result}") 