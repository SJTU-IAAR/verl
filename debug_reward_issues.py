#!/usr/bin/env python3
"""
Debug script for reward calculation issues in xhpang_search and numina_math datasets.

Issues found:
1. XHPang Search: extract_solution extracts "</execution_results>" instead of actual answer
2. Numina Math: ground_truth is empty string

This script reproduces the issues and provides fixes.
"""

import re
import random


def extract_solution_original(solution_str):
    """Original extract_solution function with issues"""
    # 首先尝试查找<answer>标签
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if answer_matches:
        # 使用最后一个<answer>标签（对多轮交互有用）
        solution_str = answer_matches[-1].group(1).strip()

    # 查找boxed格式：boxed{answer}或\boxed{answer}
    boxed_pattern = r'\\?boxed\{'
    boxed_matches = []
    
    # 查找所有boxed开始位置
    start_pos = 0
    while True:
        match = re.search(boxed_pattern, solution_str[start_pos:])
        if not match:
            break
        
        # 找到boxed开始位置
        abs_start = start_pos + match.end()
        
        # 手动匹配大括号来处理嵌套
        brace_count = 1
        current_pos = abs_start
        
        while current_pos < len(solution_str) and brace_count > 0:
            if solution_str[current_pos] == '{':
                brace_count += 1
            elif solution_str[current_pos] == '}':
                brace_count -= 1
            current_pos += 1
        
        if brace_count == 0:
            # 成功匹配到完整的boxed内容
            content = solution_str[abs_start:current_pos-1]
            boxed_matches.append(content)
        
        start_pos = abs_start
    
    if boxed_matches:
        # 对于反复boxed，递归提取最内层内容
        final_answer = boxed_matches[-1]
        
        # 如果内容本身还包含boxed，递归提取
        inner_boxed = extract_solution_original(final_answer)
        if inner_boxed is not None:
            return inner_boxed
        else:
            return final_answer.strip()
    
    # 如果没有找到标签，尝试其他启发式方法
    lines = solution_str.strip().split('\n')
    if lines:
        return lines[-1].strip()  # 这里是问题所在！
    
    return None


def extract_solution_fixed(solution_str):
    """Fixed extract_solution function"""
    original_solution_str = solution_str
    
    # 首先尝试查找<answer>标签
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if answer_matches:
        # 使用最后一个<answer>标签（对多轮交互有用）
        extracted_content = answer_matches[-1].group(1).strip()
        if extracted_content:  # 确保内容不为空
            return extracted_content

    # 查找boxed格式：boxed{answer}或\boxed{answer}
    boxed_pattern = r'\\?boxed\{'
    boxed_matches = []
    
    # 查找所有boxed开始位置
    start_pos = 0
    while True:
        match = re.search(boxed_pattern, solution_str[start_pos:])
        if not match:
            break
        
        # 找到boxed开始位置
        abs_start = start_pos + match.end()
        
        # 手动匹配大括号来处理嵌套
        brace_count = 1
        current_pos = abs_start
        
        while current_pos < len(solution_str) and brace_count > 0:
            if solution_str[current_pos] == '{':
                brace_count += 1
            elif solution_str[current_pos] == '}':
                brace_count -= 1
            current_pos += 1
        
        if brace_count == 0:
            # 成功匹配到完整的boxed内容
            content = solution_str[abs_start:current_pos-1]
            if content.strip():  # 确保内容不为空
                boxed_matches.append(content)
        
        start_pos = abs_start
    
    if boxed_matches:
        # 对于反复boxed，递归提取最内层内容
        final_answer = boxed_matches[-1]
        
        # 如果内容本身还包含boxed，递归提取
        inner_boxed = extract_solution_fixed(final_answer)
        if inner_boxed is not None:
            return inner_boxed
        else:
            return final_answer.strip()
    
    # 改进的回退逻辑：避免提取标签或空行
    lines = solution_str.strip().split('\n')
    
    # 从后往前查找有效的答案行
    for line in reversed(lines):
        line = line.strip()
        # 跳过空行和标签行
        if (line and 
            not line.startswith('<') and 
            not line.endswith('>') and
            not line.startswith('```') and
            line not in ['</execution_results>', '</search>', '</code>', '</answer>']):
            return line
    
    # 如果所有行都无效，返回None
    return None


def test_xhpang_search_issue():
    """Test case that reproduces the XHPang Search issue"""
    print("=== Testing XHPang Search Issue ===")
    
    # 模拟有问题的response
    problematic_response = """I need to find information about the capital of France.

<search>capital of France</search>

Let me also check with code:

<code>
country = "France"
query = f"What is the capital of {country}?"
result = search_web(query)
print(result)
</code>

<execution_results>
The capital of France is Paris. Paris is the largest city and capital of France.
</execution_results>"""

    print("Problematic response:")
    print(problematic_response)
    print("\n" + "="*50)
    
    # 测试原始函数（有问题）
    original_result = extract_solution_original(problematic_response)
    print(f"Original function result: '{original_result}'")
    
    # 测试修复后的函数
    fixed_result = extract_solution_fixed(problematic_response)
    print(f"Fixed function result: '{fixed_result}'")
    
    print("\n")


def test_numina_math_ground_truth():
    """Test for empty ground_truth issue"""
    print("=== Testing Numina Math Ground Truth Issue ===")
    
    # 模拟数据加载情况
    sample_data = {
        "prompt": "What is 2 + 2?",
        "data_source": "numina_math",
        "reward_model": {
            "ground_truth": ""  # 这是问题所在
        }
    }
    
    print("Sample data with empty ground_truth:")
    print(sample_data)
    
    ground_truth = sample_data["reward_model"]["ground_truth"]
    print(f"Ground truth: '{ground_truth}' (length: {len(ground_truth)})")
    
    # 检查可能的原因
    if not ground_truth:
        print("❌ Ground truth is empty!")
        print("Possible causes:")
        print("1. Data preprocessing removed ground_truth")
        print("2. Original data file has missing ground_truth fields")
        print("3. Schema standardization failed")
        print("4. Data loading configuration issues")
    
    print("\n")


def test_comprehensive_cases():
    """Test comprehensive cases with both fixed and original functions"""
    print("=== Comprehensive Testing ===")
    
    test_cases = [
        {
            "name": "Normal answer with tag",
            "text": "The answer is: <answer>Paris</answer>",
            "expected": "Paris"
        },
        {
            "name": "Boxed answer",
            "text": r"The result is \boxed{42}",
            "expected": "42"
        },
        {
            "name": "Tool use with execution but no answer tag",
            "text": """<code>print("Hello")</code>
<execution_results>
Hello
</execution_results>""",
            "expected": None  # Should not extract tag
        },
        {
            "name": "Tool use ending with execution tag",
            "text": """Let me calculate:
<code>result = 2 + 2</code>
<execution_results>
4
</execution_results>""",
            "expected": "4"  # Should extract the actual result
        },
        {
            "name": "Mixed content with final answer",
            "text": """<search>query</search>
Based on search results:
The capital is Paris.
<answer>Paris</answer>""",
            "expected": "Paris"
        }
    ]
    
    for case in test_cases:
        print(f"Test: {case['name']}")
        print(f"Expected: {case['expected']}")
        
        original = extract_solution_original(case['text'])
        fixed = extract_solution_fixed(case['text'])
        
        print(f"Original: '{original}' {'✅' if original == case['expected'] else '❌'}")
        print(f"Fixed:    '{fixed}' {'✅' if fixed == case['expected'] else '❌'}")
        print("-" * 40)


def main():
    """Main test function"""
    print("Debugging Reward Calculation Issues")
    print("=" * 50)
    
    test_xhpang_search_issue()
    test_numina_math_ground_truth()
    test_comprehensive_cases()
    
    print("\n=== Summary and Recommendations ===")
    print("1. XHPang Search Issue:")
    print("   - Problem: extract_solution uses last line as fallback")
    print("   - Solution: Improve fallback logic to skip tags and empty lines")
    
    print("\n2. Numina Math Ground Truth Issue:")
    print("   - Problem: Empty ground_truth in data")
    print("   - Solution: Check data preprocessing and loading pipeline")
    
    print("\n3. Recommended Fixes:")
    print("   - Update extract_solution function with improved fallback logic")
    print("   - Add data validation to catch empty ground_truth")
    print("   - Add debugging output to identify data issues early")


if __name__ == "__main__":
    main() 