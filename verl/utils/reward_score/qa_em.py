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

import re
import string
import random

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """提取答案，支持搜索和非搜索场景，以及boxed格式"""
    # 首先尝试查找<answer>标签
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if answer_matches:
        # 使用最后一个<answer>标签（对多轮交互有用）
        return answer_matches[-1].group(1).strip()
    
    # 查找boxed格式：boxed{answer}或\boxed{answer}
    # 改进正则表达式以处理反复boxed的情况
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
        inner_boxed = extract_solution(final_answer)
        if inner_boxed is not None:
            return inner_boxed
        else:
            return final_answer.strip()
    
    # 如果没有找到标签，尝试其他启发式方法
    lines = solution_str.strip().split('\n')
    if lines:
        return lines[-1].strip()
    
    return None


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1., return_dict=False):
    """统一的EM打分函数，适用于普通QA和搜索QA任务"""
    answer = extract_solution(solution_str=solution_str)
    # 10% 随机打印完整数据信息
    do_print = random.random() < 0.1
    
    if do_print:
        print(f"======== REWARD CALCULATION DEBUG (10% sample) ========")
        if isinstance(ground_truth, dict) and 'target' in ground_truth:
            print(f"Golden answers: {ground_truth['target']}")
        else:
            print(f"Golden answers: {ground_truth}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string (first 500 chars): {solution_str[:500]}...")
        if len(solution_str) > 500:
            print(f"Solution string (last 200 chars): ...{solution_str[-200:]}")
        else:
            print(f"Full solution string: {solution_str}")
    
    if answer is None:
        result = 0
    else:
        # 处理不同格式的ground_truth
        if isinstance(ground_truth, dict) and 'target' in ground_truth:
            targets = ground_truth['target']
            result = score if em_check(answer, targets) else format_score
        elif isinstance(ground_truth, (list, tuple)):
            result = score if em_check(answer, ground_truth) else format_score
        else:
            result = score if em_check(answer, ground_truth) else format_score
    
    if do_print:
        print(f"Final score: {result}")
        print(f"Score type: {type(result)}")
        print(f"========================================================")
    
    # 如果需要返回字典格式
    if return_dict:
        return {
            "score": result,
            "extracted_answer": answer if answer is not None else "",
        }
    
    return result


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    # 10% 随机打印完整数据信息
    do_print = random.random() < 0.1
    
    if do_print:
        print(f"======== SUBEM REWARD CALCULATION DEBUG (10% sample) ========")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string (first 500 chars): {solution_str[:500]}...")
        if len(solution_str) > 500:
            print(f"Solution string (last 200 chars): ...{solution_str[-200:]}")
        else:
            print(f"Full solution string: {solution_str}")
    
    if answer is None:
        result = 0
    else:
        if subem_check(answer, ground_truth['target']):
            result = score
        else:
            result = format_score

    if do_print:
        print(f"Final score: {result}")
        print(f"Score type: {type(result)}")
        print(f"==============================================================")

    return result

# 添加一个可选的带过程奖励的计分函数
def compute_score_with_process(solution_str, ground_truth, method='strict', search_bonus=0.3, answer_bonus=0.7):
    """带有过程奖励的评分函数，返回字典格式的详细信息"""
    # 10% 随机打印完整数据信息
    do_print = random.random() < 0.1
    
    # 检查搜索行为
    search_pattern = r'<search>(.*?)</search>'
    search_count = len(re.findall(search_pattern, solution_str))
    has_search = search_count > 0
    
    # 检查信息获取
    info_pattern = r'<information>(.*?)</information>'
    info_count = len(re.findall(info_pattern, solution_str))
    has_info = info_count > 0
    
    # 检查是否有最终答案
    answer_pattern = r'<answer>(.*?)</answer>'
    has_answer = bool(re.search(answer_pattern, solution_str))
    
    # 计算基本分数（获取字典结果）
    answer_result = compute_score_em(solution_str, ground_truth, method=method, return_dict=True)
    base_score = answer_result["score"]
    extracted_answer = answer_result["extracted_answer"]
    
    # 计算过程分数
    process_score = 0.0
    if has_search:
        process_score += 0.2
    if has_info:
        process_score += 0.3
    if has_answer:
        process_score += 0.2
    
    # 总分 = 过程分数 * 过程权重 + 基本分数 * 答案权重
    total_score = (process_score * search_bonus) + (base_score * answer_bonus)
    
    if do_print:
        print(f"======== SEARCH REWARD CALCULATION DEBUG (10% sample) ========")
        if isinstance(ground_truth, dict) and 'target' in ground_truth:
            print(f"Golden answers: {ground_truth['target']}")
        else:
            print(f"Golden answers: {ground_truth}")
        print(f"Extracted answer: {extracted_answer}")
        print(f"Search count: {search_count}, Has search: {has_search}")
        print(f"Info count: {info_count}, Has info: {has_info}")
        print(f"Has answer tag: {has_answer}")
        print(f"Process score: {process_score}")
        print(f"Base score: {base_score}")
        print(f"Search bonus weight: {search_bonus}, Answer bonus weight: {answer_bonus}")
        print(f"Total score: {total_score}")
        print(f"Solution string (first 500 chars): {solution_str[:500]}...")
        if len(solution_str) > 500:
            print(f"Solution string (last 200 chars): ...{solution_str[-200:]}")
        else:
            print(f"Full solution string: {solution_str}")
        print(f"===============================================================")
    
    # 返回字典格式，包含更详细的信息
    return {
        "score": total_score,                 # 总分数
        "answer_score": base_score,           # 答案得分
        "process_score": process_score,       # 过程得分
        "has_search": int(has_search),        # 是否包含搜索
        "has_info": int(has_info),            # 是否包含信息
        "has_answer": int(has_answer),        # 是否包含答案标记
        "search_count": search_count,         # 搜索次数
        "info_count": info_count,             # 信息块数量
        "extracted_answer": extracted_answer  # 提取的答案
    }