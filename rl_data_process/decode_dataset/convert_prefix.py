import pandas as pd
import os
import json
import argparse
import re
from tqdm import tqdm

# 新的前缀
NEW_PREFIX = """
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can utilize special tools to help you.

You can execute Python code by writing it inside <tool> tags:

<tool>
import math
from tool import search_r1
radius = 5
area = math.pi * radius ** 2
print(f"The area of a circle with radius {radius} is {area:.2f}")

# You can also use the search_r1 function to search for information
result = search_r1(queries=["What is the capital of France?"])
print(result)
</tool>

The code execution results will be returned to you. The search_r1 function allows you to search for information online.

When you have all the information you need, provide your final answer inside <answer> and </answer> tags without detailed illustrations. For example: <answer> Beijing </answer>"""

# 旧的前缀的关键部分，用于更灵活地匹配
OLD_PREFIX_PATTERN = r"Answer the given question\. You must conduct reasoning inside <think> and </think>.*?you can call a search engine by <search>.*?</search>.*?provide the answer inside <answer>.*?</answer>"

def process_dataset(input_path, output_path):
    """处理数据集，替换前缀"""
    print(f"读取数据集: {input_path}")
    df = pd.read_parquet(input_path)
    
    modified_count = 0
    total_samples = len(df)
    not_modified_examples = []
    
    # 将DataFrame转换为字典列表
    data_list = df.to_dict('records')
    
    # 处理每个记录
    for idx, record in tqdm(enumerate(data_list), total=total_samples, desc="处理数据"):
        # 获取并处理prompt字段
        prompt_list = record['prompt']
        modified = False
        
        for i, prompt_item in enumerate(prompt_list):
            if 'content' in prompt_item:
                content = prompt_item['content']
                
                # 尝试使用正则表达式匹配旧前缀模式
                if re.search(OLD_PREFIX_PATTERN, content, re.DOTALL):
                    # 找到问题的起始位置
                    question_start = content.find("Question:")
                    if question_start == -1:
                        # 如果没有找到"Question:"标记，尝试找最后一个句号后的内容作为问题
                        last_sentence = content.split('.')[-1].strip()
                        if last_sentence:
                            # 构建新的内容：新前缀 + 问题
                            new_content = NEW_PREFIX + "\n\nQuestion: " + last_sentence
                            prompt_list[i]['content'] = new_content
                            modified = True
                    else:
                        # 构建新的内容：新前缀 + 问题部分
                        question_part = content[question_start:]
                        new_content = NEW_PREFIX + "\n\n" + question_part
                        prompt_list[i]['content'] = new_content
                        modified = True
        
        if modified:
            modified_count += 1
        else:
            # 记录未修改的样本示例（最多5个）
            if len(not_modified_examples) < 5:
                sample_id = record.get('id', f"index_{idx}")
                sample_content = prompt_list[0]['content'][:200] + "..." if prompt_list and 'content' in prompt_list[0] else "无内容"
                not_modified_examples.append((sample_id, sample_content))
    
    # 使用JSON作为中间格式保存数据
    json_path = output_path.replace('.parquet', '.json')
    print(f"保存JSON格式中间数据: {json_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    # 从JSON加载并保存为Parquet
    print(f"从JSON加载并保存为Parquet: {output_path}")
    new_df = pd.read_json(json_path, orient='records')
    new_df.to_parquet(output_path)
    
    # 删除中间JSON文件
    if os.path.exists(json_path) and os.path.exists(output_path):
        os.remove(json_path)
        print(f"已删除中间JSON文件: {json_path}")
    
    print(f"总样本数: {total_samples}")
    print(f"已修改样本数: {modified_count}")
    print(f"修改比例: {modified_count/total_samples*100:.2f}%")
    
    if not_modified_examples:
        print("\n未修改的样本示例:")
        for i, (sample_id, content) in enumerate(not_modified_examples):
            print(f"样本 {i+1}, ID: {sample_id}")
            print(f"Content: {content}")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="替换数据集中的搜索前缀")
    parser.add_argument("--input_dir", type=str, default="/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search", help="输入数据集目录")
    parser.add_argument("--output_dir", type=str, default="/home/ma-user/modelarts/work/weiyu/xueban_v2/AI-Researcher_Dev/AI-Researcher_dev/verl/dataset/tool_data", help="输出数据集目录")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理训练集
    train_input = os.path.join(args.input_dir, "train.parquet")
    train_output = os.path.join(args.output_dir, "train.parquet")
    if os.path.exists(train_input):
        process_dataset(train_input, train_output)
    
    # 处理测试集
    test_input = os.path.join(args.input_dir, "test.parquet")
    test_output = os.path.join(args.output_dir, "test.parquet")
    if os.path.exists(test_input):
        process_dataset(test_input, test_output)

if __name__ == "__main__":
    main() 