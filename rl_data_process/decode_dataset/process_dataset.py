import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

# 自定义JSON编码器处理NumPy数组和其他非标准类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 将NumPy数组转换为Python列表
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        # 处理其他可能的类型
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)  # 最后的手段：将对象转换为字符串

# 旧的前缀
OLD_PREFIX = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>"""

# 新的前缀
NEW_PREFIX = """
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can utilize special tools to help you.

You can execute Python code by writing it inside <tool> tags:

<tool>
# You can perform normal calculations and data analysis with Python
import math
import numpy as np

# Example of normal calculation
radius = 5
area = math.pi * radius ** 2
print(f"The area of a circle with radius {radius} is {area:.2f}")

# Example of data analysis
data = [12, 15, 18, 22, 30, 35]
mean = np.mean(data)
std = np.std(data)
print(f"Mean: {mean}, Standard Deviation: {std}")

# IMPORTANT: To search for information online, you MUST import search_r1 first
from tools import search_r1

# Then you can use search_r1 to find information
search_result = search_r1(queries=["What is the capital of France?"])
print(search_result)
</tool>

The code execution results will be returned to you. You can use normal Python code for calculations and data analysis, but when you need to search for information, always remember to include the line 'from tools import search_r1' before using the search_r1 function.

When you have all the information you need, provide your final answer inside <answer> and </answer> tags without detailed illustrations. For example: <answer> Beijing </answer>"""

def convert_parquet_to_json(parquet_file, output_json_file):
    """步骤1: 将parquet文件转换为JSON"""
    # 读取parquet文件
    df = pd.read_parquet(parquet_file)
    
    # 创建JSON数据列表
    json_data = []
    
    # 遍历每一行数据
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"转换 {os.path.basename(parquet_file)} 到JSON"):
        # 将pandas Series转换为字典
        row_dict = row.to_dict()
        
        # 创建一个新的字典，只包含我们关心的字段
        item = {
            'prompt': row_dict['prompt'],
            'ground_truth': row_dict['reward_model'].get('ground_truth', {}) if isinstance(row_dict['reward_model'], dict) else row_dict['reward_model']
        }
        
        # 如果有question字段，添加它
        if 'question' in row_dict:
            item['question'] = row_dict['question']
        elif 'extra_info' in row_dict and isinstance(row_dict['extra_info'], dict):
            if 'question' in row_dict['extra_info']:
                item['question'] = row_dict['extra_info']['question']
            
        # 添加ability字段
        if 'ability' in row_dict:
            item['ability'] = row_dict['ability']
        
        json_data.append(item)
    
    # 写入JSON文件，使用自定义编码器
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print(f"转换完成！保存到 {output_json_file}")
    print(f"共转换 {len(json_data)} 条数据")
    
    return json_data

def replace_prefix_in_json(json_file, output_json_file):
    """步骤2: 在JSON文件中替换前缀"""
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 处理每个条目
    modified_count = 0
    for item in tqdm(json_data, desc="替换前缀"):
        if 'prompt' in item:
            # 处理prompt为列表的情况，列表中每个元素包含role和content
            if isinstance(item['prompt'], list):
                for prompt_item in item['prompt']:
                    if isinstance(prompt_item, dict) and 'content' in prompt_item:
                        content = prompt_item['content']
                        if isinstance(content, str) and content.startswith(OLD_PREFIX):
                            prompt_item['content'] = content.replace(OLD_PREFIX, NEW_PREFIX, 1)
                            modified_count += 1
            # 处理prompt为字符串的情况
            elif isinstance(item['prompt'], str) and item['prompt'].startswith(OLD_PREFIX):
                item['prompt'] = item['prompt'].replace(OLD_PREFIX, NEW_PREFIX, 1)
                modified_count += 1
    
    # 保存修改后的JSON
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print(f"替换完成！修改了 {modified_count} 条数据")
    print(f"保存到 {output_json_file}")
    
    return json_data

def convert_json_to_parquet(json_file, original_parquet, output_parquet_file):
    """步骤3: 将JSON转换回parquet"""
    # 读取原始parquet文件以获取完整结构
    original_df = pd.read_parquet(original_parquet)
    
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 创建列表存储转换后的数据
    rows = []
    
    # 遍历每个JSON条目
    for i, item in tqdm(enumerate(json_data), desc=f"转换JSON到Parquet ({os.path.basename(output_parquet_file)})"):
        # 获取原始行数据（如果i超出范围则使用第一行）
        orig_idx = min(i, len(original_df)-1)
        orig_row = original_df.iloc[orig_idx].to_dict()
        
        # 创建基本结构，从原始数据复制结构
        row = {
            'id': orig_row.get('id', f"test_{i}"),  # 保留原始ID或生成新ID
            'question': item.get('question', orig_row.get('question')),
            'golden_answers': orig_row.get('golden_answers'),
            'data_source': orig_row.get('data_source'),
            'prompt': item.get('prompt', orig_row.get('prompt')),
            'ability': item.get('ability', orig_row.get('ability')),
            'extra_info': orig_row.get('extra_info', {'index': i, 'split': 'test'})
        }
        
        # 处理reward_model字段
        if 'reward_model' in orig_row:
            if 'ground_truth' in item:
                # 使用item中的ground_truth，但保持与原始结构相似
                if isinstance(item['ground_truth'], dict) and 'target' in item['ground_truth']:
                    row['reward_model'] = {
                        'ground_truth': item['ground_truth'],
                        'style': orig_row['reward_model'].get('style', 'rule')
                    }
                else:
                    row['reward_model'] = {
                        'ground_truth': {'target': item['ground_truth']},
                        'style': orig_row['reward_model'].get('style', 'rule')
                    }
            else:
                # 使用原始数据的reward_model
                row['reward_model'] = orig_row['reward_model']
        
        rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(rows)
    
    # 将DataFrame转换为parquet文件
    df.to_parquet(output_parquet_file, index=False)
    
    print(f"转换完成！保存到 {output_parquet_file}")
    print(f"共转换 {len(df)} 条数据")
    
    return df

def process_dataset(parquet_file, output_dir):
    """处理一个数据集文件的完整流程：parquet->json->替换前缀->parquet"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(parquet_file))[0]
    
    # 步骤1: parquet -> json
    json_file = os.path.join(output_dir, f"{base_name}_original.json")
    convert_parquet_to_json(parquet_file, json_file)
    
    # 步骤2: 替换前缀
    modified_json_file = os.path.join(output_dir, f"{base_name}_modified.json")
    replace_prefix_in_json(json_file, modified_json_file)
    
    # 步骤3: json -> parquet
    output_parquet_file = os.path.join(output_dir, f"{base_name}_new.parquet")
    convert_json_to_parquet(modified_json_file, parquet_file, output_parquet_file)
    
    return output_parquet_file

if __name__ == "__main__":
    # 输入文件
    test_parquet = "/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search/test.parquet"
    train_parquet = "/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search/train.parquet"
    
    # 输出目录
    output_dir = "/home/ma-user/modelarts/work/weiyu/xueban_v2/AI-Researcher_Dev/AI-Researcher_dev/verl/dataset/tool_data"
    
    # 处理测试数据
    print("开始处理测试数据...")
    test_output = process_dataset(test_parquet, output_dir)
    
    # 处理训练数据
    print("\n开始处理训练数据...")
    train_output = process_dataset(train_parquet, output_dir)
    
    print("\n所有处理完成!")
    print(f"测试数据输出: {test_output}")
    print(f"训练数据输出: {train_output}") 