import pandas as pd
import json
import numpy as np
from tqdm import tqdm

def convert_json_to_parquet(json_file, original_parquet, output_parquet_file):
    # 读取原始parquet文件以获取完整结构
    original_df = pd.read_parquet(original_parquet)
    
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 创建列表存储转换后的数据
    rows = []
    
    # 遍历每个JSON条目
    for i, item in tqdm(enumerate(json_data), desc="转换JSON到Parquet"):
        # 获取原始行数据（如果i超出范围则使用第一行）
        orig_idx = min(i, len(original_df)-1)
        orig_row = original_df.iloc[orig_idx].to_dict()
        
        # 创建基本结构，从原始数据复制结构
        row = {
            'id': f"test_{i}",  # 根据索引生成ID
            'question': item.get('question', orig_row.get('question')),
            'golden_answers': item.get('golden_answers', orig_row.get('golden_answers')),
            'data_source': item.get('data_source', orig_row.get('data_source')),
            'prompt': item.get('prompt', orig_row.get('prompt')),
            'ability': item.get('ability', orig_row.get('ability')),
            'extra_info': {'index': i, 'split': 'test'}
        }
        
        # 处理reward_model字段 - 修复嵌套数组问题
        if 'reward_model' in item and isinstance(item['reward_model'], dict):
            if 'ground_truth' in item['reward_model'] and 'target' in item['reward_model']['ground_truth']:
                # 直接使用原始格式
                row['reward_model'] = item['reward_model']
            else:
                # 只有ground_truth但没有target
                row['reward_model'] = {
                    'ground_truth': {'target': orig_row['reward_model']['ground_truth']['target']},
                    'style': 'rule'
                }
        elif 'ground_truth' in item:
            # 使用item中的ground_truth，但保持与原始结构相似
            if isinstance(item['ground_truth'], dict) and 'target' in item['ground_truth']:
                # 已经有正确的结构
                row['reward_model'] = {
                    'ground_truth': item['ground_truth'],
                    'style': 'rule'
                }
            else:
                # 需要构建正确的结构
                row['reward_model'] = {
                    'ground_truth': {'target': orig_row['reward_model']['ground_truth']['target']},
                    'style': 'rule'
                }
        else:
            # 直接使用原始数据
            row['reward_model'] = {
                'ground_truth': {'target': orig_row['reward_model']['ground_truth']['target']},
                'style': 'rule'
            }
        
        rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(rows)
    
    # 将DataFrame转换为parquet文件
    df.to_parquet(output_parquet_file, index=False)
    
    print(f"转换完成！保存到 {output_parquet_file}")
    print(f"共转换 {len(df)} 条数据")
    
    return df


if __name__ == "__main__":
    json_file = "/home/ma-user/modelarts/work/weiyu/xueban_v2/AI-Researcher_Dev/AI-Researcher_dev/verl/data_process/decode_dataset/test.json"  # 替换为您的JSON文件路径
    output_parquet_file = "/home/ma-user/modelarts/work/weiyu/xueban_v2/AI-Researcher_Dev/AI-Researcher_dev/verl/data_process/decode_dataset/test_converted.parquet"
    original_parquet = "/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search/test.parquet"
    convert_json_to_parquet(json_file, original_parquet, output_parquet_file)