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

def convert_parquet_to_json(parquet_file, output_json_file):
    # 读取parquet文件
    df = pd.read_parquet(parquet_file)
    
    # 创建JSON数据列表
    json_data = []
    
    # 遍历每一行数据
    for i, row in tqdm(df.iterrows(), total=len(df)):
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
# 使用示例
if __name__ == "__main__":
    parquet_file = "/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search/test.parquet"  # 替换为您的parquet文件路径
    output_json_file = "test.json"
    
    convert_parquet_to_json(parquet_file, output_json_file)