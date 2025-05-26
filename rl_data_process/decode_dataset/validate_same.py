import pandas as pd
import numpy as np
from tqdm import tqdm

def compare_parquet_files(original_parquet, converted_parquet):
    # 读取两个parquet文件
    original_df = pd.read_parquet(original_parquet)
    converted_df = pd.read_parquet(converted_parquet)
    
    # 检查基本结构
    print(f"原始数据形状: {original_df.shape}")
    print(f"转换后数据形状: {converted_df.shape}")
    
    # 检查列是否匹配
    original_columns = set(original_df.columns)
    converted_columns = set(converted_df.columns)
    
    print("\n列比较:")
    print(f"原始列: {sorted(original_columns)}")
    print(f"转换后列: {sorted(converted_columns)}")
    print(f"缺失的列: {original_columns - converted_columns}")
    print(f"新增的列: {converted_columns - original_columns}")
    
    # 只比较两个DataFrame都有的列
    common_columns = list(original_columns.intersection(converted_columns))
    
    # 逐行比较关键字段
    print("\n正在比较数据内容...")
    
    # 初始化计数器
    total_rows = min(len(original_df), len(converted_df))
    different_rows = 0
    field_differences = {col: 0 for col in common_columns}
    
    # 示例查看前5行的差异
    examples = []
    
    for i in tqdm(range(total_rows), desc="比较行"):
        row_different = False
        row_diffs = {}
        
        for col in common_columns:
            # 获取原始值和转换后的值
            orig_val = original_df.iloc[i][col]
            conv_val = converted_df.iloc[i][col]
            
            # 对于某些复杂类型，可能需要特殊处理
            are_equal = compare_values(orig_val, conv_val)
            
            if not are_equal:
                row_different = True
                field_differences[col] += 1
                if len(examples) < 5:
                    row_diffs[col] = {
                        'original': orig_val,
                        'converted': conv_val
                    }
        
        if row_different:
            different_rows += 1
            if row_diffs and len(examples) < 5:
                examples.append({
                    'row_index': i,
                    'differences': row_diffs
                })
    
    # 输出统计信息
    print(f"\n总共比较: {total_rows} 行")
    print(f"不同的行: {different_rows} ({different_rows/total_rows*100:.2f}%)")
    
    print("\n字段差异统计:")
    for col in common_columns:
        if field_differences[col] > 0:
            print(f"{col}: {field_differences[col]} 行有差异 ({field_differences[col]/total_rows*100:.2f}%)")
    
    # 输出示例差异
    if examples:
        print("\n示例差异:")
        for i, example in enumerate(examples):
            print(f"\n示例 {i+1}, 行 {example['row_index']}:")
            for col, diff in example['differences'].items():
                print(f"  字段 '{col}':")
                print(f"    原始: {diff['original']}")
                print(f"    转换: {diff['converted']}")
    
    # 总结
    if different_rows == 0:
        print("\n验证结果: 数据完全一致！")
    else:
        print("\n验证结果: 数据有差异，请检查上述报告。")

def compare_values(val1, val2):
    """比较两个值，处理特殊类型"""
    
    # 同为None情况
    if val1 is None and val2 is None:
        return True
    
    # 一个是None另一个不是
    if (val1 is None and val2 is not None) or (val1 is not None and val2 is None):
        return False
    
    # 处理NumPy数组
    if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
        return np.array_equal(val1, val2)
    
    # 处理列表 - 可能包含复杂对象
    if isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            return False
        return all(compare_values(v1, v2) for v1, v2 in zip(val1, val2))
    
    # 处理字典
    if isinstance(val1, dict) and isinstance(val2, dict):
        if set(val1.keys()) != set(val2.keys()):
            return False
        return all(compare_values(val1[k], val2[k]) for k in val1.keys())
    
    # 其他类型，直接比较
    try:
        return val1 == val2
    except:
        # 如果无法直接比较，将其转换为字符串比较
        return str(val1) == str(val2)

if __name__ == "__main__":
    original_parquet = "/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search/test.parquet"          # 原始parquet文件
    converted_parquet = "/home/ma-user/modelarts/work/weiyu/xueban_v2/AI-Researcher_Dev/AI-Researcher_dev/verl/data_process/decode_dataset/test_converted.parquet"  # 转换后的parquet文件
    
    compare_parquet_files(original_parquet, converted_parquet)