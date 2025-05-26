import pandas as pd
import json
import os
from tqdm import tqdm

# 旧的前缀（用于识别）
OLD_PREFIX = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>"""

# 新的前缀（用于识别）
NEW_PREFIX_START = """
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can utilize special tools to help you."""

def analyze_prefixes(original_parquet, converted_parquet):
    """分析原始和转换后的parquet文件，验证前缀替换情况"""
    # 读取两个parquet文件
    print(f"读取原始文件: {original_parquet}")
    original_df = pd.read_parquet(original_parquet)
    
    print(f"读取转换后文件: {converted_parquet}")
    converted_df = pd.read_parquet(converted_parquet)
    
    # 检查基本信息
    print(f"\n原始数据行数: {len(original_df)}")
    print(f"转换后数据行数: {len(converted_df)}")
    
    # 统计计数器
    old_prefix_count = 0
    new_prefix_count = 0
    neither_prefix_count = 0
    
    # 样本展示
    old_prefix_sample = None
    new_prefix_sample = None
    
    # 检查每行数据
    print("\n开始分析前缀替换情况...")
    for i in tqdm(range(len(converted_df))):
        # 处理原始数据
        if i < len(original_df):
            orig_prompt = original_df.iloc[i].get('prompt', '')
            
            # 检查原始prompt是列表且包含role/content结构
            if isinstance(orig_prompt, list) and len(orig_prompt) > 0:
                for prompt_item in orig_prompt:
                    if isinstance(prompt_item, dict) and 'content' in prompt_item:
                        content = prompt_item['content']
                        if isinstance(content, str) and content.startswith(OLD_PREFIX):
                            old_prefix_count += 1
                            if old_prefix_sample is None:
                                old_prefix_sample = (i, content[:200])
                            break
            # 处理字符串情况
            elif isinstance(orig_prompt, str) and orig_prompt.startswith(OLD_PREFIX):
                old_prefix_count += 1
                if old_prefix_sample is None:
                    old_prefix_sample = (i, orig_prompt[:200])
        
        # 处理转换后数据
        conv_prompt = converted_df.iloc[i].get('prompt', '')
        has_new_prefix = False
        has_old_prefix = False
        
        # 检查转换后prompt是列表且包含role/content结构
        if isinstance(conv_prompt, list) and len(conv_prompt) > 0:
            for prompt_item in conv_prompt:
                if isinstance(prompt_item, dict) and 'content' in prompt_item:
                    content = prompt_item['content']
                    if isinstance(content, str):
                        if content.startswith(NEW_PREFIX_START):
                            has_new_prefix = True
                            if new_prefix_sample is None:
                                new_prefix_sample = (i, content[:200])
                            break
                        elif content.startswith(OLD_PREFIX):
                            has_old_prefix = True
        # 处理字符串情况
        elif isinstance(conv_prompt, str):
            if conv_prompt.startswith(NEW_PREFIX_START):
                has_new_prefix = True
                if new_prefix_sample is None:
                    new_prefix_sample = (i, conv_prompt[:200])
            elif conv_prompt.startswith(OLD_PREFIX):
                has_old_prefix = True
        
        # 统计
        if has_new_prefix:
            new_prefix_count += 1
        elif not has_old_prefix:
            neither_prefix_count += 1
    
    # 结果报告
    print("\n前缀分析结果:")
    print(f"原始文件中有旧前缀的条目: {old_prefix_count} ({old_prefix_count/len(original_df)*100:.2f}%)")
    print(f"转换后文件中有新前缀的条目: {new_prefix_count} ({new_prefix_count/len(converted_df)*100:.2f}%)")
    print(f"转换后文件中既没有新前缀也没有旧前缀的条目: {neither_prefix_count} ({neither_prefix_count/len(converted_df)*100:.2f}%)")
    
    if new_prefix_count > 0 and old_prefix_count > 0:
        success_rate = new_prefix_count / old_prefix_count * 100
        print(f"\n前缀替换成功率: {success_rate:.2f}%")
    elif new_prefix_count > 0:
        print("\n前缀替换成功")
    else:
        print("\n前缀替换失败，没有找到新前缀！")
    
    # 显示样本
    if old_prefix_sample:
        print(f"\n原始前缀样本 (行 {old_prefix_sample[0]}):")
        print(old_prefix_sample[1])
        
    if new_prefix_sample:
        print(f"\n新前缀样本 (行 {new_prefix_sample[0]}):")
        print(new_prefix_sample[1])

def check_json_files(original_json, modified_json):
    """检查JSON文件中的前缀替换情况"""
    # 读取两个JSON文件
    print(f"读取原始JSON: {original_json}")
    with open(original_json, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"读取修改后JSON: {modified_json}")
    with open(modified_json, 'r', encoding='utf-8') as f:
        modified_data = json.load(f)
    
    # 检查基本信息
    print(f"\nJSON文件条目数: 原始={len(original_data)}, 修改后={len(modified_data)}")
    
    # 统计计数器
    old_prefix_count = 0
    new_prefix_count = 0
    
    # 检查原始JSON
    print("\n检查原始JSON中的前缀...")
    for item in tqdm(original_data):
        if 'prompt' in item:
            prompt = item['prompt']
            # 处理prompt为列表的情况，列表中包含role和content
            if isinstance(prompt, list):
                for prompt_item in prompt:
                    if isinstance(prompt_item, dict) and 'content' in prompt_item:
                        content = prompt_item['content']
                        if isinstance(content, str) and content.startswith(OLD_PREFIX):
                            old_prefix_count += 1
                            break
            # 处理prompt为字符串的情况
            elif isinstance(prompt, str) and prompt.startswith(OLD_PREFIX):
                old_prefix_count += 1
    
    # 检查修改后JSON
    print("\n检查修改后JSON中的前缀...")
    for item in tqdm(modified_data):
        if 'prompt' in item:
            prompt = item['prompt']
            # 处理prompt为列表的情况，列表中包含role和content
            if isinstance(prompt, list):
                for prompt_item in prompt:
                    if isinstance(prompt_item, dict) and 'content' in prompt_item:
                        content = prompt_item['content']
                        if isinstance(content, str) and content.startswith(NEW_PREFIX_START):
                            new_prefix_count += 1
                            break
            # 处理prompt为字符串的情况
            elif isinstance(prompt, str) and prompt.startswith(NEW_PREFIX_START):
                new_prefix_count += 1
    
    # 结果报告
    print("\nJSON文件前缀分析结果:")
    print(f"原始JSON中有旧前缀的条目: {old_prefix_count} ({old_prefix_count/len(original_data)*100:.2f}%)")
    print(f"修改后JSON中有新前缀的条目: {new_prefix_count} ({new_prefix_count/len(modified_data)*100:.2f}%)")
    
    if new_prefix_count > 0 and old_prefix_count > 0:
        success_rate = new_prefix_count / old_prefix_count * 100
        print(f"\nJSON前缀替换成功率: {success_rate:.2f}%")
    elif new_prefix_count > 0:
        print("\nJSON前缀替换成功")
    else:
        print("\nJSON前缀替换可能失败，没有找到新前缀！")

if __name__ == "__main__":
    # 数据目录
    output_dir = "/home/ma-user/modelarts/work/weiyu/xueban_v2/AI-Researcher_Dev/AI-Researcher_dev/verl/dataset/tool_data"
    
    # 测试数据文件
    print("\n========== 验证测试数据 ==========")
    original_test = "/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search/test.parquet"
    converted_test = os.path.join(output_dir, "test_new.parquet")
    
    # 中间JSON文件 - 测试数据
    original_json_test = os.path.join(output_dir, "test_original.json")
    modified_json_test = os.path.join(output_dir, "test_modified.json")
    
    # 检查测试文件是否存在
    if os.path.exists(original_test) and os.path.exists(converted_test):
        print("===== 验证测试parquet文件 =====")
        analyze_prefixes(original_test, converted_test)
    else:
        print("无法找到测试parquet文件进行验证")
    
    # 检查测试JSON文件是否存在
    if os.path.exists(original_json_test) and os.path.exists(modified_json_test):
        print("\n===== 验证测试JSON文件 =====")
        check_json_files(original_json_test, modified_json_test)
    else:
        print("无法找到测试JSON文件进行验证")
    
    # 验证训练数据
    print("\n\n========== 验证训练数据 ==========")
    original_train = "/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search/train.parquet"
    converted_train = os.path.join(output_dir, "train_new.parquet")
    
    # 中间JSON文件 - 训练数据
    original_json_train = os.path.join(output_dir, "train_original.json")
    modified_json_train = os.path.join(output_dir, "train_modified.json")
    
    if os.path.exists(original_train) and os.path.exists(converted_train):
        print("===== 验证训练parquet文件 =====")
        analyze_prefixes(original_train, converted_train)
    else:
        print("无法找到训练parquet文件进行验证")
    
    # 检查训练JSON文件是否存在
    if os.path.exists(original_json_train) and os.path.exists(modified_json_train):
        print("\n===== 验证训练JSON文件 =====")
        check_json_files(original_json_train, modified_json_train)
    else:
        print("无法找到训练JSON文件进行验证") 