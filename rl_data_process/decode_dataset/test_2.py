import pandas as pd
import re
import json
from tqdm import tqdm  # 用于显示进度条

# 定义文件路径
train_path = "/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search/train.parquet"
test_path = "/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search/test.parquet"

# 要检查的前缀
expected_prefix = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>"""

def analyze_prompts(dataframe, name="dataset"):
    """分析DataFrame中的prompt字段，检查前缀"""
    
    # 计数器
    total_samples = len(dataframe)
    contains_prefix = 0
    missing_prefix = 0
    different_prefix = 0
    
    # 存储不包含前缀的样本ID和具有不同前缀的样本
    missing_prefix_ids = []
    different_prefix_samples = []
    
    # 遍历所有样本
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"分析{name}"):
        try:
            # 获取prompt内容
            prompt_content = row['prompt'][0]['content']
            
            # 检查前缀是否包含在prompt中（考虑到可能有轻微的空格或换行差异）
            # 清理文本以进行比较（删除额外的空格和换行符）
            cleaned_prefix = re.sub(r'\s+', ' ', expected_prefix).strip()
            cleaned_prompt_start = re.sub(r'\s+', ' ', prompt_content[:len(expected_prefix) + 100]).strip()
            
            if cleaned_prefix in cleaned_prompt_start:
                contains_prefix += 1
            else:
                # 检查是否是完全不同的前缀，还是只是格式不同
                # 我们使用一个简单的启发式方法：检查几个关键短语
                key_phrases = ["conduct reasoning inside <think>", "call a search engine by <search>", "provide the answer inside <answer>"]
                
                if all(phrase in prompt_content for phrase in key_phrases):
                    different_prefix += 1
                    if len(different_prefix_samples) < 5:  # 只存储5个示例
                        different_prefix_samples.append({
                            'id': row.get('id', f"index_{idx}"),
                            'prompt_start': prompt_content[:200] + "..."  # 只显示前200个字符
                        })
                else:
                    missing_prefix += 1
                    missing_prefix_ids.append(row.get('id', f"index_{idx}"))
        except (IndexError, KeyError, TypeError) as e:
            # 处理可能的错误（例如，如果prompt字段格式不符合预期）
            missing_prefix += 1
            missing_prefix_ids.append(f"{row.get('id', f'index_{idx}')} (错误: {str(e)})")
    
    # 返回分析结果
    return {
        "总样本数": total_samples,
        "包含预期前缀的样本数": contains_prefix,
        "包含前缀但格式不同的样本数": different_prefix,
        "完全缺少前缀的样本数": missing_prefix,
        "前缀完整匹配率": contains_prefix / total_samples * 100 if total_samples > 0 else 0,
        "总体包含指令率": (contains_prefix + different_prefix) / total_samples * 100 if total_samples > 0 else 0,
        "缺少前缀的样本ID (最多10个)": missing_prefix_ids[:10],
        "格式不同的前缀示例 (最多5个)": different_prefix_samples
    }

def main():
    # 读取数据集
    print("正在读取训练集...")
    train_df = pd.read_parquet(train_path)
    
    print("正在读取测试集...")
    test_df = pd.read_parquet(test_path)
    
    # 分析训练集
    print("\n开始分析训练集...")
    train_results = analyze_prompts(train_df, "训练集")
    
    # 分析测试集
    print("\n开始分析测试集...")
    test_results = analyze_prompts(test_df, "测试集")
    
    # 打印详细结果
    print("\n=== 训练集分析结果 ===")
    for key, value in train_results.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    print(f"  ID: {item['id']}")
                    print(f"  前缀: {item['prompt_start']}")
                    print("  " + "-" * 50)
                else:
                    print(f"  {item}")
        else:
            if isinstance(value, float):
                print(f"{key}: {value:.2f}%")
            else:
                print(f"{key}: {value}")
    
    print("\n=== 测试集分析结果 ===")
    for key, value in test_results.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    print(f"  ID: {item['id']}")
                    print(f"  前缀: {item['prompt_start']}")
                    print("  " + "-" * 50)
                else:
                    print(f"  {item}")
        else:
            if isinstance(value, float):
                print(f"{key}: {value:.2f}%")
            else:
                print(f"{key}: {value}")
    
    # 如果需要，将详细结果保存到文件
    print("\n是否要将完整分析结果保存到文件? (y/n)")
    print("如果需要保存，请在执行代码后输入'y'")
    print("保存路径将为: './prompt_analysis_results.json'")

if __name__ == "__main__":
    main()