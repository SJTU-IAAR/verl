#!/usr/bin/env python3
"""
将TACO数据集（JSON格式）转换为VERL强化学习框架所需的Parquet格式。

此脚本执行以下操作：
1.  以流式方式读取源JSON文件，以高效处理大文件。
2.  为每个问题添加一个标准的指令性前缀，指导模型生成格式正确的代码。
3.  将每个JSON对象重构为RLHFDataset兼容的格式，包括：
    - `prompt`: 包含角色和内容的聊天格式列表。
    - `data_source`: 硬编码为 "taco"。
    - `extra_info`: 包含用于工具执行的 `tools_kwargs` (内含 `input_outputs`)
      以及其他元数据（如id, difficulty）。
4.  将转换后的数据保存为Parquet文件。
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import datasets
import ijson
from tqdm import tqdm

# 指导模型代码生成格式的指令前缀
PROMPT_PREFIX = """You are an expert programmer. Your task is to solve the given algorithmic problem.

Follow this structured approach:

1. **Think through the problem**: Wrap your reasoning in <think></think> tags.

2. **Develop your solution iteratively**:
   - Use <code></code> tags to provide a complete Python function that you want to test and get feedback on.
   - After each code submission, you will receive execution results in <execution_results></execution_results> tags.
   - You can submit multiple <code></code> blocks to iteratively improve your solution based on the test feedback.

3. **Provide your final solution**: When you are confident in your solution, put it in <answer></answer> tags to indicate completion.

**Important formatting rules**:
- All tags must appear in pairs: <think></think>, <code></code>, <answer></answer>
- Every function you provide (whether in <code> or <answer>) must be complete and follow these rules:
  * It must be a single, complete Python function definition
  * It must accept a single string argument containing the test case input
  * It must return a string result which is the solution for the test case  
  * Do not include any code outside the function definition (no test cases, prints, etc.)

Here is the problem description:
---
"""

def transform_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单个TACO JSON对象转换为VERL RL数据集格式的字典。
    """
    question = item.get("question")
    input_output = item.get("input_output")

    if not question or not input_output:
        return None

    # 构建完整的prompt
    full_prompt_content = f"{PROMPT_PREFIX}\n{question}"

    # 构建RL数据集行
    new_row = {
        "prompt": [{"role": "user", "content": full_prompt_content}],
        "data_source": "taco",
        "reward_model": {
            "style": "rule",  # TACO uses rule-based scoring
            "ground_truth": input_output  # The input/output test cases for evaluation
        },
        "extra_info": {
            "id": item.get("id"),
            "difficulty": item.get("difficulty"),
            "source": item.get("source"),
            "tools_kwargs": {
                # ToolClient将使用此字段来获取测试用例
                "input_outputs": input_output
            }
        }
    }
    return new_row

def convert_taco_to_rl_format(source_path: Path, output_path: Path, batch_size: int = 1000):
    """
    读取taco_verified.json，将其转换为taco_rl.parquet。
    使用分批处理避免内存溢出。
    """
    if not source_path.exists():
        print(f"❌ Error: Source file not found at {source_path}")
        return

    print(f"🚀 Starting conversion from '{source_path.name}' to '{output_path.name}'...")
    print(f"📦 Batch size: {batch_size} items per batch")

    processed_count = 0
    skipped_count = 0
    batch_count = 0
    current_batch = []

    # 创建临时目录存储批次文件
    temp_dir = output_path.parent / "temp_batches"
    temp_dir.mkdir(exist_ok=True)
    
    batch_files = []

    try:
        with source_path.open('rb') as f:
            print("⏳ Parsing JSON file and processing in batches...")
            # 假设JSON的顶层是一个对象数组
            parser = ijson.items(f, 'item')
            
            for item in tqdm(parser, desc="Processing TACO items"):
                transformed = transform_item(item)
                if transformed:
                    current_batch.append(transformed)
                    processed_count += 1
                else:
                    skipped_count += 1
                
                # 当达到批次大小时，保存当前批次
                if len(current_batch) >= batch_size:
                    batch_file = temp_dir / f"batch_{batch_count:04d}.parquet"
                    batch_files.append(batch_file)
                    
                    print(f"💾 Saving batch {batch_count + 1} ({len(current_batch)} items) to {batch_file.name}")
                    hf_batch = datasets.Dataset.from_list(current_batch)
                    hf_batch.to_parquet(batch_file)
                    
                    # 清空当前批次释放内存
                    current_batch = []
                    batch_count += 1
                    
                    # 强制垃圾回收
                    import gc
                    gc.collect()
            
            # 处理最后一个不完整的批次
            if current_batch:
                batch_file = temp_dir / f"batch_{batch_count:04d}.parquet"
                batch_files.append(batch_file)
                
                print(f"💾 Saving final batch {batch_count + 1} ({len(current_batch)} items) to {batch_file.name}")
                hf_batch = datasets.Dataset.from_list(current_batch)
                hf_batch.to_parquet(batch_file)
                batch_count += 1
                
    except ijson.JSONError as e:
        print(f"❌ A fatal error occurred during JSON parsing: {e}")
        print("   The file might be corrupted or not a valid JSON array.")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return

    if not batch_files:
        print("🤷 No data was processed. The output file will not be created.")
        return

    print(f"\n✅ Successfully processed {processed_count} items in {batch_count} batches.")
    if skipped_count > 0:
        print(f"⚠️ Skipped {skipped_count} items due to missing 'question' or 'input_output' fields.")

    # 合并所有批次文件
    print(f"🔄 Merging {len(batch_files)} batch files into final dataset...")
    try:
        # 逐个读取批次文件并合并
        all_datasets = []
        for i, batch_file in enumerate(batch_files):
            print(f"📖 Loading batch {i + 1}/{len(batch_files)}: {batch_file.name}")
            batch_dataset = datasets.load_dataset('parquet', data_files=str(batch_file), split='train')
            all_datasets.append(batch_dataset)
        
        print("🔗 Concatenating all batches...")
        final_dataset = datasets.concatenate_datasets(all_datasets)
        
        print(f"💾 Saving final dataset to: {output_path}")
        final_dataset.to_parquet(output_path)
        
        print("🧹 Cleaning up temporary files...")
        for batch_file in batch_files:
            batch_file.unlink()
        temp_dir.rmdir()
        
        print("\n🎉 Conversion complete!")
        print(f"Output file: {output_path}")
        print(f"Final dataset size: {len(final_dataset)} items")
        
    except Exception as e:
        print(f"❌ Failed to merge batches or save final file: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数，处理命令行参数并启动转换。"""
    parser = argparse.ArgumentParser(
        description="Convert TACO JSON dataset to Parquet format for VERL RL training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source-file",
        type=str,
        default="/mnt/chensiheng/ai_researcher_data/TACO-verified/taco_verified.json",
        help="Path to the source TACO JSON file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/mnt/chensiheng/shuotang/git_repo_h200/AI-Researcher/verl/rl_data_process/taco_dataset/result/taco_rl.parquet",
        help="Path to save the output Parquet file."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of items to process per batch (reduce if running out of memory)."
    )
    args = parser.parse_args()

    source_path = Path(args.source_file)
    output_path = Path(args.output_file)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    convert_taco_to_rl_format(source_path, output_path, args.batch_size)

if __name__ == "__main__":
    main() 