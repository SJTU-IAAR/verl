#!/usr/bin/env python3
"""
一个简洁的脚本，用于分析数据集。
- 显示3个原始数据样本
- 使用 ijson 流式解析JSON，高效处理大文件
- 分析 data_source 字段
"""

import sys
from pathlib import Path
import json
from collections import Counter
import ijson
import datasets

def analyze_dataset(file_path):
    """
    分析给定的JSON或Parquet数据集文件。
    """
    print(f"🔍 Analyzing: {file_path}\n")
    
    p = Path(file_path)
    if not p.exists():
        print(f"❌ File not found: {file_path}")
        return

    try:
        file_type = p.suffix.lstrip('.').lower()

        if file_type == 'json':
            print("🚀 Using ijson to stream-parse the JSON file.")
            
            # 1. 查看原始数据结果 (前3个)
            print("\n--- 🔬 3 Raw Data Samples ---")
            samples_to_show = 3
            top_samples = []
            with p.open('rb') as f: # ijson works on bytes
                # Assuming the JSON contains a list of objects at the top level
                parser = ijson.items(f, 'item')
                for i, sample in enumerate(parser):
                    if i >= samples_to_show:
                        break
                    top_samples.append(sample)
                    print(f"\n[Sample {i+1}]")
                    print(json.dumps(sample, indent=2, ensure_ascii=False))
            
            if not top_samples:
                print("Could not retrieve any samples. The dataset might be empty or not a JSON array.")
                return

            print("\n" + "="*50 + "\n")

            # 2. 查看数据源 (data_source)
            if 'data_source' in top_samples[0]:
                print("--- 🌍 Data Source Analysis (analyzing up to 10,000 rows) ---")
                source_counts = Counter()
                rows_analyzed = 0
                with p.open('rb') as f:
                    parser = ijson.items(f, 'item')
                    for i, sample in enumerate(parser):
                        if i >= 10000:
                            break
                        if 'data_source' in sample and sample['data_source'] is not None:
                            source_counts[sample['data_source']] += 1
                        rows_analyzed += 1

                print(f"Value counts for 'data_source' (from first {rows_analyzed} rows):")
                if not source_counts:
                     print("  - No 'data_source' found or values are null in the first 10,000 rows.")
                else:
                    for source, count in source_counts.most_common():
                        print(f"  - {source}: {count} rows")
            else:
                print("⚠️ 'data_source' column not found in the dataset.")
                print(f"Available columns in first sample: {list(top_samples[0].keys())}")

        elif file_type == 'parquet':
            print("🚀 Using 'datasets' library to stream the Parquet file.")
            dataset = datasets.load_dataset("parquet", data_files=file_path, split="train", streaming=True)

            # 1. 查看原始数据结果 (前3个)
            print("\n--- 🔬 3 Raw Data Samples ---")
            top_samples = list(dataset.take(3))

            for i, sample in enumerate(top_samples):
                print(f"\n[Sample {i+1}]")
                print(json.dumps(sample, indent=2, ensure_ascii=False))
            
            if not top_samples:
                print("Could not retrieve any samples. The dataset might be empty.")
                return
            
            print("\n" + "="*50 + "\n")

            # 2. 查看数据源 (data_source)
            if 'data_source' in top_samples[0]:
                print("--- 🌍 Data Source Analysis (analyzing up to 10,000 rows) ---")
                
                source_counts = Counter()
                rows_analyzed = 0
                for sample in dataset.take(10000):
                    if 'data_source' in sample and sample['data_source'] is not None:
                        source_counts[sample['data_source']] += 1
                    rows_analyzed += 1

                print(f"Value counts for 'data_source' (from first {rows_analyzed} rows):")
                if not source_counts:
                     print("  - No 'data_source' found or values are null in the first 10,000 rows.")
                else:
                    for source, count in source_counts.most_common():
                        print(f"  - {source}: {count} rows")
            else:
                print("⚠️ 'data_source' column not found in the dataset.")
                print(f"Available columns in first sample: {list(top_samples[0].keys())}")

        else:
             print(f"Unsupported file type for this script: '{file_type}'. Please use .json or .parquet")
             return

    except ijson.JSONError as e:
        print(f"❌ An error occurred during JSON parsing: {e}")
        print("   The file might not be a valid JSON array or is corrupted.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

def main():
    """主函数"""
    target_file = "/mnt/chensiheng/ai_researcher_data/numinamath_new/train_subset.parquet"
    print(f"Analyzing: {target_file}")
    analyze_dataset(target_file)

if __name__ == "__main__":
    main() 