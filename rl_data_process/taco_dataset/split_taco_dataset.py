#!/usr/bin/env python3
"""
TACO数据集拆分脚本 - 使用VERL官方方式
基于Hugging Face datasets的原生split功能拆分数据集，与VERL官方示例保持一致
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset
import json

def split_taco_dataset(input_file: str, output_dir: str = "./", train_ratio: float = 0.8):
    """
    使用VERL官方方式拆分TACO数据集
    
    Args:
        input_file: 输入的Parquet文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例 (0.0 - 1.0)
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    # 确保输入文件存在
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📖 读取数据集: {input_path}")
    
    # 使用datasets加载Parquet文件
    try:
        dataset = load_dataset('parquet', data_files=str(input_path), split='train')
        total_size = len(dataset)
        print(f"✅ 成功读取 {total_size} 条数据")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 计算拆分比例
    train_percentage = int(train_ratio * 100)
    test_percentage = 100 - train_percentage
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    
    print(f"📊 数据集拆分信息:")
    print(f"   总数据量: {total_size}")
    print(f"   训练集: {train_size} ({train_ratio:.1%})")
    print(f"   测试集: {test_size} ({1-train_ratio:.1%})")
    
    # 使用VERL官方方式进行拆分
    print(f"🔀 开始拆分数据集...")
    
    try:
        # 使用datasets的split功能，类似官方示例
        train_dataset = load_dataset('parquet', data_files=str(input_path), split=f'train[:{train_percentage}%]')
        test_dataset = load_dataset('parquet', data_files=str(input_path), split=f'train[-{test_percentage}%:]')
        
        print(f"✅ 拆分完成")
        print(f"   训练集实际大小: {len(train_dataset)}")
        print(f"   测试集实际大小: {len(test_dataset)}")
        
    except Exception as e:
        print(f"❌ 数据拆分失败: {e}")
        return
    
    # 生成输出文件名
    base_name = input_path.stem  # 不包含扩展名的文件名
    train_file = output_path / f"{base_name}_train.parquet"
    test_file = output_path / f"{base_name}_test.parquet"
    
    # 保存训练集
    print(f"💾 保存训练集: {train_file}")
    try:
        train_dataset.to_parquet(str(train_file))
        print(f"✅ 训练集保存成功")
    except Exception as e:
        print(f"❌ 训练集保存失败: {e}")
        return
    
    # 保存测试集
    print(f"💾 保存测试集: {test_file}")
    try:
        test_dataset.to_parquet(str(test_file))
        print(f"✅ 测试集保存成功")
    except Exception as e:
        print(f"❌ 测试集保存失败: {e}")
        return
    
    # 保存拆分信息
    split_info = {
        "input_file": str(input_path),
        "total_samples": total_size,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "train_ratio": train_ratio,
        "test_ratio": 1 - train_ratio,
        "train_file": str(train_file),
        "test_file": str(test_file),
        "split_method": "huggingface_datasets_percentage",
        "note": "使用VERL官方方式进行数据拆分"
    }
    
    info_file = output_path / f"{base_name}_split_info.json"
    print(f"📄 保存拆分信息: {info_file}")
    try:
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        print(f"✅ 拆分信息保存成功")
    except Exception as e:
        print(f"⚠️ 拆分信息保存失败: {e}")
    
    print(f"\n🎉 数据集拆分完成!")
    print(f"   训练集: {train_file}")
    print(f"   测试集: {test_file}")
    print(f"   信息文件: {info_file}")

def validate_split(train_file: str, test_file: str):
    """
    验证拆分结果的正确性
    
    Args:
        train_file: 训练集文件路径
        test_file: 测试集文件路径
    """
    print(f"\n🔍 验证拆分结果...")
    
    try:
        # 使用datasets读取拆分后的文件
        train_dataset = load_dataset('parquet', data_files=train_file, split='train')
        test_dataset = load_dataset('parquet', data_files=test_file, split='train')
        
        print(f"✅ 验证通过:")
        print(f"   训练集大小: {len(train_dataset)}")
        print(f"   测试集大小: {len(test_dataset)}")
        print(f"   总大小: {len(train_dataset) + len(test_dataset)}")
        
        # 检查列是否一致
        train_features = list(train_dataset.features.keys())
        test_features = list(test_dataset.features.keys())
        
        if train_features == test_features:
            print(f"✅ 列结构一致: {len(train_features)} 列")
            print(f"   列名: {train_features}")
        else:
            print(f"⚠️ 列结构不一致")
            print(f"   训练集列: {train_features}")
            print(f"   测试集列: {test_features}")
        
        # 显示部分示例数据
        print(f"\n📋 训练集前3行示例:")
        for i in range(min(3, len(train_dataset))):
            example = train_dataset[i]
            print(f"   样本 {i+1}:")
            print(f"     data_source: {example.get('data_source', 'N/A')}")
            if 'extra_info' in example and example['extra_info']:
                extra_info = example['extra_info']
                print(f"     id: {extra_info.get('id', 'N/A')}")
                print(f"     difficulty: {extra_info.get('difficulty', 'N/A')}")
            print()
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用VERL官方方式将TACO数据集按比例拆分为训练集和测试集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-file", 
        type=str, 
        default="/mnt/chensiheng/shuotang/git_repo_h200/AI-Researcher/verl/rl_data_process/taco_dataset/result/taco_rl.parquet",
        help="输入的TACO Parquet文件路径"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/mnt/chensiheng/shuotang/git_repo_h200/AI-Researcher/verl/rl_data_process/taco_dataset/result/",
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--train-ratio", 
        type=float, 
        default=0.8,
        help="训练集比例 (0.0-1.0)"
    )
    
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="拆分后验证结果"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not (0.0 < args.train_ratio < 1.0):
        print(f"❌ 错误: 训练集比例必须在 0.0 和 1.0 之间，当前值: {args.train_ratio}")
        return
    
    # 执行拆分
    try:
        split_taco_dataset(
            input_file=args.input_file,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio
        )
        
        # 可选验证
        if args.validate:
            base_name = Path(args.input_file).stem
            output_path = Path(args.output_dir)
            train_file = output_path / f"{base_name}_train.parquet"
            test_file = output_path / f"{base_name}_test.parquet"
            validate_split(str(train_file), str(test_file))
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()