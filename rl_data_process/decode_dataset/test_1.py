import pandas as pd
import os

# 定义文件路径
train_path = "/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search/train.parquet"
test_path = "/home/ma-user/modelarts/work/jjw/Search-R1/data/nq_search/test.parquet"

# 读取Parquet文件
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# 查看基本信息
print("训练集形状:", train_df.shape)
print("测试集形状:", test_df.shape)

# 显示列名
print("\n数据集列名:")
print(train_df.columns.tolist())

# 显示训练集的一个样本
print("\n训练集样本:")
print(train_df.iloc[0])

# 如果存在嵌套结构，查看prompt字段的内容
if 'prompt' in train_df.columns:
    print("\nprompt字段的第一个样本:")
    print(train_df['prompt'].iloc[0])

# 如果有extra_info字段，查看其内容
if 'extra_info' in train_df.columns:
    print("\nextra_info字段的第一个样本:")
    print(train_df['extra_info'].iloc[0])