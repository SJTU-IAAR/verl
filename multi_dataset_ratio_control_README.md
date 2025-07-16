# Verl多数据集比例控制使用指南

## 概述

本文档介绍如何在verl训练框架中实现严格的多数据集比例控制，确保每个batch或step中各数据源的样本比例保持一致。

## 核心特性

- ✅ **严格比例控制**：基于WeightedRandomSampler实现的方案2
- ✅ **实时监控**：批次级别的比例分析和统计
- ✅ **配置驱动**：通过ppo_trainer.yaml简单配置
- ✅ **训练专用**：只对训练集进行比例控制，验证集保持原有逻辑
- ✅ **兼容性好**：与现有verl pipeline完全兼容

## 核心文件结构

```
verl/
├── custom_weighted_sampler.py          # 自定义采样器实现
├── multi_dataset_config_example.yaml   # 配置示例文件
├── multi_dataset_ratio_control_README.md # 本文档
├── verl/trainer/main_ppo.py            # 修改：create_rl_sampler函数
├── verl/trainer/ppo/ray_trainer.py     # 修改：批次分析集成
└── verl/trainer/config/ppo_trainer.yaml # 修改：添加多数据集配置
```

## 使用方法

### 1. 配置文件设置

在 `ppo_trainer.yaml` 中添加多数据集配置：

```yaml
data:
  train_files: 
    - ~/data/dataset_A.parquet
    - ~/data/dataset_B.parquet  
    - ~/data/dataset_C.parquet
  
  # 多数据集比例控制配置
  multi_dataset_sampling:
    enable: True  # 启用比例控制
    sampler_type: "weighted"  # 采样器类型
    
    # 目标比例（必须和为1.0）
    dataset_ratios:
      dataset_A: 0.5  # 50%
      dataset_B: 0.3  # 30% 
      dataset_C: 0.2  # 20%
    
    tolerance: 0.1  # 偏差容忍度
    log_batch_stats: True  # 启用统计记录
    log_frequency: 100  # 每100批次记录一次
```

### 2. 数据集要求

每个parquet文件必须包含 `data_source` 字段，标识数据来源：

```python
# 示例数据格式
{
    "prompt": "...",
    "data_source": "dataset_A",  # 必须字段
    # 其他字段...
}
```

### 3. 训练启动

正常启动训练，系统会自动应用比例控制：

```bash
python -m verl.trainer.main_ppo --config-name=your_config.yaml
```

## 技术实现

### 核心组件

#### 1. MultiDatasetWeightedSampler

```python
class MultiDatasetWeightedSampler(WeightedRandomSampler):
    """
    基于PyTorch WeightedRandomSampler的多数据集比例控制采样器
    
    工作原理：
    1. 统计各数据源的实际样本数量
    2. 计算权重：目标比例 / 实际比例
    3. 为每个样本分配相应权重
    4. 使用加权随机采样
    """
```

#### 2. DeterministicRatioSampler

```python
class DeterministicRatioSampler:
    """
    确定性比例采样器 - 严格保证比例
    
    工作原理：
    1. 按数据源分组样本索引
    2. 严格按比例分配每个batch的样本数
    3. 轮询各数据源获取样本
    4. 确保每个batch比例完全精确
    """
```

#### 3. BatchRatioAnalyzer

```python
class BatchRatioAnalyzer:
    """
    批次比例分析器 - 实时监控比例效果
    
    功能：
    1. 分析每个batch的实际比例
    2. 计算与目标比例的偏差
    3. 生成统计摘要报告
    4. 提供历史趋势分析
    """
```

### 集成点

#### 1. create_rl_sampler函数修改

```python
def create_rl_sampler(data_config, dataset, is_training=True):
    # 检查是否启用多数据集采样（仅训练时）
    multi_dataset_config = data_config.get("multi_dataset_sampling", {})
    enable_multi_dataset = multi_dataset_config.get("enable", False) and is_training
    
    if enable_multi_dataset:
        # 使用自定义采样器
        return create_multi_dataset_sampler(...)
    else:
        # 使用标准采样器
        return standard_sampler(...)
```

#### 2. RayPPOTrainer集成

```python
class RayPPOTrainer:
    def __init__(self):
        # 初始化批次分析器
        if multi_dataset_config.get("enable", False):
            self.batch_analyzer = BatchRatioAnalyzer(...)
    
    def fit(self):
        for batch_dict in self.train_dataloader:
            # 分析批次比例
            if self.batch_analyzer:
                batch_stats = self.batch_analyzer.analyze_batch(batch_dict)
```

## 监控和调试

### 1. 实时监控输出

训练过程中会看到如下输出：

```
[Step 100] Batch Ratio Analysis:
  dataset_A: Target 0.500, Actual 0.498, Deviation 0.0020
  dataset_B: Target 0.300, Actual 0.301, Deviation 0.0010  
  dataset_C: Target 0.200, Actual 0.201, Deviation 0.0010

============================================================
BATCH RATIO ANALYSIS SUMMARY
============================================================
Total analyzed batches: 500
Recent batches analyzed: 500

Target vs Actual Performance:
  dataset_A       | Target: 0.500 | Avg Dev: 0.0156 | Max Dev: 0.0430
  dataset_B       | Target: 0.300 | Avg Dev: 0.0142 | Max Dev: 0.0390
  dataset_C       | Target: 0.200 | Avg Dev: 0.0138 | Max Dev: 0.0410
============================================================
```

### 2. 调试建议

#### 如果比例偏差过大：

1. **检查数据源字段**：确保所有数据都有正确的`data_source`字段
2. **调整采样器类型**：尝试使用`"deterministic"`获得更严格的比例
3. **增大batch_size**：较大的batch更容易维持稳定比例
4. **检查数据分布**：确保各数据源有足够的样本

#### 性能优化：

1. **使用weighted采样器**：适合大多数场景，性能较好
2. **调整tolerance**：根据实际需求设置合理的偏差容忍度
3. **降低log_frequency**：减少统计输出频率以提升性能

## 比较分析：方案2 vs 其他方案

### 方案1：简单拼接（当前默认）
- ❌ 无比例控制
- ✅ 性能最好
- ❌ 比例随机波动

### 方案2：WeightedRandomSampler（本实现）
- ✅ 较好的比例控制
- ✅ 兼容性好
- ⚠️ 仍有小幅随机波动
- ✅ 性能较好

### 方案3：Oversample + Filter
- ✅ 严格比例控制
- ❌ 计算开销大
- ❌ 浪费采样样本
- ⚠️ 实现复杂

### 方案4：确定性构造
- ✅ 完全精确比例
- ❌ 复杂度高
- ⚠️ 可能影响随机性

## 最佳实践

### 1. 比例设计原则

- **基于数据量**：参考各数据集的原始大小
- **基于重要性**：重要任务分配更高比例
- **基于难度**：困难数据可适当增加比例
- **总和为1**：确保所有比例加起来等于1.0

### 2. 监控策略

- **训练初期**：密切监控比例是否符合预期
- **定期检查**：每隔一定步数检查统计摘要
- **异常处理**：设置合理的tolerance值进行预警

### 3. 数据准备

- **统一格式**：确保所有数据集使用相同的字段格式
- **data_source字段**：为每个数据集分配唯一的数据源标识
- **数据清洗**：移除异常或不完整的数据

## 故障排除

### 常见问题

#### 1. "数据源信息未找到"警告
```
WARNING: No data_source information found in batch
```
**解决方案**：检查数据集是否包含`data_source`字段

#### 2. "比例和不为1.0"警告  
```
WARNING: dataset ratios sum to 0.8, not 1.0. Normalizing...
```
**解决方案**：调整配置文件中的`dataset_ratios`使其和为1.0

#### 3. "导入采样器失败"错误
```
ERROR: Failed to import multi-dataset sampler
```
**解决方案**：确保`custom_weighted_sampler.py`在正确路径且无语法错误

#### 4. 比例偏差持续过大
**可能原因**：
- 某个数据源样本数量太少
- batch_size相对于目标比例太小
- 数据分布不均匀

**解决方案**：
- 增大batch_size
- 重新平衡数据集大小
- 考虑使用deterministic采样器

## 扩展和定制

### 添加新的采样策略

1. 在`custom_weighted_sampler.py`中实现新的采样器类
2. 在`create_multi_dataset_sampler`函数中添加新的sampler_type分支
3. 在配置文件中设置相应的sampler_type

### 自定义统计指标

1. 继承`BatchRatioAnalyzer`类
2. 重写`analyze_batch`或`get_summary_stats`方法
3. 添加所需的自定义统计逻辑

## 总结

verl的多数据集比例控制方案提供了：

- **易用性**：通过配置文件简单启用
- **可靠性**：严格控制训练批次比例
- **监控性**：实时统计和分析工具
- **兼容性**：与现有训练流程无缝集成

这确保了多数据集训练的稳定性和可预测性，是进行高质量多任务学习的重要基础。 