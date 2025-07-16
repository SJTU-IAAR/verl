# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom weighted sampler for multi-dataset training with controlled ratios.
"""

import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler, Sampler
from collections import Counter, defaultdict
from typing import Dict, List, Union


class MultiDatasetBatchSampler(Sampler):
    """
    批次采样器 - 保证每个batch内的数据源比例
    直接产生batch级别的索引列表，确保每个batch内严格按比例采样
    """
    
    def __init__(
        self,
        dataset,
        dataset_ratios: Dict[str, float],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        generator=None
    ):
        self.dataset = dataset
        self.dataset_ratios = dataset_ratios
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator or torch.Generator()
        
        # 验证比例之和为1
        total_ratio = sum(dataset_ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Dataset ratios sum to {total_ratio}, should be 1.0")
        
        # 按数据源分组索引
        self.data_source_indices = self._group_indices_by_source()
        
        # 计算每个batch中各数据源的样本数
        self.batch_composition = self._calculate_batch_composition()
        
        print(f"MultiDatasetBatchSampler initialized:")
        print(f"  Dataset ratios: {dataset_ratios}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batch composition: {self.batch_composition}")
        print(f"  Data source counts: {[(k, len(v)) for k, v in self.data_source_indices.items()]}")

    def _group_indices_by_source(self) -> Dict[str, List[int]]:
        """按数据源分组样本索引"""
        source_indices = defaultdict(list)
        
        # 直接访问原始dataframe避免触发完整的数据处理
        if hasattr(self.dataset, 'dataframe'):
            # 对于RLHFDataset，直接访问底层dataframe
            for idx in range(len(self.dataset.dataframe)):
                raw_item = self.dataset.dataframe[idx]
                data_source = raw_item.get('data_source', 'unknown')
                source_indices[data_source].append(idx)
        else:
            # 回退到标准方法
            for idx in range(len(self.dataset)):
                item = self.dataset[idx]
                data_source = item.get('data_source', 'unknown')
                source_indices[data_source].append(idx)
        
        return dict(source_indices)

    def _calculate_batch_composition(self) -> Dict[str, int]:
        """计算每个batch中各数据源应该有多少样本"""
        composition = {}
        total_assigned = 0
        
        # 按比例分配，向下取整
        for source, ratio in self.dataset_ratios.items():
            if source in self.data_source_indices:
                count = int(ratio * self.batch_size)
                composition[source] = count
                total_assigned += count
        
        # 处理舍入误差，将剩余样本按比例分配
        remaining = self.batch_size - total_assigned
        if remaining > 0:
            # 按比例的小数部分排序，给剩余名额
            fractional_parts = []
            for source, ratio in self.dataset_ratios.items():
                if source in self.data_source_indices:
                    fractional = (ratio * self.batch_size) % 1
                    fractional_parts.append((fractional, source))
            
            # 按小数部分降序排列，优先分配给小数部分大的
            fractional_parts.sort(reverse=True)
            for i in range(remaining):
                if i < len(fractional_parts):
                    _, source = fractional_parts[i]
                    composition[source] += 1
        
        return composition

    def __iter__(self):
        """产生每个batch的索引列表"""
        # 为每个数据源创建shuffled索引列表
        source_iterators = {}
        for source, indices in self.data_source_indices.items():
            if self.shuffle:
                # 打乱每个数据源内部的顺序
                perm = torch.randperm(len(indices), generator=self.generator)
                shuffled_indices = [indices[i] for i in perm]
                # 重复多次避免耗尽
                source_iterators[source] = iter(shuffled_indices * 1000)
            else:
                source_iterators[source] = iter(indices * 1000)
        
        # 计算总的batch数量
        num_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            num_batches += 1
        
        for batch_idx in range(num_batches):
            batch_indices = []
            
            # 最后一个batch可能不完整
            if batch_idx == num_batches - 1 and not self.drop_last:
                remaining_samples = len(self.dataset) - batch_idx * self.batch_size
                if remaining_samples < self.batch_size:
                    # 按比例缩放最后一个batch
                    scaled_composition = {}
                    total_assigned = 0
                    for source, count in self.batch_composition.items():
                        scaled_count = int(count * remaining_samples / self.batch_size)
                        scaled_composition[source] = scaled_count
                        total_assigned += scaled_count
                    
                    # 分配剩余样本
                    remaining = remaining_samples - total_assigned
                    sources = list(scaled_composition.keys())
                    for i in range(remaining):
                        if sources:
                            source = sources[i % len(sources)]
                            scaled_composition[source] += 1
                    
                    composition = scaled_composition
                else:
                    composition = self.batch_composition
            else:
                composition = self.batch_composition
            
            # 从每个数据源按指定数量采样
            for source, count in composition.items():
                if source in source_iterators and count > 0:
                    for _ in range(count):
                        try:
                            idx = next(source_iterators[source])
                            batch_indices.append(idx)
                        except StopIteration:
                            # 重新开始迭代
                            indices = self.data_source_indices[source]
                            if self.shuffle:
                                perm = torch.randperm(len(indices), generator=self.generator)
                                shuffled_indices = [indices[i] for i in perm]
                                source_iterators[source] = iter(shuffled_indices * 1000)
                            else:
                                source_iterators[source] = iter(indices * 1000)
                            idx = next(source_iterators[source])
                            batch_indices.append(idx)
            
            # 打乱batch内部的顺序（保持比例但随机排列）
            if self.shuffle and len(batch_indices) > 1:
                perm = torch.randperm(len(batch_indices), generator=self.generator)
                batch_indices = [batch_indices[i] for i in perm]
            
            yield batch_indices

    def __len__(self):
        """返回总的batch数量"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class MultiDatasetWeightedSampler(WeightedRandomSampler):
    """
    自定义的加权随机采样器，支持多数据集比例控制
    
    Args:
        dataset: 包含data_source信息的数据集
        dataset_ratios: 各数据源的目标比例 {"dataset_A": 0.5, "dataset_B": 0.3, "dataset_C": 0.2}
        batch_size: 批次大小
        generator: 随机数生成器
        replacement: 是否允许重复采样
    """
    
    def __init__(
        self, 
        dataset, 
        dataset_ratios: Dict[str, float],
        generator=None,
        replacement: bool = True
    ):
        self.dataset = dataset
        self.dataset_ratios = dataset_ratios
        
        # 验证比例之和为1
        total_ratio = sum(dataset_ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Dataset ratios sum to {total_ratio}, should be 1.0")
        
        # 统计各数据源的样本数量
        self.data_source_counts = self._count_data_sources()
        
        # 计算每个样本的权重
        sample_weights = self._compute_sample_weights()
        
        # 初始化父类
        super().__init__(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=replacement,
            generator=generator
        )
    
    def _count_data_sources(self) -> Dict[str, int]:
        """统计各数据源的样本数量"""
        data_source_counts = Counter()
        
        # 直接访问原始dataframe避免触发完整的数据处理
        if hasattr(self.dataset, 'dataframe'):
            # 对于RLHFDataset，直接访问底层dataframe
            for i in range(len(self.dataset.dataframe)):
                raw_item = self.dataset.dataframe[i]
                data_source = raw_item.get('data_source', 'unknown')
                data_source_counts[data_source] += 1
        else:
            # 回退到标准方法
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                data_source = item.get('data_source', 'unknown')
                data_source_counts[data_source] += 1
        
        print(f"Data source distribution in dataset: {dict(data_source_counts)}")
        print(f"Expected data sources from ratios: {list(self.dataset_ratios.keys())}")
        
        # 检查是否有目标数据源在数据集中不存在
        missing_sources = set(self.dataset_ratios.keys()) - set(data_source_counts.keys())
        if missing_sources:
            print(f"WARNING: Following data sources in ratios not found in dataset: {missing_sources}")
        
        # 检查是否有数据集中的数据源在目标比例中不存在
        extra_sources = set(data_source_counts.keys()) - set(self.dataset_ratios.keys())
        if extra_sources:
            print(f"WARNING: Following data sources in dataset not configured in ratios: {extra_sources}")
        
        return dict(data_source_counts)
    
    def _compute_sample_weights(self) -> torch.Tensor:
        """计算每个样本的权重"""
        sample_weights = []
        
        # 直接访问原始dataframe避免触发完整的数据处理
        if hasattr(self.dataset, 'dataframe'):
            # 对于RLHFDataset，直接访问底层dataframe
            for i in range(len(self.dataset.dataframe)):
                raw_item = self.dataset.dataframe[i]
                data_source = raw_item.get('data_source', 'unknown')
                
                # 目标比例 / 实际比例 = 权重调节因子
                target_ratio = self.dataset_ratios.get(data_source, 0.0)
                actual_count = self.data_source_counts.get(data_source, 1)
                actual_ratio = actual_count / len(self.dataset)
                
                if actual_ratio > 0:
                    weight = target_ratio / actual_ratio
                else:
                    weight = 0.0
                
                sample_weights.append(weight)
        else:
            # 回退到标准方法
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                data_source = item.get('data_source', 'unknown')
                
                # 目标比例 / 实际比例 = 权重调节因子
                target_ratio = self.dataset_ratios.get(data_source, 0.0)
                actual_count = self.data_source_counts.get(data_source, 1)
                actual_ratio = actual_count / len(self.dataset)
                
                if actual_ratio > 0:
                    weight = target_ratio / actual_ratio
                else:
                    weight = 0.0
                
                sample_weights.append(weight)
        
        sample_weights = torch.tensor(sample_weights, dtype=torch.float)
        
        # 打印权重分布信息
        unique_weights = {}
        for data_source, ratio in self.dataset_ratios.items():
            if data_source in self.data_source_counts:
                actual_ratio = self.data_source_counts[data_source] / len(self.dataset)
                weight = ratio / actual_ratio if actual_ratio > 0 else 0.0
                unique_weights[data_source] = weight
        
        print(f"Sample weights by data source: {unique_weights}")
        return sample_weights
    
    def get_expected_batch_distribution(self) -> Dict[str, float]:
        """返回期望的batch分布"""
        return self.dataset_ratios.copy()
    
    def validate_batch_distribution(self, batch_data_sources: List[str], tolerance: float = 0.1) -> Dict[str, float]:
        """验证实际batch的分布是否符合期望"""
        actual_counts = Counter(batch_data_sources)
        actual_ratios = {source: count/len(batch_data_sources) for source, count in actual_counts.items()}
        
        deviations = {}
        for source, expected_ratio in self.dataset_ratios.items():
            actual_ratio = actual_ratios.get(source, 0.0)
            deviation = abs(actual_ratio - expected_ratio)
            deviations[source] = deviation
            
            if deviation > tolerance:
                print(f"WARNING: {source} ratio deviation: expected {expected_ratio:.3f}, actual {actual_ratio:.3f}, deviation {deviation:.3f}")
        
        return {
            'expected': self.dataset_ratios,
            'actual': actual_ratios,
            'deviations': deviations
        }


class DeterministicRatioSampler:
    """
    确定性比例采样器 - 严格保证每个batch的比例
    适用于对比例要求极其严格的场景
    """
    
    def __init__(
        self,
        dataset,
        dataset_ratios: Dict[str, float],
        batch_size: int,
        shuffle: bool = True,
        generator=None
    ):
        self.dataset = dataset
        self.dataset_ratios = dataset_ratios
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator or torch.Generator()
        
        # 按数据源分组索引
        self.data_source_indices = self._group_indices_by_source()
        
        # 计算每个batch中各数据源的样本数
        self.batch_composition = self._calculate_batch_composition()
        
        # 创建轮询迭代器
        self.source_iterators = {}
        self._reset_iterators()
    
    def _group_indices_by_source(self) -> Dict[str, List[int]]:
        """按数据源分组样本索引"""
        source_indices = defaultdict(list)
        
        # 直接访问原始dataframe避免触发完整的数据处理
        if hasattr(self.dataset, 'dataframe'):
            # 对于RLHFDataset，直接访问底层dataframe
            for idx in range(len(self.dataset.dataframe)):
                raw_item = self.dataset.dataframe[idx]
                data_source = raw_item.get('data_source', 'unknown')
                source_indices[data_source].append(idx)
        else:
            # 回退到标准方法
            for idx in range(len(self.dataset)):
                item = self.dataset[idx]
                data_source = item.get('data_source', 'unknown')
                source_indices[data_source].append(idx)
        
        print(f"Indices grouped by source: {[(k, len(v)) for k, v in source_indices.items()]}")
        return dict(source_indices)
    
    def _calculate_batch_composition(self) -> Dict[str, int]:
        """计算每个batch中各数据源应该有多少样本"""
        composition = {}
        remaining_batch_size = self.batch_size
        
        # 按比例分配，确保总和等于batch_size
        for source, ratio in self.dataset_ratios.items():
            if source in self.data_source_indices:
                count = int(ratio * self.batch_size)
                composition[source] = count
                remaining_batch_size -= count
        
        # 处理舍入误差，将剩余样本分配给最大的数据源
        if remaining_batch_size > 0:
            largest_source = max(self.dataset_ratios.keys(), key=lambda k: self.dataset_ratios[k])
            if largest_source in composition:
                composition[largest_source] += remaining_batch_size
        
        print(f"Batch composition: {composition} (total: {sum(composition.values())})")
        return composition
    
    def _reset_iterators(self):
        """重置所有数据源的迭代器"""
        for source, indices in self.data_source_indices.items():
            if self.shuffle:
                # 打乱每个数据源内部的顺序
                shuffled_indices = indices.copy()
                torch.manual_seed(self.generator.initial_seed())
                torch.randperm(len(shuffled_indices), generator=self.generator)
                perm = torch.randperm(len(indices), generator=self.generator)
                shuffled_indices = [indices[i] for i in perm]
                self.source_iterators[source] = iter(shuffled_indices * 1000)  # 重复多次避免耗尽
            else:
                self.source_iterators[source] = iter(indices * 1000)
    
    def __iter__(self):
        self._reset_iterators()
        
        num_batches = len(self.dataset) // self.batch_size
        
        for _ in range(num_batches):
            batch_indices = []
            
            # 严格按比例构造batch
            for source, count in self.batch_composition.items():
                if source in self.source_iterators:
                    for _ in range(count):
                        try:
                            idx = next(self.source_iterators[source])
                            batch_indices.append(idx)
                        except StopIteration:
                            # 重新开始迭代
                            self._reset_iterators()
                            idx = next(self.source_iterators[source])
                            batch_indices.append(idx)
            
            # 打乱batch内部的顺序
            if self.shuffle:
                perm = torch.randperm(len(batch_indices), generator=self.generator)
                batch_indices = [batch_indices[i] for i in perm]
            
            yield batch_indices
    
    def __len__(self):
        return len(self.dataset) // self.batch_size


def create_multi_dataset_sampler(
    dataset, 
    data_config, 
    sampler_type: str = "weighted",
    dataset_ratios: Union[Dict[str, float], None] = None
):
    """
    创建多数据集采样器的工厂函数
    
    Args:
        dataset: 数据集
        data_config: 数据配置
        sampler_type: 采样器类型 ("weighted", "deterministic", "standard")
        dataset_ratios: 数据集比例字典
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler
    
    if dataset_ratios is None or not isinstance(dataset_ratios, dict) or not dataset_ratios:
        # 使用标准采样器
        print("Using standard sampler (no multi-dataset ratio control)")
        if data_config.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(data_config.get("seed", 1))
            return RandomSampler(data_source=dataset, generator=train_dataloader_generator)
        else:
            return SequentialSampler(data_source=dataset)
    
    # 验证比例之和
    total_ratio = sum(dataset_ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"WARNING: dataset ratios sum to {total_ratio:.6f}, normalizing to 1.0")
        dataset_ratios = {k: v/total_ratio for k, v in dataset_ratios.items()}
        print(f"Normalized ratios: {dataset_ratios}")
    
    print(f"Creating multi-dataset sampler with ratios: {dataset_ratios}")
    
    if sampler_type == "batch":
        # 使用批次采样器 - 严格保证每个batch内的比例
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        
        batch_size = data_config.get("gen_batch_size", data_config.train_batch_size)
        
        return MultiDatasetBatchSampler(
            dataset=dataset,
            dataset_ratios=dataset_ratios,
            batch_size=batch_size,
            shuffle=data_config.get("shuffle", True),
            drop_last=True,
            generator=train_dataloader_generator
        )
    
    elif sampler_type == "weighted":
        # 使用加权随机采样器 - 改变整体采样概率但不保证每个batch比例
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        
        return MultiDatasetWeightedSampler(
            dataset=dataset,
            dataset_ratios=dataset_ratios,
            generator=train_dataloader_generator,
            replacement=True
        )
    
    elif sampler_type == "deterministic":
        # 使用确定性比例采样器
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        
        # DeterministicRatioSampler需要batch_size来严格控制每个batch的比例
        batch_size = data_config.get("gen_batch_size", data_config.train_batch_size)
        
        return DeterministicRatioSampler(
            dataset=dataset,
            dataset_ratios=dataset_ratios,
            batch_size=batch_size,
            shuffle=data_config.shuffle,
            generator=train_dataloader_generator
        )
    
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")


# 批次统计分析工具
class BatchRatioAnalyzer:
    """批次比例分析器 - 用于监控和验证批次比例"""
    
    def __init__(self, target_ratios: Dict[str, float]):
        self.target_ratios = target_ratios
        self.batch_history = []
    
    def analyze_batch(self, batch_data: Dict) -> Dict:
        """分析单个batch的比例分布"""
        if '_batch_stats' in batch_data:
            # 使用OversampleFilterCollator提供的统计信息
            stats = batch_data['_batch_stats']
            self.batch_history.append(stats)
            return stats
        else:
            # 手动分析batch
            data_sources = batch_data.get('data_source', [])
            
            # 处理不同类型的数据源数据
            if hasattr(data_sources, 'tolist'):
                data_sources = data_sources.tolist()
            elif isinstance(data_sources, np.ndarray):
                data_sources = data_sources.tolist()
            elif not isinstance(data_sources, list):
                # 如果是单个值，转换为列表
                data_sources = [data_sources] if data_sources is not None else []
            
            # 过滤掉None值
            data_sources = [ds for ds in data_sources if ds is not None]
            
            if not data_sources:
                print("WARNING: No data_source information found in batch")
                return {
                    'target_ratios': self.target_ratios,
                    'actual_ratios': {},
                    'actual_counts': {},
                    'deviations': {source: 1.0 for source in self.target_ratios},
                    'total_samples': 0
                }
            
            actual_counts = Counter(data_sources)
            total_samples = sum(actual_counts.values())
            actual_ratios = {source: count/total_samples for source, count in actual_counts.items()}
            
            deviations = {}
            for source, target_ratio in self.target_ratios.items():
                actual_ratio = actual_ratios.get(source, 0.0)
                deviation = abs(actual_ratio - target_ratio)
                deviations[source] = deviation
            
            stats = {
                'target_ratios': self.target_ratios,
                'actual_ratios': actual_ratios,
                'actual_counts': dict(actual_counts),
                'deviations': deviations,
                'total_samples': total_samples
            }
            
            self.batch_history.append(stats)
            return stats
    
    def get_summary_stats(self, recent_batches: int = 100) -> Dict:
        """获取最近若干batch的统计摘要"""
        if not self.batch_history:
            return {}
        
        recent_history = self.batch_history[-recent_batches:]
        
        # 计算平均偏差
        avg_deviations = defaultdict(list)
        for batch_stats in recent_history:
            if 'deviations' in batch_stats:
                for source, deviation in batch_stats['deviations'].items():
                    avg_deviations[source].append(deviation)
        
        summary = {
            'total_analyzed_batches': len(self.batch_history),
            'recent_batches': len(recent_history),
            'target_ratios': self.target_ratios,
        }
        
        for source in self.target_ratios.keys():
            if source in avg_deviations:
                deviations = avg_deviations[source]
                summary[f'{source}_avg_deviation'] = np.mean(deviations)
                summary[f'{source}_max_deviation'] = np.max(deviations)
                summary[f'{source}_std_deviation'] = np.std(deviations)
        
        return summary
    
    def print_analysis(self, recent_batches: int = 100):
        """打印分析结果"""
        summary = self.get_summary_stats(recent_batches)
        
        print("\n" + "="*60)
        print("BATCH RATIO ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total analyzed batches: {summary.get('total_analyzed_batches', 0)}")
        print(f"Recent batches analyzed: {summary.get('recent_batches', 0)}")
        print("\nTarget vs Actual Performance:")
        
        for source, target_ratio in self.target_ratios.items():
            avg_dev = summary.get(f'{source}_avg_deviation', 'N/A')
            max_dev = summary.get(f'{source}_max_deviation', 'N/A')
            
            if avg_dev != 'N/A':
                print(f"  {source:15} | Target: {target_ratio:.3f} | Avg Dev: {avg_dev:.4f} | Max Dev: {max_dev:.4f}")
            else:
                print(f"  {source:15} | Target: {target_ratio:.3f} | No data available")
        
        print("="*60 + "\n") 