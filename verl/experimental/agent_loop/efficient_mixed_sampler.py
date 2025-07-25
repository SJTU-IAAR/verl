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
Efficient mixed dataset sampler optimized for VERL agent loop training.
This implementation focuses on performance and memory efficiency.
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Optional, Union
from torch.utils.data import Sampler
import logging

logger = logging.getLogger(__name__)


class VERLMixedBatchSampler(Sampler):
    """
    High-performance batch sampler for VERL agent loop training with mixed data sources.
    
    Key optimizations:
    - Pre-computed batch schedules for faster iteration
    - Memory-efficient index management
    - Optimized for VERL's DataProto format
    - Supports dynamic agent assignment
    """
    
    def __init__(
        self,
        dataset,
        dataset_ratios: Dict[str, float],
        agent_mapping: Dict[str, str],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        generator=None,
        precompute_batches: bool = True
    ):
        self.dataset = dataset
        self.dataset_ratios = dataset_ratios
        self.agent_mapping = agent_mapping
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator or torch.Generator()
        self.precompute_batches = precompute_batches
        
        # Validate and normalize ratios
        self._validate_ratios()
        
        # Fast index grouping
        self.source_indices = self._fast_group_indices()
        
        # Pre-compute batch composition
        self.batch_composition = self._compute_batch_composition()
        
        # Pre-compute batch schedule if enabled
        self._batch_schedule = None
        if precompute_batches:
            self._precompute_batch_schedule()
        
        logger.info(f"VERLMixedBatchSampler initialized:")
        logger.info(f"  Ratios: {self.dataset_ratios}")
        logger.info(f"  Agents: {self.agent_mapping}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Precompute: {precompute_batches}")
        logger.info(f"  Sources: {[(k, len(v)) for k, v in self.source_indices.items()]}")

    def _validate_ratios(self):
        """Validate and normalize dataset ratios."""
        total_ratio = sum(self.dataset_ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            logger.warning(f"Normalizing ratios from {total_ratio:.6f} to 1.0")
            self.dataset_ratios = {k: v/total_ratio for k, v in self.dataset_ratios.items()}

    def _fast_group_indices(self) -> Dict[str, np.ndarray]:
        """Fast grouping of indices by data source using vectorized operations."""
        source_indices = defaultdict(list)
        
        try:
            # Try vectorized approach for RLHFDataset
            if hasattr(self.dataset, 'dataframe') and hasattr(self.dataset.dataframe, '__len__'):
                # Batch process for better performance
                batch_size = 10000
                for start_idx in range(0, len(self.dataset.dataframe), batch_size):
                    end_idx = min(start_idx + batch_size, len(self.dataset.dataframe))
                    
                    for idx in range(start_idx, end_idx):
                        try:
                            item = self.dataset.dataframe[idx]
                            source = item.get('data_source', 'unknown')
                            source_indices[source].append(idx)
                        except Exception:
                            source_indices['unknown'].append(idx)
            else:
                # Fallback for other dataset types
                for idx in range(len(self.dataset)):
                    try:
                        item = self.dataset[idx]
                        source = item.get('data_source', 'unknown') if isinstance(item, dict) else 'unknown'
                        source_indices[source].append(idx)
                    except Exception:
                        source_indices['unknown'].append(idx)
        
        except Exception as e:
            logger.error(f"Error in fast grouping: {e}, falling back to safe mode")
            # Ultra-safe fallback
            for idx in range(len(self.dataset)):
                source_indices['unknown'].append(idx)
        
        # Convert to numpy arrays for faster indexing
        return {k: np.array(v, dtype=np.int32) for k, v in source_indices.items()}

    def _compute_batch_composition(self) -> Dict[str, int]:
        """Compute exact number of samples per source per batch."""
        composition = {}
        
        # Primary allocation
        for source, ratio in self.dataset_ratios.items():
            if source in self.source_indices:
                composition[source] = int(ratio * self.batch_size)
        
        # Handle rounding via priority queue
        allocated = sum(composition.values())
        remaining = self.batch_size - allocated
        
        if remaining > 0:
            # Distribute remaining slots by largest fractional parts
            fractionals = []
            for source, ratio in self.dataset_ratios.items():
                if source in self.source_indices:
                    fractional = (ratio * self.batch_size) % 1
                    fractionals.append((fractional, source))
            
            fractionals.sort(reverse=True)
            for i in range(min(remaining, len(fractionals))):
                _, source = fractionals[i]
                composition[source] += 1
        
        return composition

    def _precompute_batch_schedule(self):
        """Pre-compute entire epoch's batch schedule for maximum efficiency."""
        num_batches = self._compute_num_batches()
        self._batch_schedule = []
        
        # Create per-source iterators
        source_pools = {}
        for source, indices in self.source_indices.items():
            # Pre-shuffle if needed
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=self.generator)
                shuffled_indices = indices[perm.numpy()]
            else:
                shuffled_indices = indices
            
            # Create large enough pool to avoid re-shuffling
            repeats = max(1, (num_batches * self.batch_composition.get(source, 0)) // len(indices) + 2)
            source_pools[source] = np.tile(shuffled_indices, repeats)
        
        # Generate all batches
        source_positions = {source: 0 for source in source_pools}
        
        for batch_idx in range(num_batches):
            batch_indices = []
            batch_sources = []
            batch_agents = []
            
            for source, count in self.batch_composition.items():
                if source in source_pools and count > 0:
                    start_pos = source_positions[source]
                    end_pos = start_pos + count
                    
                    # Handle wraparound
                    pool = source_pools[source]
                    if end_pos <= len(pool):
                        selected_indices = pool[start_pos:end_pos]
                        source_positions[source] = end_pos
                    else:
                        # Need to wrap around
                        part1 = pool[start_pos:]
                        needed = count - len(part1)
                        part2 = pool[:needed]
                        selected_indices = np.concatenate([part1, part2])
                        source_positions[source] = needed
                    
                    batch_indices.extend(selected_indices.tolist())
                    batch_sources.extend([source] * count)
                    
                    agent_name = self.agent_mapping.get(source, "single_turn_agent")
                    batch_agents.extend([agent_name] * count)
            
            # Intra-batch shuffle
            if self.shuffle and len(batch_indices) > 1:
                combined = list(zip(batch_indices, batch_sources, batch_agents))
                perm = torch.randperm(len(combined), generator=self.generator)
                shuffled = [combined[i] for i in perm]
                batch_indices, batch_sources, batch_agents = zip(*shuffled)
            
            self._batch_schedule.append({
                'indices': list(batch_indices),
                'sources': list(batch_sources),
                'agents': list(batch_agents)
            })

    def _compute_num_batches(self) -> int:
        """Compute total number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Iterate over batches."""
        if self.precompute_batches and self._batch_schedule:
            # Use pre-computed schedule
            for batch_data in self._batch_schedule:
                yield batch_data
        else:
            # Dynamic generation (fallback)
            yield from self._dynamic_iter()

    def _dynamic_iter(self):
        """Dynamic batch generation (used as fallback)."""
        num_batches = self._compute_num_batches()
        
        # Initialize source iterators
        source_positions = {source: 0 for source in self.source_indices}
        
        for _ in range(num_batches):
            batch_indices = []
            batch_sources = []
            batch_agents = []
            
            for source, count in self.batch_composition.items():
                if source in self.source_indices and count > 0:
                    indices = self.source_indices[source]
                    pos = source_positions[source]
                    
                    selected = []
                    for _ in range(count):
                        selected.append(indices[pos % len(indices)])
                        pos += 1
                    
                    source_positions[source] = pos
                    
                    batch_indices.extend(selected)
                    batch_sources.extend([source] * count)
                    
                    agent_name = self.agent_mapping.get(source, "single_turn_agent")
                    batch_agents.extend([agent_name] * count)
            
            yield {
                'indices': batch_indices,
                'sources': batch_sources,
                'agents': batch_agents
            }

    def __len__(self):
        return self._compute_num_batches()


class StreamlinedVERLCollator:
    """
    Streamlined collator optimized for VERL's DataProto format.
    Minimal overhead while preserving agent loop functionality.
    """
    
    def __init__(self, base_collator, dataset=None):
        self.base_collator = base_collator
        self.dataset = dataset
    
    def __call__(self, batch_input):
        """
        Process batch with minimal overhead.
        
        Handles both:
        1. Standard batch (list of samples)
        2. Mixed batch (dict with metadata from VERLMixedBatchSampler)
        """
        if isinstance(batch_input, dict) and 'indices' in batch_input:
            # Mixed batch from our sampler
            indices = batch_input['indices']
            sources = batch_input['sources']
            agents = batch_input['agents']
            
            # Fast sample extraction
            if self.dataset is not None:
                # Direct dataset access
                if hasattr(self.dataset, 'dataframe'):
                    # Optimized for RLHFDataset
                    samples = [self.dataset.dataframe[idx] for idx in indices]
                else:
                    samples = [self.dataset[idx] for idx in indices]
            else:
                # Fallback - reconstruct from indices
                samples = [{'__index__': idx} for idx in indices]
            
            # Apply base collator
            if callable(self.base_collator):
                batch = self.base_collator(samples)
            else:
                batch = samples
            
            # Inject metadata efficiently
            if isinstance(batch, dict):
                batch['data_source'] = np.array(sources, dtype=object)
                batch['agent_name'] = np.array(agents, dtype=object)
            
            return batch
        else:
            # Standard batch
            if callable(self.base_collator):
                return self.base_collator(batch_input)
            else:
                return batch_input


def create_verl_mixed_sampler(
    dataset,
    config,
    dataset_ratios: Optional[Dict[str, float]] = None,
    agent_mapping: Optional[Dict[str, str]] = None,
    **kwargs
) -> Union[Sampler, VERLMixedBatchSampler]:
    """
    Factory function optimized for VERL agent loop training.
    
    Args:
        dataset: Dataset to sample from
        config: Training configuration (DictConfig or dict)
        dataset_ratios: Source ratios {"taco": 0.4, "nq": 0.6}
        agent_mapping: Agent mapping {"taco": "code_execution_agent", "nq": "tool_agent"}
        **kwargs: Additional arguments
        
    Returns:
        Optimized sampler
    """
    # Extract configuration
    batch_size = config.get("gen_batch_size", config.get("train_batch_size", 32))
    shuffle = config.get("shuffle", True)
    seed = config.get("seed", 1)
    
    # Performance settings
    precompute = kwargs.get("precompute_batches", True)
    
    # Setup generator
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Check if mixed sampling is needed
    if not dataset_ratios or not agent_mapping:
        logger.info("Using standard sampler (no mixed dataset configuration)")
        if shuffle:
            from torch.utils.data import RandomSampler
            return RandomSampler(dataset, generator=generator)
        else:
            from torch.utils.data import SequentialSampler
            return SequentialSampler(dataset)
    
    # Validate inputs
    if not isinstance(dataset_ratios, dict) or not isinstance(agent_mapping, dict):
        raise ValueError("dataset_ratios and agent_mapping must be dictionaries")
    
    # Ensure all sources have agent mappings
    missing_agents = set(dataset_ratios.keys()) - set(agent_mapping.keys())
    if missing_agents:
        logger.warning(f"Adding default agent mappings for: {missing_agents}")
        for source in missing_agents:
            agent_mapping[source] = "single_turn_agent"
    
    logger.info(f"Creating VERL mixed sampler:")
    logger.info(f"  Dataset ratios: {dataset_ratios}")
    logger.info(f"  Agent mapping: {agent_mapping}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Precompute batches: {precompute}")
    
    return VERLMixedBatchSampler(
        dataset=dataset,
        dataset_ratios=dataset_ratios,
        agent_mapping=agent_mapping,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        generator=generator,
        precompute_batches=precompute
    )


class PerformanceMonitor:
    """
    Lightweight performance monitor for mixed dataset training.
    """
    
    def __init__(self, target_ratios: Dict[str, float]):
        self.target_ratios = target_ratios
        self.recent_batches = []
        self.max_history = 1000  # Keep last 1000 batches
    
    def log_batch(self, batch_data, timing_info: Optional[Dict] = None):
        """Log batch statistics efficiently."""
        # Extract sources
        sources = None
        if hasattr(batch_data, 'non_tensor_batch'):
            sources = batch_data.non_tensor_batch.get('data_source')
        elif isinstance(batch_data, dict):
            sources = batch_data.get('data_source')
        
        if sources is None:
            return
        
        # Convert to list
        if hasattr(sources, 'tolist'):
            sources = sources.tolist()
        elif not isinstance(sources, list):
            sources = [sources]
        
        # Quick statistics
        from collections import Counter
        counts = Counter(sources)
        total = len(sources)
        
        # Calculate max deviation
        max_dev = 0.0
        for source, target_ratio in self.target_ratios.items():
            actual_ratio = counts.get(source, 0) / total if total > 0 else 0.0
            deviation = abs(actual_ratio - target_ratio)
            max_dev = max(max_dev, deviation)
        
        # Store lightweight record
        record = {
            'timestamp': len(self.recent_batches),
            'total_samples': total,
            'max_deviation': max_dev,
            'timing': timing_info
        }
        
        self.recent_batches.append(record)
        
        # Trim history
        if len(self.recent_batches) > self.max_history:
            self.recent_batches = self.recent_batches[-self.max_history:]
    
    def get_recent_performance(self, last_n: int = 100) -> Dict:
        """Get performance summary for recent batches."""
        if not self.recent_batches:
            return {'status': 'no_data'}
        
        recent = self.recent_batches[-last_n:]
        
        avg_deviation = np.mean([r['max_deviation'] for r in recent])
        max_deviation = np.max([r['max_deviation'] for r in recent])
        
        return {
            'status': 'good' if avg_deviation < 0.05 else 'acceptable' if avg_deviation < 0.1 else 'poor',
            'batches_analyzed': len(recent),
            'avg_max_deviation': avg_deviation,
            'worst_max_deviation': max_deviation,
            'recent_trend': 'stable'  # Could add trend analysis
        }
    
    def print_summary(self):
        """Print concise performance summary."""
        perf = self.get_recent_performance()
        status = perf.get('status', 'unknown')
        avg_dev = perf.get('avg_max_deviation', 0.0)
        
        print(f"Mixed Dataset Performance: {status.upper()} "
              f"(avg deviation: {avg_dev:.4f}, "
              f"batches: {perf.get('batches_analyzed', 0)})") 