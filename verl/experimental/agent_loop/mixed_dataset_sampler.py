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
Mixed dataset sampler for VERL agent loop training.
Adapted from the original implementation to work with VERL's agent_loop architecture.
"""

import numpy as np
import torch
from collections import Counter, defaultdict
from typing import Dict, List, Union, Optional
from torch.utils.data import WeightedRandomSampler, Sampler, BatchSampler
import logging

logger = logging.getLogger(__name__)


class AgentLoopBatchSampler(Sampler):
    """
    Batch sampler that ensures data source ratios within each batch for agent loop training.
    This sampler also assigns appropriate agent_name to each sample based on data source.
    """
    
    def __init__(
        self,
        dataset,
        dataset_ratios: Dict[str, float],
        agent_mapping: Dict[str, str],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        generator=None
    ):
        """
        Initialize the batch sampler.
        
        Args:
            dataset: The dataset to sample from
            dataset_ratios: Ratio of each data source {"taco": 0.4, "nq": 0.3, "math": 0.3}
            agent_mapping: Mapping from data source to agent name {"taco": "code_execution_agent", "nq": "tool_agent"}
            batch_size: Size of each batch
            shuffle: Whether to shuffle samples
            drop_last: Whether to drop the last incomplete batch
            generator: Random number generator
        """
        self.dataset = dataset
        self.dataset_ratios = dataset_ratios
        self.agent_mapping = agent_mapping
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator or torch.Generator()
        
        # Validate ratios
        total_ratio = sum(dataset_ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Dataset ratios sum to {total_ratio}, should be 1.0")
        
        # Group indices by data source
        self.data_source_indices = self._group_indices_by_source()
        
        # Calculate batch composition
        self.batch_composition = self._calculate_batch_composition()
        
        logger.info(f"AgentLoopBatchSampler initialized:")
        logger.info(f"  Dataset ratios: {dataset_ratios}")
        logger.info(f"  Agent mapping: {agent_mapping}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Batch composition: {self.batch_composition}")
        logger.info(f"  Data source counts: {[(k, len(v)) for k, v in self.data_source_indices.items()]}")

    def _group_indices_by_source(self) -> Dict[str, List[int]]:
        """Group dataset indices by data source."""
        source_indices = defaultdict(list)
        
        # Handle different dataset types
        if hasattr(self.dataset, 'dataframe'):
            # For RLHFDataset with direct dataframe access
            for idx in range(len(self.dataset.dataframe)):
                try:
                    raw_item = self.dataset.dataframe[idx]
                    data_source = raw_item.get('data_source', 'unknown')
                    source_indices[data_source].append(idx)
                except Exception as e:
                    logger.warning(f"Error accessing dataframe item {idx}: {e}")
                    source_indices['unknown'].append(idx)
        else:
            # Fallback to standard dataset access
            for idx in range(len(self.dataset)):
                try:
                    item = self.dataset[idx]
                    if isinstance(item, dict):
                        data_source = item.get('data_source', 'unknown')
                    else:
                        # For datasets that return other formats
                        data_source = getattr(item, 'data_source', 'unknown')
                    source_indices[data_source].append(idx)
                except Exception as e:
                    logger.warning(f"Error accessing dataset item {idx}: {e}")
                    source_indices['unknown'].append(idx)
        
        return dict(source_indices)

    def _calculate_batch_composition(self) -> Dict[str, int]:
        """Calculate how many samples from each data source per batch."""
        composition = {}
        total_assigned = 0
        
        # Assign by ratio, rounding down
        for source, ratio in self.dataset_ratios.items():
            if source in self.data_source_indices:
                count = int(ratio * self.batch_size)
                composition[source] = count
                total_assigned += count
        
        # Handle rounding errors by distributing remaining slots
        remaining = self.batch_size - total_assigned
        if remaining > 0:
            # Sort sources by fractional part (descending)
            fractional_parts = []
            for source, ratio in self.dataset_ratios.items():
                if source in self.data_source_indices:
                    fractional = (ratio * self.batch_size) % 1
                    fractional_parts.append((fractional, source))
            
            fractional_parts.sort(reverse=True)
            for i in range(remaining):
                if i < len(fractional_parts):
                    _, source = fractional_parts[i]
                    composition[source] += 1
        
        return composition

    def __iter__(self):
        """Generate batches with controlled data source ratios."""
        # Create shuffled iterators for each data source
        source_iterators = {}
        for source, indices in self.data_source_indices.items():
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=self.generator)
                shuffled_indices = [indices[i] for i in perm]
                # Repeat to avoid exhaustion
                source_iterators[source] = iter(shuffled_indices * 1000)
            else:
                source_iterators[source] = iter(indices * 1000)
        
        # Calculate number of batches
        num_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            num_batches += 1
        
        for batch_idx in range(num_batches):
            batch_indices = []
            batch_data_sources = []
            batch_agent_names = []
            
            # Handle last incomplete batch
            if batch_idx == num_batches - 1 and not self.drop_last:
                remaining_samples = len(self.dataset) - batch_idx * self.batch_size
                if remaining_samples < self.batch_size:
                    # Scale composition for incomplete batch
                    composition = self._scale_composition_for_incomplete_batch(remaining_samples)
                else:
                    composition = self.batch_composition
            else:
                composition = self.batch_composition
            
            # Sample from each data source according to composition
            for source, count in composition.items():
                if source in source_iterators and count > 0:
                    for _ in range(count):
                        try:
                            idx = next(source_iterators[source])
                            batch_indices.append(idx)
                            batch_data_sources.append(source)
                            # Map data source to agent name
                            agent_name = self.agent_mapping.get(source, "single_turn_agent")
                            batch_agent_names.append(agent_name)
                        except StopIteration:
                            # Refresh iterator and try again
                            self._refresh_iterator(source, source_iterators)
                            idx = next(source_iterators[source])
                            batch_indices.append(idx)
                            batch_data_sources.append(source)
                            agent_name = self.agent_mapping.get(source, "single_turn_agent")
                            batch_agent_names.append(agent_name)
            
            # Shuffle within batch while preserving metadata alignment
            if self.shuffle and len(batch_indices) > 1:
                combined = list(zip(batch_indices, batch_data_sources, batch_agent_names))
                perm = torch.randperm(len(combined), generator=self.generator)
                shuffled_combined = [combined[i] for i in perm]
                batch_indices, batch_data_sources, batch_agent_names = zip(*shuffled_combined)
                batch_indices = list(batch_indices)
                batch_data_sources = list(batch_data_sources)
                batch_agent_names = list(batch_agent_names)
            
            # Yield batch with metadata
            yield {
                'indices': batch_indices,
                'data_sources': batch_data_sources,
                'agent_names': batch_agent_names
            }

    def _scale_composition_for_incomplete_batch(self, remaining_samples: int) -> Dict[str, int]:
        """Scale batch composition for incomplete final batch."""
        scaled_composition = {}
        total_assigned = 0
        
        for source, count in self.batch_composition.items():
            scaled_count = int(count * remaining_samples / self.batch_size)
            scaled_composition[source] = scaled_count
            total_assigned += scaled_count
        
        # Distribute remaining samples
        remaining = remaining_samples - total_assigned
        sources = list(scaled_composition.keys())
        for i in range(remaining):
            if sources:
                source = sources[i % len(sources)]
                scaled_composition[source] += 1
        
        return scaled_composition

    def _refresh_iterator(self, source: str, source_iterators: Dict[str, iter]):
        """Refresh an exhausted iterator."""
        indices = self.data_source_indices[source]
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=self.generator)
            shuffled_indices = [indices[i] for i in perm]
            source_iterators[source] = iter(shuffled_indices * 1000)
        else:
            source_iterators[source] = iter(indices * 1000)

    def __len__(self):
        """Return total number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class MixedDatasetCollator:
    """
    Custom collator that handles mixed data sources and agent assignments.
    Works with AgentLoopBatchSampler to inject agent_name into batch data.
    """
    
    def __init__(self, base_collator, tokenizer=None):
        """
        Initialize the mixed dataset collator.
        
        Args:
            base_collator: The original collate function
            tokenizer: Tokenizer (if needed for special processing)
        """
        self.base_collator = base_collator
        self.tokenizer = tokenizer
    
    def __call__(self, batch_data):
        """
        Collate batch data with agent assignment.
        
        Args:
            batch_data: Either a list of samples or a dict from AgentLoopBatchSampler
            
        Returns:
            Collated batch with agent_name and data_source fields
        """
        if isinstance(batch_data, dict) and 'indices' in batch_data:
            # This is from our AgentLoopBatchSampler
            indices = batch_data['indices']
            data_sources = batch_data['data_sources']
            agent_names = batch_data['agent_names']
            
            # Get actual samples (this is a simplified approach)
            # In practice, you might need to handle this differently
            # depending on how your dataset works
            samples = [self._get_sample_by_index(idx) for idx in indices]
            
            # Apply base collator
            if self.base_collator:
                collated_batch = self.base_collator(samples)
            else:
                collated_batch = samples
            
            # Add metadata
            if isinstance(collated_batch, dict):
                collated_batch['data_source'] = np.array(data_sources, dtype=object)
                collated_batch['agent_name'] = np.array(agent_names, dtype=object)
            else:
                # Handle other collator return types
                logger.warning("Base collator returned non-dict, metadata may not be properly added")
            
            return collated_batch
        else:
            # Standard batch, use base collator
            if self.base_collator:
                return self.base_collator(batch_data)
            else:
                return batch_data
    
    def _get_sample_by_index(self, idx):
        """
        Get sample by index. This is a placeholder - you'll need to implement
        this based on your dataset structure.
        """
        # This is a placeholder implementation
        # You'll need to adapt this to your dataset structure
        raise NotImplementedError("Please implement _get_sample_by_index based on your dataset")


def create_mixed_agent_loop_sampler(
    dataset,
    config,
    dataset_ratios: Optional[Dict[str, float]] = None,
    agent_mapping: Optional[Dict[str, str]] = None
) -> Union[Sampler, AgentLoopBatchSampler]:
    """
    Factory function to create a mixed dataset sampler for agent loop training.
    
    Args:
        dataset: The dataset to sample from
        config: Training configuration
        dataset_ratios: Ratio of each data source in batches
        agent_mapping: Mapping from data source to agent loop name
        
    Returns:
        Appropriate sampler for the configuration
    """
    # Extract config values
    batch_size = config.get("gen_batch_size", config.get("train_batch_size", 32))
    shuffle = config.get("shuffle", True)
    seed = config.get("seed", 1)
    
    # Set up generator
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Check if mixed dataset sampling is needed
    if not dataset_ratios or not agent_mapping:
        logger.info("No mixed dataset configuration found, using standard sampler")
        # Fall back to standard sampler
        if shuffle:
            from torch.utils.data import RandomSampler
            return RandomSampler(dataset, generator=generator)
        else:
            from torch.utils.data import SequentialSampler
            return SequentialSampler(dataset)
    
    # Validate configuration
    if not isinstance(dataset_ratios, dict) or not isinstance(agent_mapping, dict):
        raise ValueError("dataset_ratios and agent_mapping must be dictionaries")
    
    # Normalize ratios
    total_ratio = sum(dataset_ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        logger.warning(f"Dataset ratios sum to {total_ratio:.6f}, normalizing to 1.0")
        dataset_ratios = {k: v/total_ratio for k, v in dataset_ratios.items()}
    
    # Check that all data sources have agent mappings
    missing_mappings = set(dataset_ratios.keys()) - set(agent_mapping.keys())
    if missing_mappings:
        logger.warning(f"Missing agent mappings for data sources: {missing_mappings}")
        # Add default mappings
        for source in missing_mappings:
            agent_mapping[source] = "single_turn_agent"
    
    logger.info(f"Creating mixed agent loop sampler:")
    logger.info(f"  Dataset ratios: {dataset_ratios}")
    logger.info(f"  Agent mapping: {agent_mapping}")
    logger.info(f"  Batch size: {batch_size}")
    
    return AgentLoopBatchSampler(
        dataset=dataset,
        dataset_ratios=dataset_ratios,
        agent_mapping=agent_mapping,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        generator=generator
    )


class BatchRatioAnalyzer:
    """
    Analyzer for monitoring batch composition in mixed dataset training.
    Adapted for VERL agent loop training.
    """
    
    def __init__(self, target_ratios: Dict[str, float]):
        self.target_ratios = target_ratios
        self.batch_history = []
    
    def analyze_batch(self, batch_data) -> Dict:
        """
        Analyze a batch's data source distribution.
        
        Args:
            batch_data: Batch data (either dict or DataProto)
            
        Returns:
            Analysis results
        """
        # Extract data sources from batch
        data_sources = None
        
        if hasattr(batch_data, 'non_tensor_batch'):
            # DataProto format
            data_sources = batch_data.non_tensor_batch.get('data_source', [])
        elif isinstance(batch_data, dict):
            # Dict format
            data_sources = batch_data.get('data_source', [])
        
        if data_sources is None:
            logger.warning("No data_source information found in batch")
            return self._empty_analysis()
        
        # Convert to list if needed
        if hasattr(data_sources, 'tolist'):
            data_sources = data_sources.tolist()
        elif isinstance(data_sources, np.ndarray):
            data_sources = data_sources.tolist()
        elif not isinstance(data_sources, list):
            data_sources = [data_sources] if data_sources is not None else []
        
        # Filter None values
        data_sources = [ds for ds in data_sources if ds is not None]
        
        if not data_sources:
            logger.warning("No valid data_source entries found in batch")
            return self._empty_analysis()
        
        # Calculate statistics
        actual_counts = Counter(data_sources)
        total_samples = sum(actual_counts.values())
        actual_ratios = {source: count/total_samples for source, count in actual_counts.items()}
        
        # Calculate deviations
        deviations = {}
        for source, target_ratio in self.target_ratios.items():
            actual_ratio = actual_ratios.get(source, 0.0)
            deviation = abs(actual_ratio - target_ratio)
            deviations[source] = deviation
        
        analysis = {
            'target_ratios': self.target_ratios,
            'actual_ratios': actual_ratios,
            'actual_counts': dict(actual_counts),
            'deviations': deviations,
            'total_samples': total_samples,
            'max_deviation': max(deviations.values()) if deviations else 0.0
        }
        
        self.batch_history.append(analysis)
        return analysis
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis when no data is available."""
        return {
            'target_ratios': self.target_ratios,
            'actual_ratios': {},
            'actual_counts': {},
            'deviations': {source: 1.0 for source in self.target_ratios},
            'total_samples': 0,
            'max_deviation': 1.0
        }
    
    def get_summary_stats(self, recent_batches: int = 100) -> Dict:
        """Get summary statistics for recent batches."""
        if not self.batch_history:
            return {}
        
        recent_history = self.batch_history[-recent_batches:]
        
        # Calculate average deviations
        avg_deviations = defaultdict(list)
        max_deviations = []
        
        for batch_stats in recent_history:
            if 'deviations' in batch_stats:
                max_deviations.append(batch_stats.get('max_deviation', 0.0))
                for source, deviation in batch_stats['deviations'].items():
                    avg_deviations[source].append(deviation)
        
        summary = {
            'total_analyzed_batches': len(self.batch_history),
            'recent_batches': len(recent_history),
            'target_ratios': self.target_ratios,
            'overall_max_deviation_avg': np.mean(max_deviations) if max_deviations else 0.0,
            'overall_max_deviation_max': np.max(max_deviations) if max_deviations else 0.0,
        }
        
        # Per-source statistics
        for source in self.target_ratios.keys():
            if source in avg_deviations:
                deviations = avg_deviations[source]
                summary[f'{source}_avg_deviation'] = np.mean(deviations)
                summary[f'{source}_max_deviation'] = np.max(deviations)
                summary[f'{source}_std_deviation'] = np.std(deviations)
            else:
                summary[f'{source}_avg_deviation'] = 'N/A'
                summary[f'{source}_max_deviation'] = 'N/A'
                summary[f'{source}_std_deviation'] = 'N/A'
        
        return summary
    
    def print_analysis(self, recent_batches: int = 100):
        """Print analysis results."""
        summary = self.get_summary_stats(recent_batches)
        
        print("\n" + "="*70)
        print("AGENT LOOP BATCH RATIO ANALYSIS")
        print("="*70)
        print(f"Total analyzed batches: {summary.get('total_analyzed_batches', 0)}")
        print(f"Recent batches analyzed: {summary.get('recent_batches', 0)}")
        print(f"Overall max deviation: avg={summary.get('overall_max_deviation_avg', 0):.4f}, "
              f"max={summary.get('overall_max_deviation_max', 0):.4f}")
        print("\nPer-source performance:")
        
        for source, target_ratio in self.target_ratios.items():
            avg_dev = summary.get(f'{source}_avg_deviation', 'N/A')
            max_dev = summary.get(f'{source}_max_deviation', 'N/A')
            
            if avg_dev != 'N/A':
                print(f"  {source:15} | Target: {target_ratio:.3f} | "
                      f"Avg Dev: {avg_dev:.4f} | Max Dev: {max_dev:.4f}")
            else:
                print(f"  {source:15} | Target: {target_ratio:.3f} | No data available")
        
        print("="*70 + "\n") 