import json
import os
import random
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading

import torch.distributed as dist


class RewardLogger:
    """
    Logger for reward calculation samples.
    
    This logger saves complete information (prompt, response, ground truth, scores)
    for a specified percentage of samples to disk for later analysis.
    It supports distributed training environments by ensuring each process
    logs to its own file.
    """

    def __init__(
        self,
        log_dir: str,
        prefix: str = "reward_logs",
        log_percentage: float = 0.1,
        verbose: bool = False,
    ):
        """
        Initialize a reward logger.
        
        Args:
            log_dir: Directory where log files will be stored
            prefix: Prefix for log filenames
            log_percentage: Percentage of samples to log (0.0 to 1.0)
            verbose: Whether to print additional information to stdout
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.prefix = prefix
        self.log_percentage = log_percentage
        self.verbose = verbose
        
        # Set up process rank for distributed training
        self.rank = 0
        self.world_size = 1
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        
        # Create log file path specific to this process
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{prefix}_{timestamp}_rank{self.rank}.jsonl"
        
        # Setup logger
        self.logger = logging.getLogger(f"RewardLogger_rank{self.rank}")
        self.logger.setLevel(logging.INFO)
        
        # Lock for thread-safe file writing
        self.file_lock = threading.Lock()
        
        # Log initialization
        if self.verbose and self.rank == 0:
            print(f"Reward logger initialized at {self.log_file}, logging {self.log_percentage*100:.1f}% of samples")
    
    def log_info(self, message: str):
        """Log general information."""
        if self.verbose and self.rank == 0:
            print(f"[RewardLogger] {message}")
        self.logger.info(message)
    
    def log_sample(
        self, 
        prompt: str, 
        response: str, 
        ground_truth: Any, 
        data_source: str,
        score: float,
        extra_info: Optional[Dict] = None,
        step: Optional[int] = None,
        batch_idx: Optional[int] = None
    ) -> bool:
        """
        Log a single sample if it's within the logging percentage.
        
        Args:
            prompt: The prompt text
            response: The model's response text
            ground_truth: The ground truth for evaluation
            data_source: Source of the data
            score: The calculated reward score
            extra_info: Additional information to log
            step: Training step number
            batch_idx: Index in the batch
            
        Returns:
            True if the sample was logged, False otherwise
        """
        # Randomly decide whether to log this sample based on log_percentage
        if random.random() > self.log_percentage:
            return False
            
        log_entry = {
            "timestamp": time.time(),
            "rank": self.rank,
            "step": step,
            "batch_idx": batch_idx,
            "data_source": data_source,
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "score": score
        }
        
        # Add extra info if provided
        if extra_info:
            log_entry["extra_info"] = extra_info
            
        # Write to file with lock to ensure thread safety
        with self.file_lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        return True
    
    def log_batch_summary(
        self,
        batch_size: int,
        avg_score: float,
        data_sources: List[str],
        step: Optional[int] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log batch summary statistics.
        
        Args:
            batch_size: Size of the batch
            avg_score: Average reward score for the batch
            data_sources: List of data sources in the batch
            step: Training step number
            additional_metrics: Any additional metrics to log
        """
        if not self.verbose:
            return
            
        summary = {
            "timestamp": time.time(),
            "rank": self.rank,
            "step": step,
            "batch_size": batch_size,
            "avg_score": avg_score,
            "data_sources": data_sources
        }
        
        if additional_metrics:
            summary.update(additional_metrics)
            
        # Only log batch summaries on rank 0
        if self.rank == 0:
            with self.file_lock:
                with open(self.log_dir / f"{self.prefix}_summary.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(summary, ensure_ascii=False) + "\n") 