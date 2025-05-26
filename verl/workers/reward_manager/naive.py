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

from collections import defaultdict
import os
from pathlib import Path

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.utils.logger.reward_logger import RewardLogger


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, 
                 tokenizer, 
                 num_examine, 
                 compute_score=None, 
                 reward_fn_key="data_source",
                 config=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        
        # Initialize reward logger if logging is enabled
        self.enable_logging = True  # Default to enabled
        self.log_percentage = 0.1  # Default to 10%
        self.log_dir = None
        
        # Parse configuration if provided
        if config is not None and hasattr(config, 'reward_model') and hasattr(config.reward_model, 'logging'):
            logging_config = config.reward_model.logging
            if hasattr(logging_config, 'enable'):
                self.enable_logging = logging_config.enable
            if hasattr(logging_config, 'log_percentage'):
                self.log_percentage = logging_config.log_percentage
            if hasattr(logging_config, 'log_dir'):
                self.log_dir = logging_config.log_dir
        
        self.reward_logger = None
        if self.enable_logging:
            # Determine log directory
            if self.log_dir is None:
                # Try to get from environment variables
                self.log_dir = os.environ.get('REWARD_LOG_DIR', 'logs/reward_logs')
                
                # Get experiment name from environment if available
                experiment_name = os.environ.get('EXPERIMENT_NAME', '')
                if experiment_name:
                    self.log_dir = Path(self.log_dir) / experiment_name
            
            # Get log percentage from environment if available
            env_log_percentage = os.environ.get('REWARD_LOG_PERCENTAGE')
            if env_log_percentage:
                try:
                    self.log_percentage = float(env_log_percentage)
                except ValueError:
                    print(f"[WARNING] Invalid REWARD_LOG_PERCENTAGE: {env_log_percentage}")
            
            print(f"[RewardLogger] Initializing with log_dir: {self.log_dir}, percentage: {self.log_percentage}")
            
            # Initialize the logger
            self.reward_logger = RewardLogger(
                log_dir=self.log_dir,
                prefix="reward_samples",
                log_percentage=self.log_percentage,
                verbose=True
            )
            
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # Track scores for batch summary
        total_score = 0.0
        data_sources_in_batch = set()

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            data_sources_in_batch.add(data_source)

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                score_value = reward  # For logging and averaging
            else:
                reward = score
                score_value = score  # For logging and averaging

            reward_tensor[i, valid_response_length - 1] = reward
            total_score += score_value

            # Log sample data using our reward logger (if enabled)
            if self.enable_logging and self.reward_logger:
                # Pass all the sample data to the logger
                # The logger internally decides whether to log this sample based on percentage
                score_dict = score if isinstance(score, dict) else {"score": score}
                self.reward_logger.log_sample(
                    prompt=prompt_str,
                    response=response_str,
                    ground_truth=ground_truth,
                    data_source=data_source,
                    score=reward,  # Primary score
                    extra_info={
                        "score_details": score_dict,
                        "extra_data": extra_info
                    },
                    batch_idx=i
                )

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
        
        # Log batch summary if logger is enabled
        if self.enable_logging and self.reward_logger and len(data) > 0:
            self.reward_logger.log_batch_summary(
                batch_size=len(data),
                avg_score=total_score / len(data),
                data_sources=list(data_sources_in_batch),
                additional_metrics={"total_score": total_score}
            )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
