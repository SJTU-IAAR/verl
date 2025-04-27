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
Tool-enabled vLLM rollout that integrates with remote code execution services.
"""
import re
import torch
import requests
import time
from typing import List, Dict, Any, Tuple, Optional
from contextlib import contextmanager
import torch.nn.utils.rnn as rnn_utils
from omegaconf import DictConfig

import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout, _pre_process_inputs
from verl.workers.rollout.vllm_rollout.tool_client import ToolClient
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

# Add a flag to control verbose logging
VERBOSE_TOOL_LOGGING = True  # Set to False to disable detailed logging

# Counter for sampling trajectories
TRAJECTORY_SAMPLE_INTERVAL = 20
TRAJECTORY_COUNTER = 0
SAMPLED_INDICES = set()

# Helper function for logging
def log_tool_info(message, *args, level='info'):
    if VERBOSE_TOOL_LOGGING or level == 'error':
        formatted_args = []
        for arg in args:
            if isinstance(arg, str) and len(arg) > 200:
                # Use markers for long text
                formatted_args.append(f"\n---\n{arg}\n---")
            else:
                formatted_args.append(str(arg))
        # Add prefix [TOOL_ROLLOUT] and suffix [/TOOL_ROLLOUT]
        print(f"[TOOL_ROLLOUT] {message}: {' '.join(formatted_args)} [/TOOL_ROLLOUT]")

class ToolEnabledVLLMRollout(vLLMRollout):
    """
    Extends vLLMRollout with tool execution capabilities via a remote server.
    """

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """
        Initialize the ToolEnabledVLLMRollout.
        
        Args:
            actor_module: module here follows huggingface APIs
            config: DictConfig with additional tool config
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initialize the generating model in vllm
            **kwargs: additional arguments
        """
        # 获取工具配置
        tool_config = config.get('tool_config', {})
        
        # 打印完整配置用于调试
        print(f"[DEBUG] Full tool_config: {tool_config}")
        
        # 仅使用tool_config.enabled来控制是否启用工具
        self.enable_tools = tool_config.get('enabled', False)
            
        print(f"[DEBUG] Tool execution enabled: {self.enable_tools}")
        
        # Initialize parent class
        super().__init__(actor_module, config, tokenizer, model_hf_config, **kwargs)
        
        # Store tokenizer for consistent access
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        print(f"[DEBUG] Using pad_token_id: {self.pad_token_id}")
        
        # Tool configuration
        self.tool_config = config.get('tool_config', {})
        self.tool_server_url = self.tool_config.get('server_url', 'http://127.0.0.1:30003')
        self.max_retries = self.tool_config.get('max_retries', 3)
        self.retry_delay = self.tool_config.get('retry_delay', 1.0)
        self.timeout = self.tool_config.get('timeout', 30.0)
        self.max_workers = self.tool_config.get('max_workers', 10)
        self.request_interval = self.tool_config.get('request_interval', 0.1)  # seconds
        self.max_turns = config.get('max_turns', 5)
        
        # Tool tokens
        self.tool_start_token = "<tool>"
        self.tool_end_token = "</tool>"
        self.tool_stop = config.get('tool_stop', "</tool>")
        
        # State masking configuration
        self.use_state_masking = config.get('state_masking', True)
        state_masking_config = config.get('state_masking', {})
        if isinstance(state_masking_config, dict):
            self.start_state_marker = state_masking_config.get('start_state_marker', "<output>")
            self.end_state_marker = state_masking_config.get('end_state_marker', "</output>")
        else:
            self.start_state_marker = "<output>"
            self.end_state_marker = "</output>"
        
        # Initialize tool client
        self.tool_client = ToolClient(
            server_url=self.tool_server_url,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            timeout=self.timeout,
            request_interval=self.request_interval,
            max_workers=self.max_workers
        )
        
        # Set up sampling parameters to stop at tool token
        self.sampling_params.stop = [self.tool_stop]
        self.sampling_params.ignore_eos = False
        if hasattr(self.sampling_params, 'detokenize'):
            self.sampling_params.detokenize = True
        
        # Record device information
        try:
            self.model_device = next(self.inference_engine.model.parameters()).device
        except:
            self.model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        print(f"ToolEnabledVLLMRollout initialized with server URL: {self.tool_server_url}")

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def _extract_tool_calls(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract tool calls from the generated text.
        
        Args:
            text: The generated text that may contain tool calls.
            
        Returns:
            List of tuples where each tuple is (before_tool, tool_code, after_tool)
        """
        # Pattern to match the tool call blocks
        pattern = rf'(.*?)({re.escape(self.tool_start_token)})(.*?)({re.escape(self.tool_end_token)})(.*?)'
        
        # Find all matches
        matches = re.findall(pattern, text, re.DOTALL)
        
        # Log the extraction process
        if VERBOSE_TOOL_LOGGING:
            log_tool_info("Extracting tool calls from text", text)
        
        # If no matches, return the original text
        if not matches:
            if VERBOSE_TOOL_LOGGING:
                log_tool_info("No tool calls found in text")
            return [(text, "", "")]
        
        # Process matches
        result = []
        for before, start_token, code, end_token, after in matches:
            result.append((before, code.strip(), after))
            if VERBOSE_TOOL_LOGGING:
                log_tool_info("Found tool call", f"Before: {before}", f"Code: {code.strip()}", f"After: {after}")
        
        return result

    def _process_tools_in_batch(self, responses, turn, sampled_indices):
        """
        Process tool calls in a batch of responses using the ToolClient.
        
        Args:
            responses: Tensor of token IDs [batch_size, response_length]
            turn: Current turn number for logging
            sampled_indices: Set of batch indices selected for detailed trajectory logging
            
        Returns:
            Tuple of (need_continue_flags, tool_results, final_outputs) where:
            - need_continue_flags: List of booleans indicating if each response needs another turn
            - tool_results: Dictionary mapping batch indices to tool execution results
            - final_outputs: List of final outputs or None if more turns needed
        """
        batch_size = responses.size(0)
        
        # Ensure responses are long tensors
        responses_long = responses.long()
        
        # Decode responses to text
        response_texts = self.tokenizer.batch_decode(responses_long, skip_special_tokens=True)
        
        # Log full decoded responses for sampled indices only
        for i in sampled_indices:
            if i < len(response_texts):
                log_tool_info(f"Turn {turn} - Sampled Response [Index {i}]", response_texts[i])
        
        # Process each response for tool calls
        tool_codes = []
        need_continue = [False] * batch_size
        tool_results_mapping = {}  # Maps batch index to tool result position
        final_outputs = [None] * batch_size
        
        for i, text in enumerate(response_texts):
            tool_calls = self._extract_tool_calls(text)
            has_tool_call = False
            
            for _, code, _ in tool_calls:
                if code.strip():
                    tool_codes.append(code.strip())
                    tool_results_mapping[i] = len(tool_codes) - 1
                    need_continue[i] = True
                    has_tool_call = True
                    if i in sampled_indices:
                        log_tool_info(f"Turn {turn} - Sampled Tool Call [Index {i}]", code.strip())
                    break  # Only process the first tool call per response
            
            # If no tool call was found, consider this response complete
            if not has_tool_call:
                if i in sampled_indices:
                    log_tool_info(f"Turn {turn} - Sampled Final Response [Index {i}] - No tool call, marking as complete")
                final_outputs[i] = text
        
        # Execute tool calls in batch
        tool_results = {}  # Dictionary mapping batch indices to tool results
        if tool_codes:
            try:
                results_list = self.tool_client.batch_execute(tool_codes)
                
                # Map results back to original batch indices
                for batch_idx, result_idx in tool_results_mapping.items():
                    if result_idx < len(results_list):
                        result = results_list[result_idx]
                        if result.get("success", False):
                            tool_results[int(batch_idx)] = result.get('output', '')
                            if batch_idx in sampled_indices:
                                log_tool_info(f"Turn {turn} - Sampled Tool Result [Index {batch_idx}]", result.get('output', ''))
                        else:
                            tool_results[int(batch_idx)] = f"Error: {result.get('error', 'Unknown error')}"
                            if batch_idx in sampled_indices:
                                log_tool_info(f"Turn {turn} - Sampled Tool Error [Index {batch_idx}]", result.get('error', 'Unknown error'))
            except Exception as e:
                print(f"Error during tool execution: {e}")
                # Provide a default response
                for batch_idx in tool_results_mapping:
                    tool_results[int(batch_idx)] = "Tool execution failed due to an error."
                    if batch_idx in sampled_indices:
                        log_tool_info(f"Turn {turn} - Sampled Tool Execution Exception [Index {batch_idx}]", str(e))
        
        print(f"[DEBUG] Tool results keys type: {[type(k) for k in tool_results.keys()][:5]}")
        print(f"[DEBUG] Need continue: {sum(need_continue)}/{len(need_continue)}")
        return need_continue, tool_results, final_outputs

    def _create_next_turn_inputs(
            self, 
            original_inputs:torch.Tensor, 
            responses:torch.Tensor, 
            tool_results: Dict[int, str]
        ):
        """Create the next turn's inputs including context and tool execution results
        
        Args:
            original_inputs: Original input IDs, shape [batch_size, seq_len]
            responses: Current response IDs, shape [batch_size, seq_len]
            tool_results: Dictionary of tool execution results, keys are batch indices
            
        Returns:
            torch.Tensor: Next turn input IDs, with uniform length
        """
        batch_size = original_inputs.size(0)
        device = original_inputs.device
        max_prompt_length = self.config.prompt_length
        
        # Print debug info
        if tool_results:
            key_types = [type(k) for k in tool_results.keys()]
            print(f"[DEBUG] tool_results keys types: {key_types[:5]}")
        
        # Decode input and response text
        response_text_list = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        tmp_input_text_list = self.tokenizer.batch_decode(original_inputs, skip_special_tokens=False)
        
        # Remove pad tokens
        input_text_list = [
            text.replace(self.tokenizer.decode(self.pad_token_id), '')
            for text in tmp_input_text_list
        ]
        
        assert len(response_text_list) == len(input_text_list), "Responses and inputs length mismatch"
        
        # Create the next turn text list
        next_text_list = []
        for i in range(batch_size):
            if i in tool_results:
                next_text = f"{input_text_list[i]}{response_text_list[i].strip()}\n\n{self.start_state_marker}{tool_results[i]}{self.end_state_marker}\n\n"
            else:
                next_text = f"{input_text_list[i]}{response_text_list[i]}"
            next_text_list.append(next_text)
        
        # Convert to tokens
        next_token_list = self.tokenizer(next_text_list, add_special_tokens=False)["input_ids"]
        next_token_list_tensor = [torch.tensor(seq, dtype=torch.long, device=device) for seq in next_token_list]
        
        # Implement left padding: reverse, right pad, then reverse back
        reversed_inputs = [seq.flip(0) for seq in next_token_list_tensor]
        padded_reversed = rnn_utils.pad_sequence(reversed_inputs, batch_first=True, padding_value=self.pad_token_id)
        padded_inputs = padded_reversed.flip(1).long()  # Back to right padding and ensure long type
        
        # Ensure length meets requirements
        if padded_inputs.shape[1] < max_prompt_length:
            pad_length = max_prompt_length - padded_inputs.shape[1]
            n_rows = padded_inputs.shape[0]
            pad_tokens = torch.full((n_rows, pad_length), self.pad_token_id, dtype=torch.long, device=device)
            padded_inputs = torch.cat([pad_tokens, padded_inputs], dim=-1)
        
        # Log length exceeds but don't truncate in intermediate turns
        if padded_inputs.shape[1] > max_prompt_length:
            print(f"[INFO] Input length exceeds max_prompt_length: {padded_inputs.shape[1]} > {max_prompt_length} (not truncating in intermediate turns)")
        
        return padded_inputs

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Override generate_sequences to use tool-enabled version when tools are enabled.
        
        Args:
            prompts: Input prompts as DataProto.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated responses as DataProto.
        """
        # 打印当前enable_tools状态
        print(f"[DEBUG] generate_sequences - Current enable_tools value: {self.enable_tools}")
        
        # Record input information
        input_batch_size = prompts.batch.batch_size[0] if prompts.batch is not None else None
        input_keys = list(prompts.batch.keys()) if prompts.batch is not None else []
        
        print(f"[DEBUG] generate_sequences - Input batch size: {input_batch_size}")
        print(f"[DEBUG] generate_sequences - Input keys: {input_keys}")

        # Ensure meta_info exists and includes eos_token_id
        if not hasattr(prompts, 'meta_info') or prompts.meta_info is None:
            prompts.meta_info = {}
            
        # If eos_token_id doesn't exist, get it from tokenizer
        if 'eos_token_id' not in prompts.meta_info:
            print("[DEBUG] Adding eos_token_id to meta_info from tokenizer")
            prompts.meta_info['eos_token_id'] = self.tokenizer.eos_token_id

        # Check if validation mode
        is_validate = prompts.meta_info.get('validate', False)
        
        # Log meta info
        print(f"[DEBUG] generate_sequences - Meta info: validate={is_validate}, keys={list(prompts.meta_info.keys())}")
        
        start_time = time.time()
        
        # Check if input is valid
        if prompts.batch is None and len(prompts.non_tensor_batch) == 0:
            print("[ERROR] generate_sequences - Both batch and non_tensor_batch are empty")
            raise ValueError("Invalid input: both batch and non_tensor_batch are empty")
        
        # Use tool-enabled generation if tools are enabled
        if self.enable_tools:
            print(f"[DEBUG] generate_sequences - Using tool-enabled generation (validate={is_validate})")
            result = self.generate_sequences_with_tools(prompts, **kwargs)
        else:
            # Otherwise, use parent class implementation
            print("[DEBUG] generate_sequences - Using standard generation (without tools)")
            result = super().generate_sequences(prompts, **kwargs)
            
            # Ensure all tensors are long type to avoid float type errors
            if result.batch is not None:
                for key in list(result.batch.keys()):
                    if torch.is_tensor(result.batch[key]) and result.batch[key].dtype != torch.bool:
                        if key in ['input_ids', 'responses', 'prompts', 'position_ids']:
                            result.batch[key] = result.batch[key].long()
        
        end_time = time.time()
        
        # Log output information
        output_batch_size = result.batch.batch_size[0] if result.batch is not None else None
        output_keys = list(result.batch.keys()) if result.batch is not None else []
        
        print(f"[DEBUG] generate_sequences - Output batch size: {output_batch_size}")
        print(f"[DEBUG] generate_sequences - Output keys: {output_keys}")
        print(f"[DEBUG] generate_sequences - Generation time: {end_time - start_time:.2f} seconds")
        
        # Check if batch sizes match
        if input_batch_size != output_batch_size:
            print(f"[DEBUG] generate_sequences - BATCH SIZE MISMATCH: Input={input_batch_size}, Output={output_batch_size}")
            # Check each tensor's shape
            for key in output_keys:
                if prompts.batch is not None and key in prompts.batch:
                    input_shape = tuple(prompts.batch[key].shape)
                    output_shape = tuple(result.batch[key].shape)
                    print(f"[DEBUG] generate_sequences - Shape comparison for '{key}': Input={input_shape}, Output={output_shape}")
        
        return result

    @torch.no_grad()
    def generate_sequences_with_tools(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Multi-turn generation with tool integration.
        
        Args:
            prompts: Input prompts as DataProto.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated responses as DataProto with tool results.
        """
        global TRAJECTORY_COUNTER
        device = self.model_device
        
        # Ensure eos_token_id exists in meta_info
        if 'eos_token_id' not in prompts.meta_info:
            prompts.meta_info['eos_token_id'] = self.tokenizer.eos_token_id
        
        # Merge tool parameters
        tool_kwargs = {
            "stop": [self.tool_stop],  # Use configured tool_stop
        }
        
        # Get generation control parameters
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        
        # Process inputs
        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        
        # Log input prompts for sampled trajectories
        batch_size = input_ids.size(0)
        SAMPLED_INDICES = set()
        if TRAJECTORY_COUNTER % TRAJECTORY_SAMPLE_INTERVAL == 0:
            step = max(1, batch_size // TRAJECTORY_SAMPLE_INTERVAL)
            SAMPLED_INDICES = set(range(0, batch_size, step)) if batch_size > 0 else set()
            for i in SAMPLED_INDICES:
                if i < batch_size:
                    input_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    log_tool_info(f"Sampled Initial Query [Index {i}]", input_text)
        TRAJECTORY_COUNTER += 1
        
        # Batch size and sampling
        n_samples = self.sampling_params.n if do_sample else 1
        expected_batch_size = batch_size * n_samples if do_sample and n_samples > 1 else batch_size
        
        print(f"[DEBUG] Original batch size: {batch_size}, Expected final: {expected_batch_size}")
        
        # Special handling for validate mode
        if not do_sample:
            tool_kwargs.update({
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0.0,
                'n': 1  # if greedy, only 1 response
            })
        elif is_validate:
            tool_kwargs.update({
                'top_k': self.config.val_kwargs.top_k if hasattr(self.config.val_kwargs, 'top_k') else -1,
                'top_p': self.config.val_kwargs.top_p if hasattr(self.config.val_kwargs, 'top_p') else 1.0,
                'temperature': self.config.val_kwargs.temperature if hasattr(self.config.val_kwargs, 'temperature') else 1.0,
                'n': 1,  # if validate, already repeat in ray_trainer
            })
            
        # Set parameters
        max_turns = min(self.max_turns, 3) if is_validate else self.max_turns
        max_prompt_length = self.config.prompt_length
        max_response_length = self.config.response_length
        
        # Handle batch repetition
        if n_samples > 1 and do_sample:
            original_inputs = input_ids.repeat_interleave(n_samples, dim=0)
            fixed_size_input_ids = original_inputs.clone()
            attention_mask = attention_mask.repeat_interleave(n_samples, dim=0)
            position_ids = position_ids.repeat_interleave(n_samples, dim=0)
        else:
            original_inputs = input_ids.clone()
            fixed_size_input_ids = original_inputs.clone()
        
        # Track active samples
        active_mask = torch.ones(expected_batch_size, dtype=torch.bool, device=device)
        final_outputs = [None] * expected_batch_size
        final_responses = ['' for _ in range(expected_batch_size)]
        
        # Adjust sampled indices for repeated samples
        adjusted_sampled_indices = set()
        for idx in SAMPLED_INDICES:
            if do_sample and n_samples > 1:
                for rep in range(n_samples):
                    new_idx = idx * n_samples + rep
                    if new_idx < expected_batch_size:
                        adjusted_sampled_indices.add(new_idx)
            else:
                if idx < expected_batch_size:
                    adjusted_sampled_indices.add(idx)
        
        # Multi-turn generation loop
        for turn in range(max_turns):
            if not active_mask.any():
                print(f"[DEBUG] All samples completed at turn {turn}")
                break
            
            # Get active examples
            num_active = active_mask.sum().item()
            print(f"Turn {turn}, active examples: {num_active}/{expected_batch_size}")
            
            # If no active samples, break loop
            if num_active == 0:
                print(f"[DEBUG] No active examples at turn {turn}, breaking early")
                break
                
            # Prepare current turn inputs
            active_prompts = DataProto.from_dict({
                'input_ids': fixed_size_input_ids[active_mask],
                'attention_mask': attention_mask[active_mask],
                'position_ids': position_ids[active_mask],
            })
            active_prompts.meta_info = prompts.meta_info.copy()
            
            # Handle non_tensor_batch
            if prompts.non_tensor_batch:
                active_non_tensor = {}
                for key, val in prompts.non_tensor_batch.items():
                    # Handle different sized non_tensor data
                    try:
                        if len(val) == batch_size and n_samples > 1 and do_sample:
                            # Repeat non_tensor data
                            import numpy as np
                            repeated_val = np.repeat(val, n_samples, axis=0)
                            active_non_tensor[key] = repeated_val[active_mask.cpu().numpy()]
                        elif len(val) == expected_batch_size:
                            active_non_tensor[key] = val[active_mask.cpu().numpy()]
                        else:
                            print(f"[WARNING] Skipping non_tensor_batch key '{key}' due to size mismatch")
                    except Exception as e:
                        print(f"[ERROR] Error processing non_tensor_batch key '{key}': {e}")
                if active_non_tensor:
                    active_prompts.non_tensor_batch = active_non_tensor
            
            # Force n=1 for each turn to avoid dimension explosion
            local_kwargs = kwargs.copy()
            local_kwargs['n'] = 1
            
            # Check batch size, ensure non-empty
            batch_size_check = active_prompts.batch.batch_size[0]
            if batch_size_check == 0:
                print(f"[ERROR] Empty batch detected at turn {turn}, breaking early")
                break
            
            # Handle merged kwargs
            final_kwargs = {**local_kwargs, **tool_kwargs}
            
            # Use parent class method for generation
            try:
                gen_output = super().generate_sequences(active_prompts, **final_kwargs)
            except RuntimeError as e:
                if "CUDA error" in str(e):
                    print(f"[ERROR] CUDA error in generate_sequences: {e}")
                    print("[DEBUG] Active batch info:")
                    print(f"  Batch size: {batch_size_check}")
                    print(f"  Active mask sum: {active_mask.sum().item()}")
                    print(f"  tool_kwargs: {tool_kwargs}")
                    print(f"  local_kwargs: {local_kwargs}")
                    break
                raise  # Re-raise other error types
                
            responses = gen_output.batch['responses']
            
            # Handle shape mismatch
            if responses.shape[0] != num_active:
                print(f"Warning: Expected responses shape[0]={num_active}, got {responses.shape[0]}")
                if responses.shape[0] > num_active:
                    responses = responses[:num_active]
            
            # Create current turn responses tensor
            turn_responses = torch.zeros(
                (expected_batch_size, responses.size(1)), 
                dtype=responses.dtype, 
                device=device
            )
            
            # Place active examples' responses in appropriate positions
            active_indices = torch.where(active_mask)[0]
            for i, idx in enumerate(active_indices):
                if i < responses.size(0):
                    turn_responses[idx] = responses[i]
            
            # Determine sampled indices for active batch
            active_sampled_indices = set(idx for idx in adjusted_sampled_indices if idx < expected_batch_size and active_mask[idx])
            
            # Process tool calls and execute tools
            need_continue, tool_results, turn_outputs = self._process_tools_in_batch(responses, turn, active_sampled_indices)
            
            # Update active mask and collect final outputs
            new_active_indices = []
            pad_token = self.tokenizer.decode(self.pad_token_id)
            for i, (needs_continuation, output) in enumerate(zip(need_continue, turn_outputs)):
                if i >= len(active_indices):
                    continue
                    
                active_idx = active_indices[i]
                if not needs_continuation:
                    # This sample is complete
                    active_mask[active_idx] = False
                    if output is not None:
                        final_outputs[active_idx] = output
                    
                    # Current round doesn't need to enter next round, directly append answer
                    current_response_str = self.tokenizer.decode(responses[i]).replace(pad_token, '')
                    final_responses[active_idx] += current_response_str.strip()
                    
                    if active_idx in adjusted_sampled_indices:
                        log_tool_info(f"Turn {turn} - Sampled Final Response [Index {active_idx}]", final_responses[active_idx])
                else:
                    # Sample needs to continue generation
                    new_active_indices.append(i)
                    if i in tool_results:
                        tool_result = tool_results[i]
                        
                        # Append tool result to current round answer
                        current_response_str = self.tokenizer.decode(responses[i]).replace(pad_token, '')
                        final_responses[active_idx] += f"{current_response_str.strip()}\n\n{self.start_state_marker}{tool_result}{self.end_state_marker}\n\n"
                        
                        if active_idx in adjusted_sampled_indices:
                            log_tool_info(f"Turn {turn} - Sampled Updated Context [Index {active_idx}] (continues to next turn)", final_responses[active_idx])
            
            # Check if any active samples remain
            if not active_mask.any():
                print(f"[DEBUG] No more active samples after processing turn {turn}")
                continue
                
            # If we need to continue, prepare next turn inputs
            if turn < max_turns - 1:
                # Ensure we have new active indices
                if not new_active_indices:
                    print(f"[DEBUG] No new active indices at turn {turn}, stopping early")
                    break
                    
                # Create next turn inputs with tool results
                active_tool_results = {
                    i: tool_results[new_active_indices[i]] 
                    for i in range(len(new_active_indices)) 
                    if new_active_indices[i] in tool_results
                }
                
                # Create next turn inputs
                try:
                    next_inputs = self._create_next_turn_inputs(
                        fixed_size_input_ids[active_mask],
                        turn_responses[active_mask],
                        active_tool_results
                    )
                    
                    # Update inputs
                    fixed_size_input_ids_clone = fixed_size_input_ids.clone()
                    for i, idx in enumerate(torch.where(active_mask)[0]):
                        if i < next_inputs.shape[0]:
                            fixed_size_input_ids_clone[idx] = next_inputs[i]
                    
                    fixed_size_input_ids = fixed_size_input_ids_clone
                    
                except Exception as e:
                    print(f"Error preparing next turn inputs: {e}")
                    print("Keeping original inputs due to error")
                    import traceback
                    traceback.print_exc()
        
        # Process final responses
        final_responses_token_list = self.tokenizer(final_responses, add_special_tokens=False)["input_ids"]
        final_responses_token_tensor = [torch.tensor(seq, dtype=torch.long, device=device) for seq in final_responses_token_list]
        
        # Align response lengths
        combined_responses = rnn_utils.pad_sequence(final_responses_token_tensor, batch_first=True, padding_value=self.pad_token_id)
        
        # Ensure long dtype for token IDs
        combined_responses = combined_responses.long()
        
        # Final output stage truncation strategy
        if combined_responses.shape[1] < max_response_length:
            # If length is less than required, pad to fixed length
            pad_length = max_response_length - combined_responses.shape[1]
            n_rows = combined_responses.shape[0]
            pad_tokens = torch.full((n_rows, pad_length), self.pad_token_id, dtype=torch.long, device=device)
            combined_responses = torch.cat([combined_responses, pad_tokens], dim=1)
        elif combined_responses.size(1) > max_response_length:
            # Only truncate in final stage, keep rightmost newest content
            print(f"[FINAL STAGE] Truncating combined responses from {combined_responses.size(1)} to {max_response_length}")
            # Default to keep from right, truncate left
            combined_responses = combined_responses[:, -max_response_length:]
        
        # Create output mask
        try:
            output_mask = self._create_output_mask(combined_responses, self.start_state_marker, self.end_state_marker)
            mask_ratio = output_mask.float().mean().item()
            print(f"[DEBUG] output_mask created - Masked token ratio: {mask_ratio:.4f}")
        except Exception as e:
            print(f"Error creating output mask: {e}")
            output_mask = torch.zeros_like(combined_responses, dtype=torch.bool, device=device)
        
        # Create attention_mask and position_ids
        attention_mask = (
            torch.cat([original_inputs.long(), combined_responses], dim=1) == self.pad_token_id
        ).int()
        
        position_ids = self.transform_tensor_v2(attention_mask)
        attention_mask = 1 - attention_mask
        
        # Process output_mask
        # Note: output_mask True indicates output regions (not trainable)
        output_mask = ((~output_mask) & attention_mask[:, -output_mask.shape[1]:]).bool()
        
        # Build final output
        final_output = {
            'prompts': original_inputs.long(),
            'responses': combined_responses,
            'input_ids': torch.cat([original_inputs.long(), combined_responses], dim=1),
            'attention_mask': attention_mask,
            'position_ids': position_ids.long(),
            'output_mask': output_mask
        }
        
        # Handle non-tensor data
        non_tensor_batch = {}
        if prompts.non_tensor_batch:
            import numpy as np
            for key, val in prompts.non_tensor_batch.items():
                # Ensure non-tensor data matches expected batch size
                if len(val) != expected_batch_size:
                    if len(val) == batch_size and n_samples > 1 and do_sample:
                        non_tensor_batch[key] = np.repeat(val, n_samples, axis=0)
                    else:
                        print(f"[WARNING] Non-tensor batch size mismatch for key '{key}': {len(val)} vs {expected_batch_size}")
                        # Try to adjust size
                        if len(val) > expected_batch_size:
                            non_tensor_batch[key] = val[:expected_batch_size]
                        elif len(val) < expected_batch_size and len(val) > 0:
                            # Repeat last element to fill
                            padding = np.repeat(val[-1:], expected_batch_size - len(val), axis=0)
                            non_tensor_batch[key] = np.concatenate([val, padding], axis=0)
                else:
                    non_tensor_batch[key] = val
        
        # Create result
        result = DataProto.from_dict(tensors=final_output, non_tensors=non_tensor_batch, meta_info=prompts.meta_info.copy())
        
        # Ensure final_outputs are all strings to avoid float type issues
        sanitized_outputs = []
        for output in final_outputs:
            if output is None:
                sanitized_outputs.append("")
            else:
                sanitized_outputs.append(str(output))
                
        result.meta_info['final_outputs'] = sanitized_outputs
        
        # Ensure all ID tensors are long type
        for key in ['input_ids', 'responses', 'prompts', 'position_ids']:
            if key in result.batch:
                result.batch[key] = result.batch[key].long()
        
        # Log final shapes
        print(f"[DEBUG] Final output shapes:")
        for key, value in final_output.items():
            print(f"  {key}: {value.shape}")
        
        return result

    def _create_output_mask(self, responses, start_marker, end_marker):
        """
        Create output region mask, marking regions between start_marker and end_marker
        
        Args:
            responses: Response tensor [batch_size, seq_len]
            start_marker: Start marker (e.g., self.start_state_marker)
            end_marker: End marker (e.g., self.end_state_marker)
            
        Returns:
            torch.Tensor: Mask tensor, True indicates output regions (not trainable)
        """
        # If state masking is disabled, return all-False mask (all tokens trainable)
        if not self.use_state_masking:
            batch_size, seq_len = responses.size()
            device = responses.device
            return torch.zeros_like(responses, dtype=torch.bool, device=device)
            
        batch_size, seq_len = responses.size()
        device = responses.device
        output_mask = torch.zeros_like(responses, dtype=torch.bool, device=device)
        
        # Ensure responses are long tensors before decoding
        responses_long = responses.long()
        
        # Decode responses to find markers
        response_texts = self.tokenizer.batch_decode(responses_long, skip_special_tokens=False)
        
        for i, text in enumerate(response_texts):
            # Find all output regions
            try:
                start_positions = [m.start() for m in re.finditer(re.escape(start_marker), text)]
                end_positions = [m.start() for m in re.finditer(re.escape(end_marker), text)]
                
                # Ensure markers appear in pairs
                num_regions = min(len(start_positions), len(end_positions))
                
                if num_regions == 0:
                    continue
                
                # Create mask for each region
                for j in range(num_regions):
                    start_pos = start_positions[j]
                    # Find corresponding end position (first end marker after start)
                    valid_ends = [pos for pos in end_positions if pos > start_pos]
                    if not valid_ends:
                        continue
                    end_pos = min(valid_ends)
                    
                    # Find corresponding token indices
                    start_token_idx = len(self.tokenizer.encode(text[:start_pos], add_special_tokens=False))
                    end_token_idx = len(self.tokenizer.encode(text[:end_pos + len(end_marker)], add_special_tokens=False))
                    
                    # Set mask, ensure indices are in range
                    start_idx = min(start_token_idx, seq_len-1)
                    end_idx = min(end_token_idx, seq_len)
                    if start_idx < end_idx:
                        output_mask[i, start_idx:end_idx] = True
            except Exception as e:
                print(f"Error creating output mask for sample {i}: {e}")
                continue
        
        # Print statistics
        mask_ratio = output_mask.float().mean().item()
        print(f"[DEBUG] Output mask statistics: {mask_ratio:.4f} of tokens are marked as output (not trainable)")
        
        return output_mask

    def transform_tensor_v2(self, tensor: torch.Tensor):
        """
        Transform attention_mask tensor to position_ids
        For each row, find positions of 0-value elements and generate appropriate position indices
        """
        result = torch.zeros_like(tensor, dtype=torch.long)
        num_cols = tensor.size(1)

        for i, row in enumerate(tensor):
            zero_mask = (row == 0)
            if zero_mask.any():
                start = zero_mask.nonzero(as_tuple=True)[0][0].item()
                count = zero_mask.sum().item()
                result[i, start:] = torch.arange(0, num_cols - start, device=tensor.device, dtype=torch.long)
        return result 