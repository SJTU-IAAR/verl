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
The search enabled vllm_rollout_spmd that extends the standard vllm_rollout_spmd
with search capabilities during generation.

This version is specifically designed to work with the SPMD variant of vLLM rollout,
inheriting its parameter structure and adding search functionality.
"""
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
import re
import requests
import time
import torch.nn.utils.rnn as rnn_utils

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout, _pre_process_inputs, _repeat_interleave
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version


class HTTPSearchClient:
    """HTTP client for remote search API calls"""
    
    def __init__(self, search_url, topk=3):
        self.search_url = search_url
        self.topk = topk
        
    def batch_search(self, queries: List[str]):
        """Call remote search API and return search results"""
        if not queries:
            return []
            
        payload = {
            "queries": queries,
            "topk": self.topk,
            "return_scores": True
        }
        
        try:
            response = requests.post(self.search_url, json=payload, timeout=10).json()
            return response["result"]
        except Exception as e:
            print(f"Search API error: {e}")
            # 返回空结果作为后备
            return [[] for _ in range(len(queries))]
        
    def format_search_results(self, results):
        """Format search results as readable text"""
        formatted_results = []
        
        for result_set in results:
            format_reference = ''
            for idx, doc_item in enumerate(result_set):
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            formatted_results.append(format_reference)
            
        return formatted_results


class SearchEnabledVLLMRollout(vLLMRollout):
    """vLLM rollout with search capabilities during generation"""
    
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A search-enabled vLLM rollout based on vLLMRollout from vllm_rollout_spmd.py

        Args:
            model_path: Path to the model
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        # Configure CUDA memory allocator to avoid conflicts with vLLM
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False,backend:native'
        print(f"[DEBUG] CUDA Allocator Config Set: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
        
        # Initialize parent class first
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        
        # Save tokenizer as instance attribute
        self.tokenizer = tokenizer
        
        # Set search-related configuration
        self.enable_search = config.get('enable_search', False)
        self.search_url = config.get('search_url', 'http://localhost:8000/retrieve')
        self.search_topk = config.get('search_topk', 3)
        self.max_turns = config.get('max_turns', 5)
        self.search_stop = config.get('search_stop', "</search>")
        
        # State masking configuration for handling information sections
        self.use_state_masking = config.get('state_masking', True)
        state_masking_config = config.get('state_masking', {})
        if isinstance(state_masking_config, dict):
            self.start_state_marker = state_masking_config.get('start_state_marker', "<information>")
            self.end_state_marker = state_masking_config.get('end_state_marker', "</information>")
        else:
            self.start_state_marker = "<information>"
            self.end_state_marker = "</information>"
        
        # Initialize search client
        self.search_client = HTTPSearchClient(
            search_url=self.search_url,
            topk=self.search_topk
        )
        
        # Configure regex patterns for search and answer extraction
        self.search_pattern = r'<search>(.*?)</search>'
        self.answer_pattern = r'<answer>(.*?)</answer>'
        
        # Device configuration
        self.model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Configure sampling parameters
        self.sampling_params.stop = [self.search_stop]
        self.sampling_params.ignore_eos = False
        if hasattr(self.sampling_params, 'detokenize') and vllm_version != '0.3.1':
            self.sampling_params.detokenize = True

    def _process_search_in_batch(self, responses):
        """
        Process a batch of responses to identify and execute searches
        
        Args:
            responses: Tensor containing response token IDs
        
        Returns:
            Tuple of (need_continue_flags, search_results, final_answers)
        """
        batch_size = responses.size(0)
        
        # Ensure responses is in the correct format for tokenizer
        if responses.dtype != torch.long:
            responses = responses.to(dtype=torch.long)
        
        # Decode responses to text
        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        
        # Process each response to find search queries or answers
        search_queries = []
        need_continue = [False] * batch_size
        search_results_mapping = {}  # Maps batch index to search result position
        final_answers = [None] * batch_size
        
        for i, text in enumerate(response_texts):
            search_match = re.search(self.search_pattern, text, re.DOTALL)
            answer_match = re.search(self.answer_pattern, text, re.DOTALL)
            
            if search_match:
                # Extract search query and add to batch
                query = search_match.group(1).strip()
                search_queries.append(query)
                search_results_mapping[i] = len(search_queries) - 1
                need_continue[i] = True
            elif answer_match:
                # Extract final answer
                final_answers[i] = answer_match.group(1).strip()
                need_continue[i] = False
            else:
                # No valid operation found
                need_continue[i] = False
        
        # Perform batch search for all queries
        search_results = {}
        if search_queries:
            try:
                results = self.search_client.batch_search(search_queries)
                formatted_results = self.search_client.format_search_results(results)
                
                # Map results back to original batch indices
                for batch_idx, result_idx in search_results_mapping.items():
                    if result_idx < len(formatted_results):
                        search_results[int(batch_idx)] = formatted_results[result_idx]
            except Exception as e:
                print(f"Error during search: {e}")
                # Provide a default response
                for batch_idx in search_results_mapping:
                    search_results[int(batch_idx)] = "No search results found due to an error."
        
        return need_continue, search_results, final_answers
    
    def _create_next_turn_inputs(
            self, 
            original_inputs: torch.Tensor, 
            responses: torch.Tensor, 
            search_results: Dict[int, str]
        ):
        """Create inputs for the next conversation turn, incorporating search results
        
        Args:
            original_inputs: Original input token IDs [batch_size, seq_len]
            responses: Response token IDs [batch_size, seq_len]
            search_results: Dictionary mapping batch indices to search result text
            
        Returns:
            torch.Tensor: Next turn input token IDs with uniform length
        """
        batch_size = original_inputs.size(0)
        device = original_inputs.device
        max_prompt_length = self.config.prompt_length
        
        # Ensure inputs are in long format for tokenizer
        responses = responses.to(dtype=torch.long)
        original_inputs = original_inputs.to(dtype=torch.long)
        
        # Decode token sequences to text
        response_text_list = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        input_text_list_raw = self.tokenizer.batch_decode(original_inputs, skip_special_tokens=False)
        
        # Remove padding tokens from input text
        input_text_list = [
            text.replace(self.tokenizer.decode(self.pad_token_id), '')
            for text in input_text_list_raw
        ]
        
        # Verify lengths match
        assert len(response_text_list) == len(input_text_list), "Response and input lists must have the same length"
        
        # Create next turn text by combining input, response, and search results
        next_text_list = []
        for i in range(batch_size):
            if i in search_results:
                next_text = f"{input_text_list[i]}{response_text_list[i].strip()}\n\n<information>{search_results[i]}</information>\n\n"
            else:
                next_text = f"{input_text_list[i]}{response_text_list[i]}"
            next_text_list.append(next_text)
        
        # Convert to tokens
        next_token_list = self.tokenizer(next_text_list, add_special_tokens=False)["input_ids"]
        next_token_list_tensor = [torch.tensor(seq, device=device) for seq in next_token_list]
        
        # Implement left padding: reverse, right-pad, then reverse back
        reversed_inputs = [seq.flip(0) for seq in next_token_list_tensor]
        padded_reversed = rnn_utils.pad_sequence(reversed_inputs, batch_first=True, padding_value=self.pad_token_id)
        padded_inputs = padded_reversed.flip(1)
        
        # Ensure length meets requirements
        if padded_inputs.shape[1] < max_prompt_length:
            # Add left padding if needed
            pad_length = max_prompt_length - padded_inputs.shape[1]
            pad_tokens = torch.full((batch_size, pad_length), self.pad_token_id, dtype=torch.long, device=device)
            padded_inputs = torch.cat([pad_tokens, padded_inputs], dim=-1)
        elif padded_inputs.shape[1] > max_prompt_length:
            # Truncate from left to keep maximum context if too long
            padded_inputs = padded_inputs[:, -max_prompt_length:]
        
        return padded_inputs

    @torch.no_grad()
    def generate_sequences_with_search(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences with multi-turn search interaction
        
        Args:
            prompts: Input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            DataProto: Generated responses with search information
        """
        # For vLLM versions > 0.6.3, cache engine initialization is handled automatically
        # The old init_cache_engine method no longer exists in vLLM 0.8.3+
        # Remove the problematic code that calls init_cache_engine
            
        # Extract inputs and metadata
        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        device = input_ids.device
        
        # Ensure meta_info exists and contains eos_token_id
        if not hasattr(prompts, 'meta_info') or prompts.meta_info is None:
            prompts.meta_info = {}
        
        if 'eos_token_id' not in prompts.meta_info:
            prompts.meta_info['eos_token_id'] = self.tokenizer.eos_token_id
        
        # Get generation parameters
        eos_token_id = prompts.meta_info["eos_token_id"]
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        
        # Set up sampling parameters
        n_samples = self.sampling_params.n if do_sample else 1
        batch_size = input_ids.size(0)
        expected_final_batch_size = batch_size * n_samples if do_sample and n_samples > 1 else batch_size
        
        search_kwargs = {}
        if not do_sample:
            search_kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1
            }
        elif is_validate:
            search_kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,
            }
        
        # Configure maximum turns
        max_turns = min(self.max_turns, 3) if is_validate else self.max_turns
        max_prompt_length = self.config.prompt_length
        max_response_length = self.config.response_length
        
        # Handle n_samples > 1 by repeating inputs
        if n_samples > 1 and do_sample:
            original_inputs = input_ids.repeat_interleave(n_samples, dim=0)
            fixed_size_input_ids = original_inputs.clone()
            attention_mask = attention_mask.repeat_interleave(n_samples, dim=0)
            position_ids = position_ids.repeat_interleave(n_samples, dim=0)
        else:
            original_inputs = input_ids.clone()
            fixed_size_input_ids = original_inputs.clone()
        
        # Track which samples need further processing
        active_mask = torch.ones(expected_final_batch_size, dtype=torch.bool, device=device)
        final_answers = [None] * expected_final_batch_size
        final_responses = ['' for _ in range(expected_final_batch_size)]
        
        # Main generation loop
        for turn in range(max_turns):
            if not active_mask.any():
                print(f"[DEBUG] All samples completed at turn {turn}")
                break
            
            num_active = active_mask.sum().item()
            print(f"Turn {turn}, active examples: {num_active}/{expected_final_batch_size}")
            
            # Prepare active prompts
            active_prompts = DataProto.from_dict({
                'input_ids': fixed_size_input_ids[active_mask],
                'attention_mask': attention_mask[active_mask],
                'position_ids': position_ids[active_mask],
            })
            active_prompts.meta_info = prompts.meta_info.copy()
            
            # Process non-tensor batch items
            active_non_tensor = {}
            for key, val in prompts.non_tensor_batch.items():
                try:
                    if len(val) == batch_size and n_samples > 1 and do_sample:
                        # Handle repeating non-tensor data
                        repeated_val = np.repeat(val, n_samples, axis=0)
                        active_non_tensor[key] = repeated_val[active_mask.cpu().numpy()]
                    else:
                        print(f"[WARNING] Skipping non_tensor_batch key '{key}' due to size mismatch")
                except Exception as e:
                    print(f"[ERROR] Processing non_tensor_batch key '{key}': {e}")
            active_prompts.non_tensor_batch = active_non_tensor
            
            # Force n=1 for each turn to avoid dimension explosion
            local_kwargs = kwargs.copy()
            local_kwargs['n'] = 1
            
            # Call parent class's generate_sequences for this turn
            gen_output = super().generate_sequences(active_prompts, **{**local_kwargs, **search_kwargs})
            responses = gen_output.batch['responses']
            
            # Ensure response dimensions are correct
            if responses.shape[0] != num_active:
                print(f"Warning: Expected responses shape[0]={num_active}, got {responses.shape[0]}")
                if responses.shape[0] > num_active:
                    responses = responses[:num_active]
            
            # Create response tensor for this turn
            turn_responses = torch.zeros(
                (expected_final_batch_size, responses.size(1)), 
                dtype=responses.dtype, 
                device=device
            )
            
            # Place active responses in appropriate positions
            active_indices = torch.where(active_mask)[0]
            for i, idx in enumerate(active_indices):
                if i < responses.size(0):
                    turn_responses[idx] = responses[i]
            
            # Process search queries and execute searches
            need_continue, search_results, turn_answers = self._process_search_in_batch(responses)
            
            # Update active mask and collect final answers
            new_active_indices = []
            pad_token = self.tokenizer.decode(self.pad_token_id)
            for i, (needs_continuation, answer) in enumerate(zip(need_continue, turn_answers)):
                if i >= len(active_indices):
                    continue
                    
                active_idx = active_indices[i]
                if not needs_continuation:
                    # This sample is complete
                    active_mask[active_idx] = False
                    if answer is not None:
                        final_answers[active_idx] = answer
                    
                    # Current round doesn't need to continue, append answer directly
                    current_response_str = self.tokenizer.decode(responses[i]).replace(pad_token, '')
                    final_responses[active_idx] += current_response_str.strip()
                else:
                    # Sample needs to continue
                    new_active_indices.append(i)
                    if i in search_results:
                        search_result = search_results[i]
                        
                        # Append search result to the current round's answer
                        current_response_str = self.tokenizer.decode(responses[i]).replace(pad_token, '')
                        final_responses[active_idx] += f"{current_response_str.strip()}\n\n<information>{search_result}</information>\n\n"
            
            # Prepare for next turn if needed
            if active_mask.any() and turn < max_turns - 1:
                # Create inputs with search results for the next turn
                active_search_results = {
                    i: search_results[new_active_indices[i]] 
                    for i in range(len(new_active_indices)) 
                    if new_active_indices[i] in search_results
                }
                
                try:
                    next_inputs = self._create_next_turn_inputs(
                        fixed_size_input_ids[active_mask],
                        turn_responses[active_mask],
                        active_search_results
                    )
                    
                    # Update inputs for active samples
                    fixed_size_input_ids_clone = fixed_size_input_ids.clone()
                    for i, idx in enumerate(torch.where(active_mask)[0]):
                        if i < next_inputs.shape[0]:
                            fixed_size_input_ids_clone[idx] = next_inputs[i]
                    
                    fixed_size_input_ids = fixed_size_input_ids_clone
                    
                except Exception as e:
                    print(f"Error preparing next turn inputs: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Process final responses
        final_responses_token_list = self.tokenizer(final_responses, add_special_tokens=False)["input_ids"]
        final_responses_token_tensor = [torch.tensor(seq, device=device) for seq in final_responses_token_list]
        
        # Align response lengths
        combined_responses = rnn_utils.pad_sequence(final_responses_token_tensor, batch_first=True, padding_value=self.pad_token_id)
        
        # Ensure correct response length
        if combined_responses.shape[1] < max_response_length:
            pad_length = max_response_length - combined_responses.shape[1]
            pad_tokens = torch.full((expected_final_batch_size, pad_length), self.pad_token_id, dtype=torch.long, device=device)
            combined_responses = torch.cat([combined_responses, pad_tokens], dim=1)
        elif combined_responses.size(1) > max_response_length:
            combined_responses = combined_responses[:, :max_response_length]
        
        # Create info mask for masking out retrieved information during training
        try:
            info_mask = self._create_info_mask(combined_responses, self.start_state_marker, self.end_state_marker)
        except Exception as e:
            print(f"Error creating info mask: {e}")
            info_mask = torch.zeros_like(combined_responses, dtype=torch.bool, device=device)
        
        # Create attention_mask and position_ids
        attention_mask = (
            torch.cat([original_inputs, combined_responses], dim=1) == self.pad_token_id
        ).int()
        
        position_ids = self.transform_tensor_v2(attention_mask)
        attention_mask = 1 - attention_mask
        
        # Ensure info_mask has correct format
        info_mask = ((~info_mask) & attention_mask[:, -info_mask.shape[1]:]).bool()
        
        # Build final output
        final_output = {
            'prompts': original_inputs,
            'responses': combined_responses,
            'input_ids': torch.cat([original_inputs, combined_responses], dim=1),
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'info_mask': info_mask
        }
        
        # Handle non-tensor batch data
        non_tensor_batch = {}
        if prompts.non_tensor_batch:
            for key, val in prompts.non_tensor_batch.items():
                if len(val) != expected_final_batch_size:
                    if len(val) == batch_size and n_samples > 1 and do_sample:
                        non_tensor_batch[key] = np.repeat(val, n_samples, axis=0)
                    else:
                        print(f"[WARNING] Non-tensor batch size mismatch for key '{key}': {len(val)} vs {expected_final_batch_size}")
                        if len(val) > expected_final_batch_size:
                            non_tensor_batch[key] = val[:expected_final_batch_size]
                        elif len(val) < expected_final_batch_size and len(val) > 0:
                            padding = np.repeat(val[-1:], expected_final_batch_size - len(val), axis=0)
                            non_tensor_batch[key] = np.concatenate([val, padding], axis=0)
                else:
                    non_tensor_batch[key] = val
        
        # Create and return final result
        result = DataProto.from_dict(tensors=final_output, non_tensors=non_tensor_batch, meta_info=prompts.meta_info.copy())
        result.meta_info['final_answers'] = final_answers
        
        return result
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Main entry point for generating sequences, with or without search
        
        Args:
            prompts: Input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            DataProto: Generated responses
        """
        # Log input information
        input_batch_size = prompts.batch.batch_size[0] if prompts.batch is not None else None
        
        # Ensure meta_info exists and contains eos_token_id
        if not hasattr(prompts, 'meta_info') or prompts.meta_info is None:
            prompts.meta_info = {}
            
        if 'eos_token_id' not in prompts.meta_info:
            prompts.meta_info['eos_token_id'] = self.tokenizer.eos_token_id

        # Check if we're in validation mode
        is_validate = prompts.meta_info.get('validate', False)
        
        start_time = time.time()
        
        # Choose generation method based on search configuration
        if self.enable_search:
            print(f"[DEBUG] Using search-enabled generation (validate={is_validate})")
            result = self.generate_sequences_with_search(prompts, **kwargs)
        else:
            # If search is disabled, use standard generation from parent class
            print("[DEBUG] Using standard generation (without search)")
            result = super().generate_sequences(prompts, **kwargs)
        
        end_time = time.time()
        print(f"[DEBUG] Generation time: {end_time - start_time:.2f} seconds")
        
        # Verify output consistency
        output_batch_size = result.batch.batch_size[0] if result.batch is not None else None
        
        # Log any batch size mismatch
        if input_batch_size != output_batch_size:
            print(f"[WARNING] Batch size mismatch: Input={input_batch_size}, Output={output_batch_size}")
            
            # Debug output for shape comparison
            if prompts.batch is not None and result.batch is not None:
                for key in result.batch.keys():
                    if key in prompts.batch:
                        input_shape = tuple(prompts.batch[key].shape)
                        output_shape = tuple(result.batch[key].shape)
                        print(f"[DEBUG] Shape comparison for '{key}': Input={input_shape}, Output={output_shape}")
        
        return result

    def _create_info_mask(self, responses, start_marker, end_marker):
        """
        Create an information region mask to identify sections between marker tags
        
        Args:
            responses: Response token IDs [batch_size, seq_len]
            start_marker: Start marker string (e.g., "<information>")
            end_marker: End marker string (e.g., "</information>")
            
        Returns:
            torch.Tensor: Mask tensor where True indicates information regions (not for training)
        """
        # If state masking is disabled, return an empty mask (all tokens trainable)
        if not self.use_state_masking:
            batch_size, seq_len = responses.size()
            device = responses.device
            return torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        
        # Decode responses to text
        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        batch_size, seq_len = responses.size()
        device = responses.device
        
        # Initialize mask: True values will correspond to information sections (not for training)
        info_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        
        for i, text in enumerate(response_texts):
            try:
                # Find all information regions
                start_positions = [m.start() for m in re.finditer(re.escape(start_marker), text)]
                end_positions = [m.start() for m in re.finditer(re.escape(end_marker), text)]
                
                # Ensure markers appear in pairs
                if len(start_positions) != len(end_positions):
                    print(f"[WARNING] Mismatched markers for sample {i}: {len(start_positions)} starts, {len(end_positions)} ends")
                    continue
                
                if len(start_positions) == 0:
                    continue
                
                # Convert text positions to token positions
                for start_pos, end_pos in zip(start_positions, end_positions):
                    if start_pos >= end_pos:
                        continue
                    
                    # Find the token index for start and end markers
                    prefix_text = text[:start_pos]
                    prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                    start_idx = len(prefix_tokens)
                    
                    # Find end token index
                    info_text = text[:end_pos]
                    info_tokens = self.tokenizer.encode(info_text, add_special_tokens=False)
                    end_idx = len(info_tokens)
                    
                    # Set mask for information region
                    if start_idx < end_idx:
                        info_mask[i, start_idx:end_idx] = True
            except Exception as e:
                print(f"Error creating info mask for sample {i}: {e}")
                continue
        
        # True values indicate information regions (not for training)
        return info_mask

    def transform_tensor_v2(self, tensor:torch.Tensor):
        """
        转换attention_mask张量为position_ids
        对于每一行，找到0值元素的位置并生成相应的位置索引
        """
        result = torch.zeros_like(tensor)
        num_cols = tensor.size(1)

        for i, row in enumerate(tensor):
            zero_mask = (row == 0)
            if zero_mask.any():
                start = zero_mask.nonzero(as_tuple=True)[0][0].item()
                result[i, start:] = torch.arange(0, num_cols - start, device=tensor.device)
        return result 