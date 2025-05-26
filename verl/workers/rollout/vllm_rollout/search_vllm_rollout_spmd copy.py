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
        # 在调用父类初始化前，确保设置环境变量以禁用expandable_segments
        import os
        
        # 显式设置PYTORCH_CUDA_ALLOC_CONF，避免与内存池冲突
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False,backend:native'
        
        # 记录当前设置用于调试
        print(f"[DEBUG] CUDA Allocator Config Set: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
        
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        
        # 保存tokenizer作为实例属性
        self.tokenizer = tokenizer
        
        # 确保搜索相关配置被保留为OmegaConf类
        self.enable_search = config.get('enable_search', False)
        self.search_url = config.get('search_url', 'http://localhost:8000/retrieve')
        self.search_topk = config.get('search_topk', 3)
        self.max_turns = config.get('max_turns', 5)
        self.search_stop = config.get('search_stop', "</search>")
        
        # 状态掩码配置
        self.use_state_masking = config.get('state_masking', True)
        self.start_state_marker = config.get('start_state_marker', "<information>")
        self.end_state_marker = config.get('end_state_marker', "</information>")
        
        # 初始化搜索客户端
        self.search_client = HTTPSearchClient(
            search_url=self.search_url,
            topk=self.search_topk
        )
        
        # 设置搜索相关配置
        self.search_pattern = r'<search>(.*?)</search>'  # 检测搜索查询的正则表达式模式
        self.answer_pattern = r'<answer>(.*?)</answer>'  # 检测最终答案的正则表达式模式
        
        # 记录设备信息，避免设备不一致问题
        self.model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        ### 设置停止
        self.sampling_params.stop = [self.search_stop]
        self.sampling_params.ignore_eos = False
        if hasattr(self.sampling_params, 'detokenize') and vllm_version != '0.3.1':
            self.sampling_params.detokenize = True

    def _process_search_in_batch(self, responses):
        """
        Process a batch of responses to identify and execute searches
        
        Returns:
            Tuple of (need_continue_flags, search_results, final_answers)
        """
        batch_size = responses.size(0)
        
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
                        search_results[batch_idx] = formatted_results[result_idx]
            except Exception as e:
                print(f"Error during search: {e}")
                # 提供一个默认响应
                for batch_idx in search_results_mapping:
                    search_results[batch_idx] = "No search results found due to an error."
        
        return need_continue, search_results, final_answers
    
    def _create_next_turn_inputs(
            self, 
            original_inputs:torch.Tensor, 
            responses:torch.Tensor, 
            search_results
        ):
        """创建下一轮的输入"""
        batch_size = original_inputs.size(0)
        device = original_inputs.device
        max_prompt_length = self.config.prompt_length
        
        # 解码输入和响应
        response_text_list = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        
        tmp_input_text_list = self.tokenizer.batch_decode(original_inputs, skip_special_tokens=False)
        
        # 移除填充标记
        input_text_list = [
            text.replace(self.tokenizer.decode(self.pad_token_id), '')
            for text in tmp_input_text_list
        ]
        
        assert len(response_text_list) == len(input_text_list)
        
        # 创建下一轮文本
        next_text_list = []
        for i in range(batch_size):
            if i in search_results:
                next_text = f"{input_text_list[i]}{response_text_list[i].strip()}\n\n<information>{search_results[i]}</information>\n\n"
            else:
                next_text = f"{input_text_list[i]}{response_text_list[i]}"
            next_text_list.append(next_text)
        
        # 转换为token
        next_token_list = self.tokenizer(next_text_list, add_special_tokens=False)["input_ids"]
        
        next_token_list_tensor = [torch.tensor(seq, device=device) for seq in next_token_list]
        
        # 实现左填充：先反转，右填充，再反转回来
        reversed_inputs = [seq.flip(0) for seq in next_token_list_tensor]
        padded_reversed = rnn_utils.pad_sequence(reversed_inputs, batch_first=True, padding_value=self.pad_token_id)
        padded_inputs = padded_reversed.flip(1)
        
        # 确保长度符合要求
        if padded_inputs.shape[1] < max_prompt_length:
            pad_length = max_prompt_length - padded_inputs.shape[1]
            n_rows = padded_inputs.shape[0]
            pad_tokens = torch.full((n_rows, pad_length), self.pad_token_id, dtype=torch.long, device=device)
            padded_inputs = torch.cat([pad_tokens, padded_inputs], dim=-1)
        elif padded_inputs.shape[1] > max_prompt_length:
            # 如果长度超过最大值，截断左侧（保留右侧的重要内容）
            padded_inputs = padded_inputs[:, -max_prompt_length:]
        
        return padded_inputs

    @torch.no_grad()
    def generate_sequences_with_search(self, prompts: DataProto, **kwargs) -> DataProto:
        """支持多轮交互搜索的生成方法，始终保持固定大小的输出张量"""
        # 获取初始输入和配置信息
        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        device = input_ids.device
        
        # 确保meta_info存在并包含eos_token_id
        if not hasattr(prompts, 'meta_info') or prompts.meta_info is None:
            prompts.meta_info = {}
        
        # 如果eos_token_id不存在，从tokenizer中获取
        if 'eos_token_id' not in prompts.meta_info:
            print("[DEBUG] Adding eos_token_id to meta_info from tokenizer")
            prompts.meta_info['eos_token_id'] = self.tokenizer.eos_token_id
        
        # 获取采样参数
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        
        # 记录采样参数信息
        print(f"[DEBUG] generate_sequences_with_search - do_sample={do_sample}, n={self.sampling_params.n}")
        
        # 初始批次大小和采样数量
        n_samples = self.sampling_params.n if do_sample else 1
        batch_size = input_ids.size(0)
        expected_final_batch_size = batch_size * n_samples if do_sample and n_samples > 1 else batch_size
        
        print(f"[DEBUG] Original batch size: {batch_size}, Expected final: {expected_final_batch_size}")
        
        # 准备搜索参数
        search_kwargs = {}
        if not do_sample:
            search_kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            search_kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }
        
        # 最大轮次限制
        max_turns = min(self.max_turns, 3) if is_validate else self.max_turns
        max_prompt_length = self.config.prompt_length
        max_response_length = self.config.response_length
        
        # 预处理输入（根据是否do_sample和n）
        if n_samples > 1 and do_sample:
            original_inputs = input_ids.repeat_interleave(n_samples, dim=0)
            fixed_size_input_ids = original_inputs.clone()
            attention_mask = attention_mask.repeat_interleave(n_samples, dim=0)
            position_ids = position_ids.repeat_interleave(n_samples, dim=0)
        else:
            original_inputs = input_ids.clone()
            fixed_size_input_ids = original_inputs.clone()
        
        # 活动掩码跟踪哪些样本需要继续生成
        active_mask = torch.ones(expected_final_batch_size, dtype=torch.bool, device=device)
        final_answers = [None] * expected_final_batch_size
        final_responses = ['' for _ in range(expected_final_batch_size)]
        
        # 多轮生成循环
        for turn in range(max_turns):
            if not active_mask.any():
                print(f"[DEBUG] All samples completed at turn {turn}")
                break
            
            # 获取活跃示例
            num_active = active_mask.sum().item()
            print(f"Turn {turn}, active examples: {num_active}/{expected_final_batch_size}")
            
            # 准备当前轮次的输入
            active_prompts = DataProto.from_dict({
                'input_ids': fixed_size_input_ids[active_mask],
                'attention_mask': attention_mask[active_mask],
                'position_ids': position_ids[active_mask],
            })
            active_prompts.meta_info = prompts.meta_info.copy()
            
            # 处理non_tensor_batch
            if prompts.non_tensor_batch:
                active_non_tensor = {}
                for key, val in prompts.non_tensor_batch.items():
                    if len(val) == batch_size and n_samples > 1 and do_sample:
                        # 处理重复的non_tensor数据
                        repeated_val = np.repeat(val, n_samples, axis=0)
                        active_non_tensor[key] = repeated_val[active_mask.cpu().numpy()]
                    elif len(val) == expected_final_batch_size:
                        active_non_tensor[key] = val[active_mask.cpu().numpy()]
                    else:
                        print(f"[WARNING] Skipping non_tensor_batch key '{key}' due to size mismatch")
                active_prompts.non_tensor_batch = active_non_tensor
            
            # 每轮强制n=1，避免维度爆炸
            local_kwargs = kwargs.copy()
            local_kwargs['n'] = 1
            
            # 使用父类生成响应
            gen_output = super().generate_sequences(active_prompts, **{**local_kwargs, **search_kwargs})
            responses = gen_output.batch['responses']
            
            # 确保响应维度正确
            if responses.shape[0] != num_active:
                print(f"Warning: Expected responses shape[0]={num_active}, got {responses.shape[0]}")
                if responses.shape[0] > num_active:
                    responses = responses[:num_active]
            
            # 创建当前轮次的响应张量
            turn_responses = torch.zeros(
                (expected_final_batch_size, responses.size(1)), 
                dtype=responses.dtype, 
                device=device
            )
            
            # 将活跃示例的响应放入适当位置
            active_indices = torch.where(active_mask)[0]
            for i, idx in enumerate(active_indices):
                if i < responses.size(0):
                    turn_responses[idx] = responses[i]
            
            # 处理搜索查询并执行搜索
            need_continue, search_results, turn_answers = self._process_search_in_batch(responses)
            
            # 更新活跃掩码和收集最终答案
            new_active_indices = []
            pad_token = self.tokenizer.decode(self.pad_token_id)
            for i, (needs_continuation, answer) in enumerate(zip(need_continue, turn_answers)):
                if i >= len(active_indices):
                    continue
                    
                active_idx = active_indices[i]
                if not needs_continuation:
                    # 该样本已完成
                    active_mask[active_idx] = False
                    if answer is not None:
                        final_answers[active_idx] = answer
                    
                    # 当前round不需要进入下一个round，直接拼接答案
                    current_response_str = self.tokenizer.decode(responses[i]).replace(pad_token, '')
                    final_responses[active_idx] += current_response_str.strip()
                else:
                    # 样本需要继续生成
                    new_active_indices.append(i)
                    if i in search_results:
                        search_result = search_results[i]
                        
                        # 将search result拼接得到当前round的答案
                        current_response_str = self.tokenizer.decode(responses[i]).replace(pad_token, '')
                        final_responses[active_idx] += f"{current_response_str.strip()}\n\n<information>{search_result}</information>\n\n"
            
            # 如果需要继续，准备下一轮输入
            if active_mask.any() and turn < max_turns - 1:
                # 创建带搜索结果的下一轮输入
                active_search_results = {
                    i: search_results[new_active_indices[i]] 
                    for i in range(len(new_active_indices)) 
                    if new_active_indices[i] in search_results
                }
                
                # 创建下一轮输入
                try:
                    next_inputs = self._create_next_turn_inputs(
                        fixed_size_input_ids[active_mask],
                        turn_responses[active_mask],
                        active_search_results
                    )
                    
                    # 更新输入
                    fixed_size_input_ids_clone = fixed_size_input_ids.clone()
                    for i, idx in enumerate(torch.where(active_mask)[0]):
                        if i < next_inputs.shape[0]:
                            fixed_size_input_ids_clone[idx] = next_inputs[i]
                    
                    fixed_size_input_ids = fixed_size_input_ids_clone
                    
                except Exception as e:
                    print(f"Error preparing next turn inputs: {e}")
                    print("Keeping original inputs due to error")
        
        # 处理最终响应
        final_responses_token_list = self.tokenizer(final_responses, add_special_tokens=False)["input_ids"]
        final_responses_token_tensor = [torch.tensor(seq, device=device) for seq in final_responses_token_list]
        
        # 对齐响应长度
        combined_responses = rnn_utils.pad_sequence(final_responses_token_tensor, batch_first=True, padding_value=self.pad_token_id)
        
        # 确保不超过最大响应长度
        if combined_responses.shape[1] < max_response_length:
            pad_length = max_response_length - combined_responses.shape[1]
            n_rows = combined_responses.shape[0]
            pad_tokens = torch.full((n_rows, pad_length), self.pad_token_id, dtype=torch.long, device=device)
            combined_responses = torch.cat([combined_responses, pad_tokens], dim=-1)
        elif combined_responses.size(1) > max_response_length:
            print(f"Truncating combined responses from {combined_responses.size(1)} to {max_response_length}")
            combined_responses = combined_responses[:, :max_response_length]
        
        # 创建信息掩码
        try:
            start_marker = self.start_state_marker
            end_marker = self.end_state_marker
            info_mask = self._create_info_mask(combined_responses, start_marker, end_marker)
        except Exception as e:
            print(f"Error creating info mask: {e}")
            info_mask = torch.zeros_like(combined_responses, dtype=torch.bool, device=device)
        
        # 创建attention_mask和position_ids
        attention_mask = (
            torch.cat([original_inputs, combined_responses], dim=1) == self.pad_token_id
        ).int()
        
        position_ids = self.transform_tensor_v2(attention_mask)
        attention_mask = 1 - attention_mask
        
        # 处理info_mask
        # 注意这里把info_mask转换为与输出要求一致的格式：
        # True代表可训练的token（非信息区），False代表信息区（不训练）
        info_mask = ((~info_mask) & attention_mask[:, -info_mask.shape[1]:]).bool()
        
        # 构建最终输出
        final_output = {
            'prompts': original_inputs,
            'responses': combined_responses,
            'input_ids': torch.cat([original_inputs, combined_responses], dim=1),
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'info_mask': info_mask
        }
        
        # 处理非张量数据
        non_tensor_batch = {}
        if prompts.non_tensor_batch:
            for key, val in prompts.non_tensor_batch.items():
                # 确保非张量数据与期望的批次大小匹配
                if len(val) != expected_final_batch_size:
                    if len(val) == batch_size and n_samples > 1 and do_sample:
                        non_tensor_batch[key] = np.repeat(val, n_samples, axis=0)
                    else:
                        print(f"[WARNING] Non-tensor batch size mismatch for key '{key}': {len(val)} vs {expected_final_batch_size}")
                        # 尝试调整大小
                        if len(val) > expected_final_batch_size:
                            non_tensor_batch[key] = val[:expected_final_batch_size]
                        elif len(val) < expected_final_batch_size and len(val) > 0:
                            # 重复最后一个元素填充
                            padding = np.repeat(val[-1:], expected_final_batch_size - len(val), axis=0)
                            non_tensor_batch[key] = np.concatenate([val, padding], axis=0)
                else:
                    non_tensor_batch[key] = val
        
        # 创建结果
        result = DataProto.from_dict(tensors=final_output, non_tensors=non_tensor_batch, meta_info=prompts.meta_info.copy())
        result.meta_info['final_answers'] = final_answers
        
        # 记录最终形状
        print(f"[DEBUG] Final output shapes:")
        for key, value in final_output.items():
            print(f"  {key}: {value.shape}")
        
        return result
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Override the generate_sequences method to use the search-enabled version
        """
        # 记录输入信息
        input_batch_size = prompts.batch.batch_size[0] if prompts.batch is not None else None
        input_keys = list(prompts.batch.keys()) if prompts.batch is not None else []
        
        print(f"[DEBUG] generate_sequences - Input batch size: {input_batch_size}")
        print(f"[DEBUG] generate_sequences - Input keys: {input_keys}")

        # 确保meta_info存在并包含eos_token_id
        if not hasattr(prompts, 'meta_info') or prompts.meta_info is None:
            prompts.meta_info = {}
            
        # 如果eos_token_id不存在，从tokenizer中获取
        if 'eos_token_id' not in prompts.meta_info:
            print("[DEBUG] Adding eos_token_id to meta_info from tokenizer")
            prompts.meta_info['eos_token_id'] = self.tokenizer.eos_token_id

        # 检查是否是验证模式
        is_validate = prompts.meta_info.get('validate', False)
        
        # 记录元信息
        print(f"[DEBUG] generate_sequences - Meta info: validate={is_validate}, keys={list(prompts.meta_info.keys())}")
        
        start_time = time.time()
        
        # 无论是验证还是训练，只要启用了搜索就使用搜索
        if self.enable_search:
            print(f"[DEBUG] generate_sequences - Using search-enabled generation (validate={is_validate})")
            result = self.generate_sequences_with_search(prompts, **kwargs)
        else:
            # 否则，使用父类实现
            print("[DEBUG] generate_sequences - Using standard generation (without search)")
            result = super().generate_sequences(prompts, **kwargs)
        
        end_time = time.time()
        
        # 记录输出信息
        output_batch_size = result.batch.batch_size[0] if result.batch is not None else None
        output_keys = list(result.batch.keys()) if result.batch is not None else []
        
        print(f"[DEBUG] generate_sequences - Output batch size: {output_batch_size}")
        print(f"[DEBUG] generate_sequences - Output keys: {output_keys}")
        print(f"[DEBUG] generate_sequences - Generation time: {end_time - start_time:.2f} seconds")
        
        # 检查批次大小是否匹配
        if input_batch_size != output_batch_size:
            print(f"[DEBUG] generate_sequences - BATCH SIZE MISMATCH: Input={input_batch_size}, Output={output_batch_size}")
            # 检查每个张量的形状
            for key in output_keys:
                if key in prompts.batch:
                    input_shape = tuple(prompts.batch[key].shape)
                    output_shape = tuple(result.batch[key].shape)
                    print(f"[DEBUG] generate_sequences - Shape comparison for '{key}': Input={input_shape}, Output={output_shape}")
        
        return result

    def _create_info_mask(self, responses, start_marker, end_marker):
        """
        创建信息区域掩码，标记<information>和</information>之间的内容
        """
        # 如果没有启用状态掩码，返回全部为True的掩码（不过滤任何token）
        if not self.use_state_masking:
            batch_size, seq_len = responses.size()
            device = responses.device
            return torch.ones_like(responses, dtype=torch.bool, device=device)
            
        batch_size, seq_len = responses.size()
        device = responses.device
        info_mask = torch.zeros_like(responses, dtype=torch.bool, device=device)
        
        # 解码响应以查找标记
        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=False)
        
        for i, text in enumerate(response_texts):
            # 找到所有信息区域
            start_positions = [m.start() for m in re.finditer(re.escape(start_marker), text)]
            end_positions = [m.start() for m in re.finditer(re.escape(end_marker), text)]
            
            # 确保标记成对出现
            num_regions = min(len(start_positions), len(end_positions))
            
            if num_regions == 0:
                continue
            
            # 对每个区域创建掩码
            for j in range(num_regions):
                start_pos = start_positions[j]
                # 找到对应的结束位置（在开始位置之后的第一个结束标记）
                valid_ends = [pos for pos in end_positions if pos > start_pos]
                if not valid_ends:
                    continue
                end_pos = min(valid_ends)
                
                # 找到对应的token索引
                start_token_idx = len(self.tokenizer.encode(text[:start_pos], add_special_tokens=False))
                end_token_idx = len(self.tokenizer.encode(text[:end_pos + len(end_marker)], add_special_tokens=False))
                
                # 设置掩码，确保索引不超出范围
                start_idx = min(start_token_idx, seq_len-1)
                end_idx = min(end_token_idx, seq_len)
                if start_idx < end_idx:
                    info_mask[i, start_idx:end_idx] = True
        
        return ~info_mask  # 返回反转的掩码：True表示需要训练的token，False表示信息区域（不参与训练） 

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
                count = zero_mask.sum().item()
                result[i, start:] = torch.arange(0, num_cols - start, device=tensor.device)
        return result 