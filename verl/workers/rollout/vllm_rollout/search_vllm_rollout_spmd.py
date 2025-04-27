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
The vllm_rollout with search capabilities during generation.
"""
import re
import torch
import requests
import time
from typing import List, Dict, Any, Tuple
from contextlib import contextmanager
from omegaconf import DictConfig
import torch.distributed
from tensordict import TensorDict
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from verl import DataProto
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
import time

def search_documents(queries:list, top_n: int = 5, threshold: float = 0.8,type = "wiki"):

    if type == "wiki":
        url = "http://192.168.239.37:30080/batch_search"
    elif type == "book":
        url = "http://192.168.239.37:30010/batch_search"
    payload = {
        "query": queries,
        "top_n": top_n,
        "return_score": True
    }


    try:
        # 发送 POST 请求
        response = requests.post(url, json=payload,timeout=3)
        response.raise_for_status()  # 检查请求是否成功

        # 获取结果
        results = response.json()

        # 筛除低于阈值的结果
        documents, scores = results
        filtered_documents = []
        for idx,answer_list in enumerate(documents):
            score = scores[idx]
            filtered_answer_list = []
            for i,answer in enumerate(answer_list):
                if score[i] >= threshold:
                    filtered_answer_list.append(answer)
            answer_list = filtered_answer_list
            filtered_documents.append(answer_list)
        return filtered_documents

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

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
        """A vLLM rollout with search capabilities.

        Args:
            actor_module: module that follows huggingface APIs
            config: DictConfig with search configuration
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine
        """
        # 保存tokenizer和相关token ID之前初始化flash_rag配置
        self.flash_rag = config.get('flash_rag', True)
        self.search_documents = config.get('search_documents', False)
        
        # 确保搜索相关配置被保留
        self.enable_search = config.get('enable_search', False)
        self.search_url = config.get('search_url', 'http://localhost:8000/retrieve')
        self.search_topk = config.get('search_topk', 3)
        
        # 初始化flash_rag的特殊URL
        if self.flash_rag:
            self.search_url = config.get('search_url', 'http://192.168.175.87:8014/batch_search')
            print(f"Using Flash RAG search mode with URL: {self.search_url}")
        
        # 先初始化父类
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        
        # 保存tokenizer和相关token ID，确保一致性
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        print(f"[DEBUG] Using pad_token_id: {self.pad_token_id}")
        
        self.max_turns = config.get('max_turns', 5)
        self.search_stop = config.get('search_stop', "</search>")
        
        # 状态掩码配置
        self.use_state_masking = config.get('state_masking', True)
        state_masking_config = config.get('state_masking', {})
        if isinstance(state_masking_config, dict):
            self.start_state_marker = state_masking_config.get('start_state_marker', "<information>")
            self.end_state_marker = state_masking_config.get('end_state_marker', "</information>")
        else:
            self.start_state_marker = "<information>"
            self.end_state_marker = "</information>"
        
        # 只在非flash_rag模式下初始化搜索客户端
        if not self.flash_rag:
            self.search_client = HTTPSearchClient(
                search_url=self.search_url,
                topk=self.search_topk
            )
        
        # 设置搜索相关配置
        self.search_pattern = r'<search>(.*?)</search>'  # 检测搜索查询的正则表达式模式
        self.answer_pattern = r'<answer>(.*?)</answer>'  # 检测最终答案的正则表达式模式
        
        # 记录设备信息，避免设备不一致问题
        try:
            self.model_device = next(self.inference_engine.model.parameters()).device
        except:
            self.model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.sampling_params.stop = [self.search_stop]
        self.sampling_params.ignore_eos = False
        if hasattr(self.sampling_params, 'detokenize'):
            self.sampling_params.detokenize = True

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

    def _flash_rag_batch_search(self, queries):
        """使用flash_rag模式实现的批量搜索"""
        print(f"========== Batch search with {len(queries)} queries ==========")
        if not queries:
            return []
        
        try:
            # 使用batch_search端点处理批量查询
            if len(queries) > 1:
                url = self.search_url
                data = {'query': queries, 'top_n': self.search_topk}
                response = requests.post(url, json=data, timeout=10)
                
                result_list = []
                # 处理二维数组格式的返回结果
                for item_group in response.json():
                    curr_result = ''
                    for item in item_group:
                        curr_result += f"{item['contents']}\n\n"
                    result_list.append(curr_result.strip())
                
                return result_list
            # 使用单个查询
            else:
                url = self.search_url.replace('/batch_search', '/search')
                data = {'query': queries[0], 'top_n': self.search_topk}
                response = requests.post(url, json=data, timeout=10)
                
                curr_result = ''
                for item in response.json():
                    curr_result += f"{item['contents']}\n\n"
                
                return [curr_result.strip()]
        except Exception as e:
            print(f"Flash RAG Search API error: {e}")
            # 返回空结果作为后备
            return ["No search results found due to an error."] * len(queries)

    def _process_search_in_batch(self, responses):
        """
        处理批次响应，识别搜索查询并执行搜索
        
        Args:
            responses: 响应张量 [batch_size, response_length]
            
        Returns:
            Tuple of (need_continue_flags, search_results, final_answers)
        """
        batch_size = responses.size(0)
        
        # 解码响应文本
        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        
        # 处理每个响应查找搜索查询或答案
        search_queries = []
        need_continue = [False] * batch_size
        search_results_mapping = {}  # 将批次索引映射到搜索结果位置
        final_answers = [None] * batch_size
        
        for i, text in enumerate(response_texts):
            search_match = re.search(self.search_pattern, text, re.DOTALL)
            answer_match = re.search(self.answer_pattern, text, re.DOTALL)
            
            if search_match:
                # 提取搜索查询
                query = search_match.group(1).strip()
                search_queries.append(query)
                search_results_mapping[i] = len(search_queries) - 1
                need_continue[i] = True
            elif answer_match:
                # 提取最终答案
                final_answers[i] = answer_match.group(1).strip()
                need_continue[i] = False
            else:
                # 未找到有效操作
                need_continue[i] = False
        
        # 执行批量搜索
        search_results = {}  # 确保这是一个dict[int, str]
        if search_queries:
            self.flash_rag = True
            try:
                if self.flash_rag:
                    # 使用flash_rag方式进行批量搜索
                    formatted_results = self._flash_rag_batch_search(search_queries)
                elif self.search_documents:
                    print(f"========== Search documents with {len(search_queries)} queries ==========")
                    formatted_results = search_documents(search_queries)
                else:
                    # 使用原有方式进行批量搜索
                    results = self.search_client.batch_search(search_queries)
                    formatted_results = self.search_client.format_search_results(results)
                
                # 将结果映射回原始批次索引
                for batch_idx, result_idx in search_results_mapping.items():
                    if result_idx < len(formatted_results):
                        # 确保batch_idx是整数类型
                        search_results[int(batch_idx)] = formatted_results[result_idx]
            except Exception as e:
                print(f"Error during search: {e}")
                # 提供一个默认响应
                for batch_idx in search_results_mapping:
                    # 确保batch_idx是整数类型
                    search_results[int(batch_idx)] = "No search results found due to an error."
        
        print(f"[DEBUG] Search results keys type: {[type(k) for k in search_results.keys()][:5]}")
        print(f"[DEBUG] Need continue: {sum(need_continue)}/{len(need_continue)}")
        return need_continue, search_results, final_answers
    
    def _create_next_turn_inputs(
            self, 
            original_inputs:torch.Tensor, 
            responses:torch.Tensor, 
            search_results: Dict[int, str]
        ):
        """创建下一轮对话的输入，包含上下文和搜索结果
        
        Args:
            original_inputs: 原始输入ID，形状为 [batch_size, seq_len]
            responses: 当前响应ID，形状为 [batch_size, seq_len]
            search_results: 搜索结果文本字典，键为批次索引
            
        Returns:
            torch.Tensor: 下一轮输入ID，统一长度
        """
        batch_size = original_inputs.size(0)
        device = original_inputs.device
        max_prompt_length = self.config.prompt_length
        
        # 打印调试信息
        if search_results:
            key_types = [type(k) for k in search_results.keys()]
            print(f"[DEBUG] search_results keys types: {key_types[:5]}")
        
        # 解码输入和响应文本
        response_text_list = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        tmp_input_text_list = self.tokenizer.batch_decode(original_inputs, skip_special_tokens=False)
        
        # 移除填充标记
        input_text_list = [
            text.replace(self.tokenizer.decode(self.pad_token_id), '')
            for text in tmp_input_text_list
        ]
        
        assert len(response_text_list) == len(input_text_list), "响应和输入长度不匹配"
        
        # 创建下一轮文本列表
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
        padded_inputs = padded_reversed.flip(1)  # 变回右填充
        
        # 确保长度符合要求
        if padded_inputs.shape[1] < max_prompt_length:
            pad_length = max_prompt_length - padded_inputs.shape[1]
            n_rows = padded_inputs.shape[0]
            pad_tokens = torch.full((n_rows, pad_length), self.pad_token_id, dtype=torch.long, device=device)
            padded_inputs = torch.cat([pad_tokens, padded_inputs], dim=-1)
        # 策略：在中间轮次不进行截断，保留完整上下文
        # 注释掉截断代码
        # elif padded_inputs.shape[1] > max_prompt_length:
        #     # 如果长度超过最大值，截断左侧（保留右侧的重要内容）
        #     print(f"[WARNING] Truncating inputs from {padded_inputs.shape[1]} to {max_prompt_length}")
        #     padded_inputs = padded_inputs[:, -max_prompt_length:]
        
        # 记录长度超出情况，但不截断
        if padded_inputs.shape[1] > max_prompt_length:
            print(f"[INFO] Input length exceeds max_prompt_length: {padded_inputs.shape[1]} > {max_prompt_length} (not truncating in intermediate turns)")
        
        return padded_inputs
    
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

    @torch.no_grad()
    def generate_sequences_with_search(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        支持多轮交互搜索的生成方法
        
        Args:
            prompts: 包含输入的DataProto对象
            **kwargs: 附加参数，传递给生成方法
        
        Returns:
            DataProto: 包含生成结果的DataProto对象
        """
        # breakpoint()
        device = self.model_device
        
        # 确保在meta_info中存在eos_token_id
        if 'eos_token_id' not in prompts.meta_info:
            prompts.meta_info['eos_token_id'] = self.tokenizer.eos_token_id
        
        # 合并搜索参数
        search_kwargs = {
            "stop": [self.search_stop],  # 使用配置的search_stop
        }
        
        # 获取生成控制参数
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        
        # 处理输入
        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        
        # 批次大小和采样数
        n_samples = self.sampling_params.n if do_sample else 1
        batch_size = input_ids.size(0)
        expected_batch_size = batch_size * n_samples if do_sample and n_samples > 1 else batch_size
        
        print(f"[DEBUG] Original batch size: {batch_size}, Expected final: {expected_batch_size}")
        
        # 验证模式的特殊处理 - 直接使用与父类相同的参数处理方式
        if not do_sample:
            search_kwargs.update({
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0.0,  # 与父类保持一致
                'n': 1  # if greedy, only 1 response
            })
        elif is_validate:
            search_kwargs.update({
                'top_k': self.config.val_kwargs.top_k if hasattr(self.config.val_kwargs, 'top_k') else -1,
                'top_p': self.config.val_kwargs.top_p if hasattr(self.config.val_kwargs, 'top_p') else 1.0,
                'temperature': self.config.val_kwargs.temperature if hasattr(self.config.val_kwargs, 'temperature') else 1.0,
                'n': 1,  # if validate, already repeat in ray_trainer
            })
            
        # 设置参数
        max_turns = min(self.max_turns, 3) if is_validate else self.max_turns
        max_prompt_length = self.config.prompt_length
        max_response_length = self.config.response_length
        
        # 处理批次重复
        if n_samples > 1 and do_sample:
            original_inputs = input_ids.repeat_interleave(n_samples, dim=0)
            fixed_size_input_ids = original_inputs.clone()
            attention_mask = attention_mask.repeat_interleave(n_samples, dim=0)
            position_ids = position_ids.repeat_interleave(n_samples, dim=0)
        else:
            original_inputs = input_ids.clone()
            fixed_size_input_ids = original_inputs.clone()
        
        # 跟踪活动样本
        active_mask = torch.ones(expected_batch_size, dtype=torch.bool, device=device)
        final_answers = [None] * expected_batch_size
        final_responses = ['' for _ in range(expected_batch_size)]
        
        # 多轮生成循环
        for turn in range(max_turns):
            if not active_mask.any():
                print(f"[DEBUG] All samples completed at turn {turn}")
                break
            
            # 获取活跃示例
            num_active = active_mask.sum().item()
            print(f"Turn {turn}, active examples: {num_active}/{expected_batch_size}")
            
            # 如果没有活跃样本，直接跳出循环
            if num_active == 0:
                print(f"[DEBUG] No active examples at turn {turn}, breaking early")
                break
                
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
                    # 处理不同大小的non_tensor数据
                    try:
                        if len(val) == batch_size and n_samples > 1 and do_sample:
                            # 重复non_tensor数据
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
            
            # 强制每轮n=1，避免维度爆炸
            local_kwargs = kwargs.copy()
            local_kwargs['n'] = 1
            
            # 检查批次大小，确保非空
            batch_size_check = active_prompts.batch.batch_size[0]
            if batch_size_check == 0:
                print(f"[ERROR] Empty batch detected at turn {turn}, breaking early")
                break
            
            # 处理合并后的kwargs，确保所有参数有效
            final_kwargs = {**local_kwargs, **search_kwargs}
            
            # 直接使用父类的方法，不进行额外的参数检查和转换，保持一致性
            breakpoint()
            try:
                gen_output = super().generate_sequences(active_prompts, **final_kwargs)
            except RuntimeError as e:
                if "CUDA error" in str(e):
                    print(f"[ERROR] CUDA error in generate_sequences: {e}")
                    print("[DEBUG] Active batch info:")
                    print(f"  Batch size: {batch_size_check}")
                    print(f"  Active mask sum: {active_mask.sum().item()}")
                    print(f"  search_kwargs: {search_kwargs}")
                    print(f"  local_kwargs: {local_kwargs}")
                    break
                raise  # 重新抛出其他类型的错误
            breakpoint()
            responses = gen_output.batch['responses']
            
            # 处理形状不匹配问题
            if responses.shape[0] != num_active:
                print(f"Warning: Expected responses shape[0]={num_active}, got {responses.shape[0]}")
                if responses.shape[0] > num_active:
                    responses = responses[:num_active]
            
            # 创建当前轮次的响应张量
            turn_responses = torch.zeros(
                (expected_batch_size, responses.size(1)), 
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
            
            # 检查是否还有活跃样本，如果没有则不再准备下一轮输入
            if not active_mask.any():
                print(f"[DEBUG] No more active samples after processing turn {turn}")
                continue
                
            # 如果需要继续，准备下一轮输入
            if turn < max_turns - 1:
                # 确保有新的活跃索引
                if not new_active_indices:
                    print(f"[DEBUG] No new active indices at turn {turn}, stopping early")
                    break
                    
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
                    import traceback
                    traceback.print_exc()
        
        # 处理最终响应
        final_responses_token_list = self.tokenizer(final_responses, add_special_tokens=False)["input_ids"]
        final_responses_token_tensor = [torch.tensor(seq, device=device) for seq in final_responses_token_list]
        
        # 对齐响应长度
        combined_responses = rnn_utils.pad_sequence(final_responses_token_tensor, batch_first=True, padding_value=self.pad_token_id)
        
        # 最终输出阶段的截断策略
        if combined_responses.shape[1] < max_response_length:
            # 如果长度小于要求，填充到固定长度
            pad_length = max_response_length - combined_responses.shape[1]
            n_rows = combined_responses.shape[0]
            pad_tokens = torch.full((n_rows, pad_length), self.pad_token_id, dtype=torch.long, device=device)
            combined_responses = torch.cat([combined_responses, pad_tokens], dim=1)
        elif combined_responses.size(1) > max_response_length:
            # 只在最终阶段进行截断，保留右侧最新内容
            print(f"[FINAL STAGE] Truncating combined responses from {combined_responses.size(1)} to {max_response_length}")
            # 默认从右侧保留，截断左侧
            combined_responses = combined_responses[:, -max_response_length:]
        
        # 创建信息掩码
        try:
            info_mask = self._create_info_mask(combined_responses, self.start_state_marker, self.end_state_marker)
            mask_ratio = info_mask.float().mean().item()
            print(f"[DEBUG] info_mask created - Masked token ratio: {mask_ratio:.4f}")
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
        # 注意: info_mask中True表示信息区域（不参与训练），与最终要求相反
        # 最终的info_mask: True代表可训练的token，False代表信息区（不训练）
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
            import numpy as np
            for key, val in prompts.non_tensor_batch.items():
                # 确保非张量数据与期望的批次大小匹配
                if len(val) != expected_batch_size:
                    if len(val) == batch_size and n_samples > 1 and do_sample:
                        non_tensor_batch[key] = np.repeat(val, n_samples, axis=0)
                    else:
                        print(f"[WARNING] Non-tensor batch size mismatch for key '{key}': {len(val)} vs {expected_batch_size}")
                        # 尝试调整大小
                        if len(val) > expected_batch_size:
                            non_tensor_batch[key] = val[:expected_batch_size]
                        elif len(val) < expected_batch_size and len(val) > 0:
                            # 重复最后一个元素填充
                            padding = np.repeat(val[-1:], expected_batch_size - len(val), axis=0)
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

    def _create_info_mask(self, responses, start_marker, end_marker):
        """
        创建信息区域掩码，标记<information>和</information>之间的内容
        
        Args:
            responses: 响应张量 [batch_size, seq_len]
            start_marker: 开始标记（例如"<information>"）
            end_marker: 结束标记（例如"</information>"）
            
        Returns:
            torch.Tensor: 掩码张量，True表示信息区域（不参与训练）
        """
        # 如果没有启用状态掩码，返回全部为False的掩码（所有token都参与训练）
        if not self.use_state_masking:
            batch_size, seq_len = responses.size()
            device = responses.device
            return torch.zeros_like(responses, dtype=torch.bool, device=device)
            
        batch_size, seq_len = responses.size()
        device = responses.device
        info_mask = torch.zeros_like(responses, dtype=torch.bool, device=device)
        
        # 解码响应以查找标记
        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=False)
        
        for i, text in enumerate(response_texts):
            # 找到所有信息区域
            try:
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
            except Exception as e:
                print(f"Error creating info mask for sample {i}: {e}")
                continue
        
        # 打印统计信息
        mask_ratio = info_mask.float().mean().item()
        print(f"[DEBUG] Info mask statistics: {mask_ratio:.4f} of tokens are marked as info (not trainable)")
        
        return info_mask  # True表示信息区域（不参与训练）

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        重写generate_sequences方法使用支持搜索的版本
        
        Args:
            prompts: 输入提示
            **kwargs: 其他参数传递给生成方法
            
        Returns:
            DataProto: 生成结果
        """
        # 记录输入信息
        breakpoint()
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
        
        # 检查输入是否有效
        if prompts.batch is None and len(prompts.non_tensor_batch) == 0:
            print("[ERROR] generate_sequences - Both batch and non_tensor_batch are empty")
            raise ValueError("Invalid input: both batch and non_tensor_batch are empty")
        
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
                if prompts.batch is not None and key in prompts.batch:
                    input_shape = tuple(prompts.batch[key].shape)
                    output_shape = tuple(result.batch[key].shape)
                    print(f"[DEBUG] generate_sequences - Shape comparison for '{key}': Input={input_shape}, Output={output_shape}")
        
        return result
