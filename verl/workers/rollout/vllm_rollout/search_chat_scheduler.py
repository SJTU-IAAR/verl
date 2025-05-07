# search_chat_scheduler.py
import asyncio
import re
import requests
from typing import Any, Dict, List

import torch
import numpy as np
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.rollout.async_server import ChatCompletionScheduler
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

import aiohttp
import random
import time


class HTTPSearchClient:
    """HTTP client for remote search API calls with load balancing"""
    """为了负载均衡，需要修改search_url，添加8005备用服务器"""
    
    # 添加额外的备用服务器URLs
    DEFAULT_BACKUP_URLS = [
        "http://192.168.200.194:30014/retrieve",
        "http://192.168.200.194:30013/retrieve"
    ]
    
    def __init__(self, search_url, topk=3):
        # 支持多个后端URL的负载均衡
        if isinstance(search_url, str):
            # 基本URL列表，以主URL开始
            self.search_urls = [search_url]
            
            # 检查主URL是否使用8004端口，如果是，自动添加8005备用服务器
            if "8004" in search_url:
                self.search_urls.append(search_url.replace("8004", "8005"))
            
            # 添加额外的备用URLs
            self.search_urls.extend(self.DEFAULT_BACKUP_URLS)
            print(f"Configured load balancing between: {self.search_urls}")
        elif isinstance(search_url, (list, tuple)):
            self.search_urls = list(search_url)
            # 添加额外的备用URLs
            self.search_urls.extend(self.DEFAULT_BACKUP_URLS)
        else:
            self.search_urls = [search_url]
            self.search_urls.extend(self.DEFAULT_BACKUP_URLS)
            
        self.topk = topk
        self.max_retries = 1  # 减少每个URL的重试次数，快速失败
        self.max_backoff = 3.0  # 降低最大退避时间，避免长时间等待
        self.timeout = 10  # 更短的超时时间，单位：秒
        
        # 负载均衡状态跟踪
        self.url_index = 0  # 当前使用的URL索引
        self.last_success = {}  # 存储每个URL的上次成功时间
        self.consecutive_failures = {url: 0 for url in self.search_urls}  # 连续失败计数
        
        print(f"Initialized HTTPSearchClient with {len(self.search_urls)} endpoints")
        
    def _get_next_url(self):
        """获取下一个要使用的URL，基于简单的轮询策略和健康状态"""
        # 检查连续失败次数，如果超过阈值，则尝试其他URL
        for _ in range(len(self.search_urls)):
            url = self.search_urls[self.url_index]
            self.url_index = (self.url_index + 1) % len(self.search_urls)
            
            # 如果URL连续失败次数少于3次，则使用它
            if self.consecutive_failures[url] < 3:
                return url
                
        # 所有URL都有问题，使用第一个
        self.url_index = 0
        return self.search_urls[0]
    
    def _mark_url_success(self, url):
        """标记URL请求成功"""
        self.consecutive_failures[url] = 0
        self.last_success[url] = time.time()
    
    def _mark_url_failure(self, url):
        """标记URL请求失败"""
        self.consecutive_failures[url] = self.consecutive_failures.get(url, 0) + 1
        
    def _create_fallback_result(self, query, error_msg):
        """创建后备搜索结果，确保训练能继续进行"""
        # 创建一个带有错误消息的假文档
        return [{
            "document": {
                "contents": f"Search Error\nThe search request failed: {error_msg}. Please continue based on your knowledge."
            }
        }]
    
    async def batch_search(self, queries: List[str]):
        """Call remote search API with load balancing and better failure handling"""
        if not queries:
            return []
        
        results = []
        for query in queries:
            # Process one query at a time to avoid entire batch failing
            result = await self._search_single_query(query)
            results.append(result)
            
        return results
            
    async def _search_single_query(self, query: str):
        """Process a single search query with fallback mechanism"""
        payload = {
            "queries": [query],
            "topk": self.topk,
            "return_scores": True
        }
        
        # Track attempts across all endpoints
        total_attempts = 0
        max_total_attempts = len(self.search_urls) * (self.max_retries + 1)
        
        # Try all available URLs
        for _ in range(max_total_attempts):
            if total_attempts >= max_total_attempts:
                break
                
            total_attempts += 1
            current_url = self._get_next_url()
            
            try:
                async with aiohttp.ClientSession() as session:
                    # Use shorter timeout to avoid blocking for too long
                    async with session.post(
                        current_url, 
                        json=payload, 
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        
                        # Check if the expected key exists
                        if "result" not in result:
                            self._mark_url_failure(current_url)
                            continue
                        
                        # Success, mark this URL as healthy
                        self._mark_url_success(current_url)
                        
                        # Return just the first result since we're processing one query
                        if result["result"] and len(result["result"]) > 0:
                            return result["result"][0]
                        else:
                            # Empty result but successful API call
                            return self._create_fallback_result(
                                query, 
                                "No documents found for this query"
                            )
                        
            except aiohttp.ClientResponseError as e:
                # Server returned an error status code
                self._mark_url_failure(current_url)
                error_type = f"HTTP {e.status}"
                # Short backoff before trying next endpoint
                await asyncio.sleep(0.1)
                
            except asyncio.TimeoutError:
                # Request timed out
                self._mark_url_failure(current_url)
                error_type = "timeout"
                # Short backoff before trying next endpoint
                await asyncio.sleep(0.1)
                
            except Exception as e:
                # Other error
                self._mark_url_failure(current_url)
                error_type = "general error"
                # Short backoff before trying next endpoint
                await asyncio.sleep(0.1)
        
        # All endpoints failed, return a fallback result
        return self._create_fallback_result(
            query, 
            f"All search endpoints failed after {total_attempts} attempts"
        )
        
    def format_search_results(self, results):
        """Format search results as readable text, handling potential errors"""
        formatted_results = []
        
        for result_set in results:
            # Check if it's an error marker (new format from _create_fallback_result)
            if (result_set and isinstance(result_set, list) and len(result_set) == 1 
                and isinstance(result_set[0], dict) and "document" in result_set[0]):
                doc = result_set[0]["document"]
                if "contents" in doc and doc["contents"].startswith("Search Error"):
                    formatted_results.append(f"<information>{doc['contents']}</information>")
                    continue
            
            # Check if it's an error marker (old format)
            if result_set and isinstance(result_set, list) and len(result_set) == 1 and isinstance(result_set[0], dict) and "error" in result_set[0]:
                formatted_results.append(f"<information>Search Error: {result_set[0]['error']}. Please continue with your best knowledge.</information>")
            elif not result_set:  # Handle empty result set (no documents found)
                formatted_results.append("<information>No search results found. Please continue with your best knowledge.</information>") 
            else:
                # Normal formatting of search results
                format_reference = ''
                for idx, doc_item in enumerate(result_set):
                    # Safely access dictionary keys
                    doc = doc_item.get('document', {})
                    content = doc.get('contents', 'Content unavailable')
                    
                    # Split content safely
                    content_parts = content.split("\n")
                    title = content_parts[0] if content_parts else 'Title unavailable'
                    text = "\n".join(content_parts[1:]) if len(content_parts) > 1 else 'Text unavailable'
                    
                    format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
                formatted_results.append(format_reference.strip())  # Strip trailing newline
        
        return formatted_results


class SearchEnabledChatCompletionScheduler(ChatCompletionScheduler):
    """
    ChatCompletionScheduler with search capabilities during generation.
    Supports multi-turn conversations with search integration.
    """

    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        super().__init__(config, model_path, server_addresses, max_cache_size)
        
        # Initialize tokenizer - important for processing text properly
        local_path = copy_to_local(model_path)
        trust_remote_code = config.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        
        # Set search-related configuration
        self.enable_search = config.get('enable_search', False)
        self.search_url = config.get('search_url', 'http://localhost:8000/retrieve')
        self.search_topk = config.get('search_topk', 3)
        self.max_turns = config.get('max_turns', 5)
        self.search_stop = config.get('search_stop', "</search>")
        
        # State masking configuration 
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
        # To match search_vllm_rollout.py
        self.search_pattern = r'<search>(.*?)</search>'
        self.answer_pattern = r'<answer>(.*?)</answer>'
        
        # print(f"[INFO] Initialized SearchEnabledChatCompletionScheduler with URL: {self.search_url}, topk: {self.search_topk}")
        # print(f"[INFO] Using state masking: {self.use_state_masking}, markers: {self.start_state_marker}/{self.end_state_marker}")
        # print(f"[INFO] Search pattern: {self.search_pattern}, Answer pattern: {self.answer_pattern}")

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        if not self.enable_search:
            # Fall back to standard generation if search is disabled
            # print("[INFO] Search is disabled, using standard generation mode")
            return await self._generate_standard(batch, **sampling_params)
        
        # print("[INFO] Search is enabled, using search-enhanced generation mode")
        return await self._generate_with_search(batch, **sampling_params)

    async def _generate_standard(self, batch: DataProto, **sampling_params) -> DataProto:
        """Standard generation without search - similar to NaiveChatCompletionScheduler"""
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[SearchEnabledChatCompletionScheduler] generate_sequences standard mode, params: {kwargs}")

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            if exception:
                print(f"Error in generation: {exception}")
                return
                
            conversation, batch_conversations, batch_index = (
                info["conversation"],
                info["batch_conversations"],
                info["batch_index"],
            )

            conversations = []
            for choice in completions.choices:
                chat = conversation.copy()
                chat.append({"role": choice.message.role, "content": choice.message.content})
                conversations.append(chat)
            batch_conversations[batch_index] = conversations

        tasks, batch_conversations = [], [None] * len(batch)
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_index": batch_index,
                            "conversation": list(conversation),
                        },
                        model=self.model_name,
                        messages=conversation,
                        **kwargs,
                    )
                )
            )
        await asyncio.gather(*tasks)
        print(f"[INFO] Completed standard generation for {len(batch)} examples")

        return self._postprocess(batch, batch_conversations, kwargs["n"])

    async def _generate_with_search(self, batch: DataProto, **sampling_params) -> DataProto:
        """Generate with search capability - multi-turn interaction with search"""
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop=[self.search_stop]
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        max_turns = min(self.max_turns, 3) if is_validate else self.max_turns
        # print(f"[SearchEnabledChatCompletionScheduler] generate_sequences with search, max_turns: {max_turns}, params: {kwargs}")

        # 添加互斥锁来保护共享状态访问
        lock = asyncio.Lock()

        # The search-enabled callback is more complex, handling multi-turn interactions
        async def search_callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            # 处理异常但不直接返回，以确保至少更新batch_conversations
            has_error = False
            batch_index = info["batch_index"]
            turn = info["turn"]
            
            if exception:
                # print(f"Error in generation for sample {batch_index}, turn {turn}: {exception}")
                has_error = True
                
            conversation = info["conversation"]
            context = info["context"]
            batch_conversations = info["batch_conversations"]
            
            # Check if we need to continue with search
            need_continue = False
            search_query = None
            
            if not has_error and completions and completions.choices:
                response_text = completions.choices[0].message.content
                
                # 处理因stop token导致的不完整搜索标记
                if completions.choices[0].finish_reason == "stop" and "<search>" in response_text and "</search>" not in response_text:
                    # print(f"[INFO] Detected incomplete search tag in sample {batch_index}, turn {turn} - appending closing tag")
                    response_text += "</search>"
                    # 更新completions中的响应内容，确保修改后的文本被使用
                    completions.choices[0].message.content = response_text
                
                # 严格按照search_vllm_rollout.py的逻辑检查搜索模式
                search_match = re.search(self.search_pattern, response_text, re.DOTALL)
                answer_match = re.search(self.answer_pattern, response_text, re.DOTALL)
                
                # Update the conversation with current response
                conversation.append({"role": "assistant", "content": response_text})
                
                if search_match and turn < max_turns - 1:
                    # Extract search query
                    search_query = search_match.group(1).strip()
                    need_continue = True
                    # print(f"[INFO] Turn {turn}, sample {batch_index}: Found search query '{search_query}'")
                elif answer_match:
                    # We have a final answer
                    async with lock:
                        context["final_answer"] = answer_match.group(1).strip()
                    need_continue = False
                    # print(f"[INFO] Turn {turn}, sample {batch_index}: Found final answer")
                elif turn >= max_turns - 1:
                    # Reached max turns, end conversation
                    # print(f"[INFO] Turn {turn}, sample {batch_index}: Reached max turns")
                    need_continue = False
                
                # Store intermediate results
                async with lock:
                    batch_conversations[batch_index] = [conversation]
            else:
                # Handle error case by storing the conversation up to this point
                if has_error:
                    # print(f"[WARNING] Error occurred for sample {batch_index} at turn {turn}")
                    if turn > 0:  # Only if it's not the first turn
                        # Keep the conversation up to this point
                        async with lock:
                            batch_conversations[batch_index] = [conversation]
            
            # Continue with search if needed and no errors
            if not has_error and need_continue and search_query:
                try:
                    # Add small random delay to stagger requests
                    await asyncio.sleep(random.uniform(0.1, 0.5))
                    
                    # Perform search
                    print(f"Processing search query: '{search_query[:30]}{'...' if len(search_query) > 30 else ''}'")
                    search_results = await self.search_client.batch_search([search_query])
                    formatted_results = self.search_client.format_search_results(search_results)
                    
                    if formatted_results and formatted_results[0].strip():
                        # Add search results to conversation - 保持原有的格式
                        search_result_text = formatted_results[0]
                        
                        conversation.append({
                            "role": "user", 
                            "content": f"\n\n{self.start_state_marker}{search_result_text}{self.end_state_marker}\n\nBased on the information above, please continue."
                        })
                        
                        # print(f"[INFO] Turn {turn}, sample {batch_index}: Added search results, continuing to next turn")
                        
                        # Submit next turn
                        await self.submit_chat_completions(
                            callback=search_callback,
                            callback_additional_info={
                                "conversation": conversation,
                                "context": context,
                                "batch_conversations": batch_conversations,
                                "batch_index": batch_index,
                                "turn": turn + 1
                            },
                            model=self.model_name,
                            messages=conversation,
                            **kwargs
                        )
                    else:
                        # print(f"[WARNING] Turn {turn}, sample {batch_index}: No search results found")
                        conversation.append({
                            "role": "user", 
                            "content": "\n\nNo search results found. Please provide your best answer based on your knowledge."
                        })
                        
                        # Submit next turn when no search results
                        await self.submit_chat_completions(
                            callback=search_callback,
                            callback_additional_info={
                                "conversation": conversation,
                                "context": context,
                                "batch_conversations": batch_conversations,
                                "batch_index": batch_index,
                                "turn": turn + 1
                            },
                            model=self.model_name,
                            messages=conversation,
                            **kwargs
                        )
                except Exception as e:
                    print(f"[ERROR] Error during search or next turn submission for sample {batch_index}: {e}")
                    # Ensure we have a result even if there's an error
                    async with lock:
                        batch_conversations[batch_index] = [conversation]

        # Start the multi-turn process for each batch item
        tasks = []
        batch_conversations = [None] * len(batch)
        final_contexts = [{"final_answer": None} for _ in range(len(batch))]
        
        print(f"[INFO] Starting multi-turn generation with search for {len(batch)} examples")
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=search_callback,
                        callback_additional_info={
                            "conversation": list(conversation),
                            "context": final_contexts[batch_index],
                            "batch_conversations": batch_conversations,
                            "batch_index": batch_index,
                            "turn": 0
                        },
                        model=self.model_name,
                        messages=conversation,
                        **kwargs,
                    )
                )
            )
        await asyncio.gather(*tasks)
        
        # Handle missing conversations
        none_count = sum(1 for conv in batch_conversations if conv is None)
        if none_count > 0:
            print(f"[WARNING] Found {none_count}/{len(batch_conversations)} None values in batch_conversations")
            for i in range(len(batch_conversations)):
                if batch_conversations[i] is None:
                    batch_conversations[i] = [batch.non_tensor_batch["raw_prompt"][i]]
        
        # Store final answers in the batch metadata
        final_answers = [context["final_answer"] for context in final_contexts]
        
        # Process final outputs
        result = self._postprocess(batch, batch_conversations, kwargs["n"])
        
        # Add info_mask for state masking - to mask out information sections during training
        if self.use_state_masking and "responses" in result.batch:
            responses = result.batch["responses"]
            # 先生成原始info_mask
            raw_info_mask = self._create_info_mask(responses)
            # 转换格式 - 完全按照search_vllm_rollout.py的逻辑
            attention_mask = (responses != self.tokenizer.pad_token_id).int()
            info_mask = (~raw_info_mask & attention_mask).bool()
            result.batch["info_mask"] = info_mask
            
        # Store final answers in meta_info
        result.meta_info["final_answers"] = final_answers
        
        # print(f"[INFO] Completed multi-turn generation with search for {len(batch)} examples")
        return result

    def _create_info_mask(self, responses):
        """Create mask for information regions (between markers)"""
        batch_size, seq_len = responses.size()
        device = responses.device
        
        # If state masking is disabled, return an empty mask (all tokens trainable)
        if not self.use_state_masking:
            return torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        
        # Decode responses to text
        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        # 初始掩码 - True表示信息区域（不训练）
        info_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        
        for i, text in enumerate(response_texts):
            try:
                # Find all information regions
                start_positions = [m.start() for m in re.finditer(re.escape(self.start_state_marker), text)]
                end_positions = [m.start() + len(self.end_state_marker) for m in re.finditer(re.escape(self.end_state_marker), text)]
                
                # Ensure markers appear in pairs
                n_regions = min(len(start_positions), len(end_positions))
                if n_regions == 0:
                    continue
                
                # Convert text positions to token positions
                for j in range(n_regions):
                    start_pos = start_positions[j]
                    valid_ends = [pos for pos in end_positions if pos > start_pos]
                    if not valid_ends:
                        continue
                    end_pos = min(valid_ends)
                    
                    # Find token indices
                    prefix_text = text[:start_pos]
                    prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                    start_idx = len(prefix_tokens)
                    
                    info_text = text[:end_pos]
                    info_tokens = self.tokenizer.encode(info_text, add_special_tokens=False)
                    end_idx = len(info_tokens)
                    
                    # Set mask for information region - True表示要掩盖的区域
                    if start_idx < end_idx and start_idx < seq_len and end_idx <= seq_len:
                        info_mask[i, start_idx:end_idx] = True
            except Exception as e:
                print(f"Error creating info mask for sample {i}: {e}")
                continue
        
        return info_mask  # 返回原始info_mask，True表示信息区域（要掩盖）
        
    def _postprocess(self, batch: DataProto, batch_conversations: List[List[List[Dict[str, str]]]], n: int) -> DataProto:
        """Convert conversation history back to tokenized format"""
        # Prepare prompts
        prompts = [self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in batch.non_tensor_batch["raw_prompt"]]

        # Flatten batch_conversations if n > 1
        assert len(batch_conversations) == len(prompts), f"Expected {len(prompts)} conversations, got {len(batch_conversations)}"
        
        # Handle missing conversations (could happen if some failed)
        for i in range(len(batch_conversations)):
            if batch_conversations[i] is None:
                print(f"[WARNING] No conversation for prompt {i}, using empty conversation")
                batch_conversations[i] = [batch.non_tensor_batch["raw_prompt"][i]]
        
        batch_conversations = [conversation for conversations in batch_conversations for conversation in conversations]
        expected_count = len(prompts) * n
        actual_count = len(batch_conversations)
        
        if actual_count != expected_count:
            print(f"[WARNING] Expected {expected_count} conversations, got {actual_count}")
            # Ensure we have the right number by duplicating or truncating
            if actual_count < expected_count:
                # Duplicate last conversation to match expected count
                batch_conversations.extend([batch_conversations[-1]] * (expected_count - actual_count))
            else:
                # Truncate to expected count
                batch_conversations = batch_conversations[:expected_count]

        # Create sequences and responses
        sequences = [self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False) for conversation in batch_conversations]
        responses = [sequence[len(prompts[i // n]):] for i, sequence in enumerate(sequences)]

        # Tokenize
        prompts_tokenized = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses_tokenized = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        
        if n > 1:
            prompts_tokenized["input_ids"] = prompts_tokenized["input_ids"].repeat_interleave(n, dim=0)
            prompts_tokenized["attention_mask"] = prompts_tokenized["attention_mask"].repeat_interleave(n, dim=0)

        input_ids = torch.cat([prompts_tokenized["input_ids"], responses_tokenized["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts_tokenized["attention_mask"], responses_tokenized["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch_tensors = {
            "prompts": prompts_tokenized["input_ids"],
            "responses": responses_tokenized["input_ids"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        
        # 创建新的DataProto并确保正确复制元数据
        meta_info = batch.meta_info.copy() if hasattr(batch, "meta_info") and batch.meta_info is not None else {}
        
        result = DataProto(
            batch=TensorDict(batch_tensors, batch_size=len(input_ids)),
            meta_info=meta_info,
            non_tensor_batch={}
        )
        
        # Copy non_tensor_batch from original batch if needed
        if hasattr(batch, "non_tensor_batch") and batch.non_tensor_batch:
            for key, value in batch.non_tensor_batch.items():
                if key != "raw_prompt":  # Skip raw_prompt as we've already processed it
                    try:
                        if isinstance(value, np.ndarray) and len(value) == len(prompts_tokenized["input_ids"]) // n:
                            # Handle arrays that need to be repeated for n samples
                            result.non_tensor_batch[key] = np.repeat(value, n, axis=0)
                        else:
                            result.non_tensor_batch[key] = value
                    except Exception as e:
                        print(f"[WARNING] Failed to copy non_tensor_batch key '{key}': {e}")
            
        return result