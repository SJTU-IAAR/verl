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
import asyncio
import re
import time
from typing import Any, Dict, List, Tuple, Optional, Union
import aiohttp
import torch
import numpy as np
from cachetools import LRUCache
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.rollout.async_server import ChatCompletionScheduler


class AsyncLimiter:
    """Rate limiter for controlling API request rates."""
    
    def __init__(self, max_calls: int, time_period: float):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the time period
            time_period: Time period in seconds
        """
        self.max_calls = max_calls
        self.time_period = time_period
        self.calls = []
        self.lock = asyncio.Lock()
        
    async def __aenter__(self):
        async with self.lock:
            # Remove calls older than the time period
            current_time = time.time()
            self.calls = [t for t in self.calls if current_time - t < self.time_period]
            
            # Wait if we've reached the maximum number of calls
            if len(self.calls) >= self.max_calls:
                oldest_call = self.calls[0]
                sleep_time = oldest_call + self.time_period - current_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self.calls = self.calls[1:]  # Remove the oldest call
                
            # Add the current call
            self.calls.append(time.time())
            
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class AsyncHTTPSearchClient:
    """Asynchronous HTTP client for search API calls."""
    
    def __init__(self, search_url: str, topk: int = 3, max_concurrent: int = 20, rate_limit: Tuple[int, float] = (100, 60.0)):
        """Initialize the search client.
        
        Args:
            search_url: URL of the search API
            topk: Number of top results to retrieve
            max_concurrent: Maximum number of concurrent requests
            rate_limit: Tuple of (max_calls, time_period) for rate limiting
        """
        self.search_url = search_url
        self.topk = topk
        self.session = None
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = AsyncLimiter(*rate_limit)
        
        # Statistics
        self.total_requests = 0
        self.failed_requests = 0
        self.last_error = None
    
    async def ensure_session(self):
        """Ensure an aiohttp session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),  # Increased timeout
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "SearchChatScheduler/1.0"
                }
            )
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session is not None:
            await self.session.close()
            self.session = None
    
    async def batch_search(self, queries: List[str]) -> List[List[Dict]]:
        """Perform batch search asynchronously with concurrency control.
        
        Args:
            queries: List of search queries
            
        Returns:
            List of search results for each query
        """
        if not queries:
            return []
        
        self.total_requests += 1
        
        try:
            # Use both the semaphore and rate limiter for comprehensive control
            async with self.semaphore, self.rate_limiter:
                await self.ensure_session()
                
                payload = {
                    "queries": queries,
                    "topk": self.topk,
                    "return_scores": True
                }
                
                async with self.session.post(
                    self.search_url, 
                    json=payload, 
                    timeout=20  # Extended timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("result", [[] for _ in range(len(queries))])
                    else:
                        error_text = await response.text()
                        self.last_error = f"Status {response.status}: {error_text[:100]}"
                        self.failed_requests += 1
                        print(f"Search API error: {self.last_error}")
                        return [[] for _ in range(len(queries))]
        except asyncio.TimeoutError:
            self.failed_requests += 1
            self.last_error = "Request timeout"
            print(f"Search API timeout after 20s")
            return [[] for _ in range(len(queries))]
        except Exception as e:
            self.failed_requests += 1
            self.last_error = str(e)
            print(f"Search API error: {e}")
            return [[] for _ in range(len(queries))]
    
    def format_search_results(self, results: List[List[Dict]]) -> List[str]:
        """Format search results as readable text.
        
        Args:
            results: List of search results for each query
            
        Returns:
            List of formatted search results as strings
        """
        formatted_results = []
        
        for result_set in results:
            format_reference = ''
            for idx, doc_item in enumerate(result_set):
                if "document" not in doc_item:
                    continue
                
                content = doc_item['document']['contents']
                title = content.split("\n")[0] if "\n" in content else "Unknown Title"
                text = "\n".join(content.split("\n")[1:]) if "\n" in content else content
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            
            formatted_results.append(format_reference)
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.total_requests - self.failed_requests) / max(1, self.total_requests),
            "last_error": self.last_error
        }


class ConversationState:
    """Class to maintain conversation state across multiple turns."""
    
    def __init__(self, conversation: List[Dict], tokenizer, max_turns: int = 5):
        """Initialize conversation state.
        
        Args:
            conversation: Initial conversation
            tokenizer: Tokenizer to use
            max_turns: Maximum number of turns allowed
        """
        self.conversation = conversation.copy()
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_count = 0
        self.search_count = 0
        self.completed = False
        self.final_answer = None
        
        # For debugging and monitoring
        self.last_error = None
        self.processing_times = []
    
    def add_assistant_message(self, content: str):
        """Add assistant message to conversation.
        
        Args:
            content: Message content
        """
        self.conversation.append({"role": "assistant", "content": content})
        self.turn_count += 1
        
        # Check if the message contains a final answer
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if answer_match:
            self.final_answer = answer_match.group(1).strip()
            self.completed = True
    
    def add_search_results(self, search_results: str):
        """Add search results to conversation.
        
        Args:
            search_results: Formatted search results
        """
        self.conversation.append({
            "role": "system", 
            "content": f"<information>{search_results}</information>"
        })
        self.search_count += 1
    
    def should_continue(self) -> bool:
        """Check if conversation should continue.
        
        Returns:
            True if conversation should continue, False otherwise
        """
        if self.completed:
            return False
        
        if self.turn_count >= self.max_turns:
            return False
        
        return True
    
    def has_search_request(self) -> Tuple[bool, Optional[str]]:
        """Check if latest message contains a search request.
        
        Returns:
            Tuple of (has_search, query)
        """
        if not self.conversation or self.conversation[-1]["role"] != "assistant":
            return False, None
        
        content = self.conversation[-1]["content"]
        search_match = re.search(r'<search>(.*?)</search>', content, re.DOTALL)
        
        if search_match:
            return True, search_match.group(1).strip()
        
        return False, None
    
    def to_tokenized_messages(self):
        """Convert conversation to tokenized messages for the model.
        
        Returns:
            List of messages suitable for model input
        """
        return self.conversation
    
    def log_processing_time(self, duration: float):
        """Log processing time for this conversation."""
        self.processing_times.append(duration)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        avg_time = sum(self.processing_times) / max(1, len(self.processing_times))
        return {
            "turn_count": self.turn_count,
            "search_count": self.search_count,
            "completed": self.completed,
            "avg_processing_time": avg_time,
            "last_error": self.last_error
        }


class SearchChatScheduler(ChatCompletionScheduler):
    """Chat completion scheduler with search capabilities."""
    
    def __init__(
        self,
        config,
        model_path: str,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        """Initialize search chat scheduler.
        
        Args:
            config: Configuration object
            model_path: Path to the model
            server_addresses: List of server addresses
            max_cache_size: Maximum cache size
        """
        super().__init__(config, model_path, server_addresses, max_cache_size)
        
        # Setup search configuration
        self.enable_search = config.get('enable_search', False)
        self.search_url = config.get('search_url', 'http://localhost:8000/retrieve')
        self.search_topk = config.get('search_topk', 3)
        self.max_turns = config.get('max_turns', 5)
        self.ignore_eos = config.get('ignore_eos', False)
        
        # Concurrency control
        max_concurrent_searches = config.get('max_concurrent_searches', 20)
        max_concurrent_completions = config.get('max_concurrent_completions', 100)
        search_rate_limit = (
            config.get('searches_per_minute', 100),  # searches per minute
            60.0,  # time period in seconds
        )
        
        # State masking configuration for handling information sections
        self.use_state_masking = config.get('state_masking', True)
        state_masking_config = config.get('state_masking', {})
        if isinstance(state_masking_config, dict):
            self.start_state_marker = state_masking_config.get('start_state_marker', "<information>")
            self.end_state_marker = state_masking_config.get('end_state_marker', "</information>")
        else:
            self.start_state_marker = "<information>"
            self.end_state_marker = "</information>"
        
        # Initialize search client with concurrency controls
        self.search_client = AsyncHTTPSearchClient(
            search_url=self.search_url,
            topk=self.search_topk,
            max_concurrent=max_concurrent_searches,
            rate_limit=search_rate_limit
        )
        
        # Configure regex patterns
        self.search_pattern = r'<search>(.*?)</search>'
        self.answer_pattern = r'<answer>(.*?)</answer>'
        
        # Cache active conversations by batch_id and request_id
        self.active_conversations = {}
        
        # Semaphores for different operations
        self.search_semaphore = asyncio.Semaphore(max_concurrent_searches)
        self.completion_semaphore = asyncio.Semaphore(max_concurrent_completions)
        
        # Statistics 
        self.total_batches = 0
        self.total_conversations = 0
        self.start_time = time.time()
        
        # Debug information
        print(f"[SearchChatScheduler] Initialized with search_url={self.search_url}, topk={self.search_topk}, max_turns={self.max_turns}")
        print(f"[SearchChatScheduler] Concurrency limits: searches={max_concurrent_searches}, completions={max_concurrent_completions}")
        print(f"[SearchChatScheduler] Rate limits: searches={search_rate_limit[0]}/minute")

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        """Generate sequences with search capability.
        
        Args:
            batch: Input batch
            **sampling_params: Sampling parameters
            
        Returns:
            Output batch with generated sequences
        """
        batch_start_time = time.time()
        self.total_batches += 1
        
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # Handle validation or deterministic settings
        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[SearchChatScheduler] generate_sequences sampling params: {kwargs}")

        # Create batch_id to track this batch
        batch_id = f"batch_{int(time.time())}_{id(batch)}"
        batch_size = len(batch.non_tensor_batch["raw_prompt"])
        batch_conversations = [None] * batch_size
        
        self.total_conversations += batch_size

        # Initialize conversation states for each item in batch
        conversation_states = []
        for conversation in batch.non_tensor_batch["raw_prompt"]:
            state = ConversationState(
                conversation=list(conversation),
                tokenizer=self.tokenizer,
                max_turns=self.max_turns if not is_validate else min(3, self.max_turns)
            )
            conversation_states.append(state)
            
        # Track which examples are still active
        active_indices = list(range(batch_size))
        
        # Main generation loop - continue until all examples are complete
        turn_counter = 0
        while active_indices and turn_counter < self.max_turns:
            turn_counter += 1
            turn_start_time = time.time()
            
            active_states = [conversation_states[i] for i in active_indices]
            active_raw_prompts = [state.to_tokenized_messages() for state in active_states]
            
            print(f"[SearchChatScheduler] Turn {turn_counter}: processing {len(active_indices)} active conversations")
            
            # Create tasks for submitting each active conversation
            tasks = []
            for idx, (batch_idx, conversation) in enumerate(zip(active_indices, active_raw_prompts)):
                tasks.append(
                    asyncio.create_task(
                        self._process_conversation_turn(
                            batch_id=batch_id,
                            batch_idx=batch_idx,
                            conversation=conversation,
                            state=conversation_states[batch_idx],
                            **kwargs
                        )
                    )
                )
            
            # Wait for all conversation turns to complete
            await asyncio.gather(*tasks)
            
            # Update active indices - keep only those that should continue
            active_indices = [
                idx for idx in active_indices 
                if conversation_states[idx].should_continue()
            ]
            
            turn_duration = time.time() - turn_start_time
            print(f"[SearchChatScheduler] Turn {turn_counter} completed in {turn_duration:.2f}s, {len(active_indices)} conversations continuing")
        
        # Collect final conversations
        for idx, state in enumerate(conversation_states):
            # Store in the batch
            batch_conversations[idx] = [state.conversation]
        
        batch_duration = time.time() - batch_start_time
        print(f"[SearchChatScheduler] generate_sequences completed in {batch_duration:.2f}s")
        
        # Print search client statistics
        search_stats = self.search_client.get_stats()
        print(f"[SearchChatScheduler] Search client stats: {search_stats}")
        
        # Clean up any lingering conversations
        keys_to_delete = [k for k in self.active_conversations.keys() if k.startswith(batch_id)]
        for key in keys_to_delete:
            self.active_conversations.pop(key, None)
            
        # Process the results
        return self._postprocess(batch, batch_conversations, kwargs["n"])

    async def _process_completion_callback(
        self,
        completions: ChatCompletion, 
        state: ConversationState,
        exception: Optional[Exception] = None
    ) -> None:
        """Process the completion results and handle search requests.
        
        Args:
            completions: Chat completion response
            state: Conversation state
            exception: Exception if any occurred
        """
        if exception:
            # Handle exceptions
            print(f"Error in completion: {exception}")
            state.completed = True
            state.last_error = str(exception)
        else:
            # Extract generated text
            choice = completions.choices[0]
            generated_text = choice.message.content
            
            # Update conversation state
            state.add_assistant_message(generated_text)
            
            # Check for search request
            has_search, query = state.has_search_request()
            
            if has_search and query:
                # Process search request
                search_start_time = time.time()
                try:
                    async with self.search_semaphore:
                        search_results = await self.search_client.batch_search([query])
                        formatted_results = self.search_client.format_search_results(search_results)
                        
                        if formatted_results and formatted_results[0]:
                            # Add search results to conversation
                            state.add_search_results(formatted_results[0])
                except Exception as e:
                    print(f"Error during search: {e}")
                    state.last_error = f"Search error: {str(e)}"
                finally:
                    search_duration = time.time() - search_start_time
                    print(f"[SearchChatScheduler] Search completed in {search_duration:.2f}s")

    async def _process_conversation_turn(
        self, 
        batch_id: str,
        batch_idx: int,
        conversation: List[Dict],
        state: ConversationState,
        **kwargs
    ) -> None:
        """Process a single conversation turn.
        
        Args:
            batch_id: Batch identifier
            batch_idx: Index in the batch
            conversation: Current conversation
            state: Conversation state
            **kwargs: Generation parameters
        """
        turn_start = time.time()
        
        # Create a unique ID for this conversation
        conversation_id = f"{batch_id}_{batch_idx}"
        
        # Store in active conversations
        self.active_conversations[conversation_id] = state
        
        # Define callback wrapper to pass state
        async def callback_wrapper(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            await self._process_completion_callback(completions, state, exception)
        
        # Submit chat completion with concurrency control
        async with self.completion_semaphore:
            try:
                await self.submit_chat_completions(
                    callback=callback_wrapper,
                    callback_additional_info={
                        "conversation_id": conversation_id,
                    },
                    model=self.model_name,
                    messages=conversation,
                    **kwargs
                )
            except Exception as e:
                print(f"Error submitting chat completion: {e}")
                state.completed = True
                state.last_error = f"Completion error: {str(e)}"
        
        # Log processing time
        turn_duration = time.time() - turn_start
        state.log_processing_time(turn_duration)

    def _postprocess(self, batch: DataProto, batch_conversations: List[List[List[Dict[str, str]]]], n: int) -> DataProto:
        """Post-process the generated completions.
        
        Args:
            batch: Input batch
            batch_conversations: Generated conversations
            n: Number of samples per prompt
            
        Returns:
            Processed DataProto object
        """
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [
            self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) 
            for prompt in batch.non_tensor_batch["raw_prompt"]
        ]

        # flatten batch_conversations if n > 1
        assert len(batch_conversations) == len(prompts)
        batch_conversations = [
            conversation for conversations in batch_conversations 
            for conversation in conversations
        ]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [
            self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False) 
            for conversation in batch_conversations
        ]

        # responses: [response]
        # Extract only the response part from each sequence
        responses = [sequence[len(prompts[i // n]):] for i, sequence in enumerate(sequences)]

        # Tokenize prompts and responses
        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        # Concatenate for full input_ids
        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        # Create info_mask to mask out retrieved information during training
        try:
            info_mask = self._create_info_mask(
                responses=responses["input_ids"],
                start_marker=self.start_state_marker,
                end_marker=self.end_state_marker
            )
            mask_ratio = info_mask.float().mean().item()
            print(f"[DEBUG] info_mask created - Masked token ratio: {mask_ratio:.4f}")
        except Exception as e:
            print(f"Error creating info mask: {e}")
            info_mask = torch.zeros_like(responses["input_ids"], dtype=torch.bool, device=responses["input_ids"].device)
            
        # Process the info_mask to match the required format
        # 注意: info_mask中True表示信息区域（不参与训练），与最终要求相反
        # 最终的info_mask: True代表可训练的token，False代表信息区（不训练）
        info_mask = ((~info_mask) & attention_mask[:, -info_mask.shape[1]:]).bool()
        
        # Final batch
        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],
                "responses": responses["input_ids"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "info_mask": info_mask
            },
            batch_size=len(input_ids),
        )

        return DataProto(batch=batch)
    
    def _create_info_mask(self, responses: torch.Tensor, start_marker: str, end_marker: str) -> torch.Tensor:
        """Create an information region mask to identify sections between marker tags.
        
        Args:
            responses: Response token IDs [batch_size, seq_len]
            start_marker: Start marker string (e.g., "<information>")
            end_marker: End marker string (e.g., "</information>")
            
        Returns:
            Mask tensor where True indicates information regions (not for training)
        """
        # If state masking is disabled, return an empty mask (all tokens trainable)
        if not self.use_state_masking:
            batch_size, seq_len = responses.size()
            return torch.zeros_like(responses, dtype=torch.bool, device=responses.device)
            
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        uptime = time.time() - self.start_time
        return {
            "total_batches": self.total_batches,
            "total_conversations": self.total_conversations,
            "uptime_seconds": uptime,
            "conversations_per_second": self.total_conversations / max(1, uptime),
            "search_stats": self.search_client.get_stats() if hasattr(self.search_client, "get_stats") else {},
            "active_conversations": len(self.active_conversations)
        }
