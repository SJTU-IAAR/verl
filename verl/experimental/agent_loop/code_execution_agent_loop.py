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
import json
import logging
import os
import re
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("code_execution_agent")
class CodeExecutionAgentLoop(AgentLoopBase):
    """Agent loop with custom ToolClient for code execution"""

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level CodeExecutionAgentLoop initialization")

        # Initialize tokenizer and config
        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        
        # Tool client configuration
        tool_config = config.actor_rollout_ref.rollout.multi_turn.get('tool_config', {})
        cls.tool_server_url = tool_config.get('server_url', 'http://127.0.0.1:30003')
        cls.max_retries = tool_config.get('max_retries', 3)
        cls.timeout = tool_config.get('timeout', 60.0)
        cls.max_workers = tool_config.get('max_workers', 10)
        cls.rate_limit_per_minute = tool_config.get('rate_limit_per_minute', 120.0)
        cls.enable_global_rate_limit = tool_config.get('enable_global_rate_limit', True)
        
        # Code detection patterns (configurable)
        code_start_tag = tool_config.get('code_start_tag', '<code>')
        code_end_tag = tool_config.get('code_end_tag', '</code>')
        answer_start_tag = tool_config.get('answer_start_tag', '<answer>')
        answer_end_tag = tool_config.get('answer_end_tag', '</answer>')
        
        cls.code_pattern = re.compile(f'{re.escape(code_start_tag)}(.*?){re.escape(code_end_tag)}', re.DOTALL)
        cls.answer_pattern = re.compile(f'{re.escape(answer_start_tag)}(.*?){re.escape(answer_end_tag)}', re.DOTALL)
        
        # State masking configuration
        cls.use_state_masking = tool_config.get('state_masking', True)
        cls.start_state_marker = tool_config.get('start_state_marker', '<execution_results>')
        cls.end_state_marker = tool_config.get('end_state_marker', '</execution_results>')
        
        # Store tags for use in response formatting
        cls.code_start_tag = code_start_tag
        cls.code_end_tag = code_end_tag
        cls.answer_start_tag = answer_start_tag
        cls.answer_end_tag = answer_end_tag
        
        # Initialize ToolClient
        from .tool_client import ToolClient
        cls.tool_client = ToolClient(
            server_url=cls.tool_server_url,
            max_retries=cls.max_retries,
            timeout=cls.timeout,
            max_workers=cls.max_workers,
            rate_limit_per_minute=cls.rate_limit_per_minute,
            enable_global_rate_limit=cls.enable_global_rate_limit
        )
        
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
        
        print(f"CodeExecutionAgentLoop initialized with server: {cls.tool_server_url}")

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        
        # Initial prompt encoding
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        response_mask = []
        
        user_turns, assistant_turns = 0, 0
        
        while True:
            # Generate response from LLM
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            assistant_turns += 1
            
            # Check termination conditions
            if len(response_mask) >= self.response_length:
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break
            
            # Decode response and check for code execution
            response_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )
            
            # Extract code for execution
            code_to_execute = self._extract_code(response_text)
            if not code_to_execute:
                break  # No code found, end conversation
            
            # Get data source for this message (if available)
            data_source = None
            if hasattr(messages[0], 'get'):
                data_source = messages[0].get('data_source')
            elif isinstance(messages, list) and len(messages) > 0:
                # Try to get from first message or context
                if isinstance(messages[0], dict):
                    data_source = messages[0].get('data_source')
            
            # Execute code using our ToolClient
            with simple_timer("tool_calls", metrics):
                try:
                    tool_results = await self._execute_code_async(
                        code_to_execute, data_source=data_source
                    )
                    if not tool_results or not tool_results[0].get('success', False):
                        break  # Tool execution failed
                except Exception as e:
                    logger.exception(f"Tool execution error: {e}")
                    break
            
            # Format tool response
            tool_response = self._format_tool_response(tool_results[0])
            
            # Encode tool response
            tool_response_message = [{"role": "tool", "content": tool_response}]
            tool_response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    tool_response_message, add_generation_prompt=True, tokenize=True
                ),
            )
            tool_response_ids = tool_response_ids[len(self.system_prompt):]
            
            # Check if adding tool response would exceed max length
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break
            
            # Add tool response to sequence
            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)  # Tool response not trained
            user_turns += 1
        
        # Prepare final output
        response_ids = prompt_ids[-len(response_mask):]
        prompt_ids = prompt_ids[:len(prompt_ids) - len(response_mask)]
        
        # Apply state masking if enabled
        if self.use_state_masking:
            response_mask = self._apply_state_masking(response_ids, response_mask)
        
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
        )
        return output
    
    def _extract_code(self, text: str) -> str:
        """Extract code from response text using patterns"""
        # Try <code> pattern first
        code_match = self.code_pattern.search(text)
        if code_match:
            return code_match.group(1).strip()
        
        # Try <answer> pattern (for math/reasoning tasks)
        answer_match = self.answer_pattern.search(text)
        if answer_match:
            return answer_match.group(1).strip()
        
        return ""
    
    async def _execute_code_async(self, code: str, data_source: str = None) -> list[dict[str, Any]]:
        """Execute code using our ToolClient in async context"""
        # Run the synchronous batch_execute in a thread pool
        return await self.loop.run_in_executor(
            None, 
            lambda: self.tool_client.batch_execute(
                codes=[code], 
                data_sources=[data_source] if data_source else None
            )
        )
    
    def _format_tool_response(self, result: dict[str, Any]) -> str:
        """Format tool execution result for the conversation"""
        if result.get('success', False):
            output = result.get('output', '')
            execution_time = result.get('execution_time', 0)
            return f"{self.start_state_marker}\nExecution successful (took {execution_time:.2f}s):\n{output}\n{self.end_state_marker}"
        else:
            error = result.get('error', 'Unknown error')
            return f"{self.start_state_marker}\nExecution failed: {error}\n{self.end_state_marker}"
    
    def _apply_state_masking(self, response_ids: list[int], response_mask: list[int]) -> list[int]:
        """Apply state masking to exclude tool execution results from training"""
        if not self.use_state_masking:
            return response_mask
        
        # Decode to find state markers
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Find marker positions
        start_positions = [m.start() for m in re.finditer(re.escape(self.start_state_marker), response_text)]
        end_positions = [m.end() for m in re.finditer(re.escape(self.end_state_marker), response_text)]
        
        # Create new mask
        new_mask = response_mask.copy()
        
        # Mask out content between markers
        for start_pos, end_pos in zip(start_positions, end_positions):
            start_token_idx = self._text_pos_to_token_idx(response_text, start_pos, response_ids)
            end_token_idx = self._text_pos_to_token_idx(response_text, end_pos, response_ids)
            
            # Set mask to 0 for tokens between markers
            for i in range(start_token_idx, min(end_token_idx, len(new_mask))):
                new_mask[i] = 0
        
        return new_mask
    
    def _text_pos_to_token_idx(self, text: str, pos: int, token_ids: list[int]) -> int:
        """Convert text position to token index (approximate)"""
        # This is a simplified implementation
        # In practice, you might need more sophisticated text-to-token mapping
        char_ratio = pos / len(text) if len(text) > 0 else 0
        return min(int(char_ratio * len(token_ids)), len(token_ids) - 1) 