# Tool-Enabled PPO Training in VERL

This document describes how to use the tool-enabled PPO training capabilities in VERL.

## Overview

VERL now supports training models that can execute code during generation using a remote tool server. This enables models to perform tasks such as:

1. Running Python code
2. Performing calculations
3. Accessing external search APIs
4. Using specialized tools

## Configuration

### Tool Server Setup

The tool functionality requires a running tool server that can execute Python code on demand. The server should expose an HTTP endpoint that accepts POST requests with a JSON payload containing code to execute.

### Training Script

Use the `train_tool_vllm_0_6_3.sh` script as a starting point. The script sets up the necessary configuration for tool-enabled training.

Key environment variables:
```bash
export ENABLE_TOOLS=true
export TOOL_SERVER_URL="http://127.0.0.1:30003"  # Update with your tool server URL
export MAX_TOOL_TURNS=3
```

Configuration parameters for the rollout:
```bash
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tool_config.enabled=$ENABLE_TOOLS
actor_rollout_ref.rollout.tool_config.server_url=\"$TOOL_SERVER_URL\"
actor_rollout_ref.rollout.tool_config.max_tool_turns=$MAX_TOOL_TURNS
actor_rollout_ref.rollout.tool_config.max_retries=3
actor_rollout_ref.rollout.tool_config.retry_delay=1.0
actor_rollout_ref.rollout.tool_config.timeout=30.0
actor_rollout_ref.rollout.tool_config.max_workers=10
actor_rollout_ref.rollout.tool_config.request_interval=0.1
```

注意：工具功能的启用是通过 `tool_config.enabled` 参数控制的，而不是通过改变 `rollout.name`。这与搜索功能的控制方式相同，使用 `enable_search` 参数来启用搜索增强生成。

## How It Works

### Components

The tool integration has several key components:

1. **ToolClient** (`tool_client.py`): Handles HTTP communication with the tool server
2. **ToolEnabledVLLMRollout** (`tool_vllm_rollout.py`): Modified vLLM rollout that supports tool execution
3. **FSDP Worker Integration**: Worker code that supports the tool-enabled rollout

### Selection Logic

`fsdp_workers.py` 中的 `_build_rollout` 方法按以下优先级选择 rollout 类型：
1. 如果 `enable_search=true`，使用 SearchEnabledVLLMRollout
2. 如果 `tool_config.enabled=true`，使用 ToolEnabledVLLMRollout
3. 否则使用标准的 vLLMRollout

### Tool Protocol

The tool execution follows this protocol:

1. The model generates text containing code between `<tool>` and `</tool>` tokens
2. The ToolEnabledVLLMRollout extracts this code
3. The code is sent to the tool server via ToolClient
4. Results are inserted back into the response
5. This can happen multiple times in a single generation (multi-turn)

### Batch Processing

For efficiency, tool calls are processed in batches:

1. Multiple tool calls are collected from all responses in a batch
2. The calls are processed in parallel using a thread pool
3. Results are mapped back to their respective responses

## Usage Guide

1. **Start the Tool Server**:
   Ensure your tool server is running and accessible at the URL specified in the training script.

2. **Prepare Training Data**:
   The training data should include examples where tools are used. The format should include the `<tool>` and `</tool>` markers.

3. **Run Training**:
   ```bash
   ./AI-Researcher_dev/scripts/train_tool_vllm_0_6_3.sh
   ```

4. **Monitor Logs**:
   Check the logs in `/home/ma-user/modelarts/work/weiyu/xueban_v2/AI-Researcher_Dev/AI-Researcher_dev/scripts/logs/tool_experiments/` to track training progress.

## Troubleshooting

- **Connection Errors**: Ensure the tool server is reachable from the training nodes
- **Timeout Errors**: Increase the `timeout` parameter if tool execution takes longer
- **Memory Issues**: Reduce `max_workers` if parallel processing causes memory problems
- **Response Changes**: Check that the model is properly using the tool markers

## Advanced Configuration

The tool integration supports advanced configuration:

- `max_tool_turns`: Maximum number of back-and-forth tool interactions
- `max_retries`: Number of retry attempts for failed tool calls
- `max_workers`: Maximum number of parallel tool executions
- `retry_delay`: Delay between retry attempts
- `request_interval`: Interval between consecutive requests

## Implementation Details

- `fsdp_workers.py`: Contains the actor rollout worker with tool support
- `tool_vllm_rollout.py`: Extends vLLMRollout with tool capabilities
- `tool_client.py`: Manages communication with the tool server
- `__init__.py`: Registers the ToolEnabledVLLMRollout class

The implementation ensures that tool calls are handled efficiently and robustly, with proper error handling and retry mechanisms. 