# VERL Mixed Agent Loop Training

Enhanced VERL framework with mixed dataset support, global rate limiting, and agent loop capabilities.

## Quick Start

### 1. Set Up Ray Cluster

From jumphost (aicoder), configure and start the Ray cluster:

```bash
# Edit configuration in setup_cluster.sh
vim verl/verl/experimental/agent_loop/setup_cluster.sh

# Modify these settings:
# - MASTER_IP: Head node IP
# - WORKER_NODES: Array of worker node IPs  
# - CONDA_ENV: Your conda environment name
# - WANDB_DIR: Your WandB log directory

# Start the cluster
./verl/verl/experimental/agent_loop/setup_cluster.sh setup
```

### 2. Start Training

From jumphost, configure and start training:

```bash
# Edit training configuration
vim verl/verl/experimental/agent_loop/run_mixed_training.sh

# Modify these settings:
# - MODEL_PATH: Your model path
# - TRAIN_FILES/VAL_FILES: Dataset file paths
# - Dataset ratios (TACO_RATIO, NQ_RATIO, MATH_RATIO)
# - Training hyperparameters
# - Tool server configuration

# Start training
./verl/verl/experimental/agent_loop/run_mixed_training.sh
```

## Configuration Guide

### Cluster Configuration (`setup_cluster.sh`)

**Frequently Modified Settings:**
- `MASTER_IP`: Head node IP address
- `WORKER_NODES`: Array of worker node IPs
- `CONDA_ENV`: Conda environment name
- `WANDB_DIR`: WandB logging directory

**Stable Settings (rarely changed):**
- `CONDA_PATH`: Path to conda activation script
- `RAY_TEMP_DIR`: Ray temporary directory
- `RAY_DASHBOARD_PORT`: Ray dashboard port

### Training Configuration (`run_mixed_training.sh`)

**Frequently Adjusted Parameters:**
- Dataset paths and ratios
- Model path and training hyperparameters
- Hardware configuration (GPUs, nodes, batch sizes)
- Tool server settings

**Stable Configuration (`config/mixed_dataset_grpo.yaml`):**
- Base algorithm settings
- FSDP and optimization configurations
- Multi-turn conversation settings
- Agent loop configurations

## Features

### Mixed Dataset Support
- **Efficient Sampling**: Precomputed batch schedules for optimal performance
- **Strict Ratio Control**: Maintains exact dataset ratios within each batch
- **Agent Mapping**: Assigns specific agent types to different data sources
- **Performance Monitoring**: Real-time batch composition tracking

### Global Rate Limiting
- **Cross-Process Coordination**: File-lock based rate limiting across distributed processes
- **Tool Call Management**: Prevents API rate limit violations
- **Configurable Limits**: Per-minute rate limits with burst handling

### Agent Loop Integration
- **Multi-Turn Conversations**: Support for extended tool-assisted conversations
- **Configurable Special Tokens**: Customizable code blocks and answer markers
- **State Masking**: Hide execution results from training loss
- **Tool Integration**: Seamless integration with external tool servers

## Usage Examples

### Basic Mixed Dataset Training
```bash
# Set dataset ratios
TACO_RATIO=0.4    # 40% code execution
NQ_RATIO=0.3      # 30% search tasks  
MATH_RATIO=0.3    # 30% math problems

# Start training with current settings
./run_mixed_training.sh
```

### Custom Model Training
```bash
# Edit run_mixed_training.sh:
MODEL_PATH="/path/to/your/model"
LEARNING_RATE=5e-7
TRAIN_BATCH_SIZE=256

./run_mixed_training.sh
```

### Multi-Node Setup
```bash
# Edit setup_cluster.sh:
MASTER_IP="172.26.81.88"
WORKER_NODES=(
    "172.26.81.89"
    "172.26.81.90" 
    "172.26.81.91"
)

./setup_cluster.sh setup
```

## Monitoring

### Cluster Status
```bash
# Check Ray cluster status
./setup_cluster.sh status

# Check individual node sessions
ssh 172.26.81.88 'tmux attach -t head'
ssh 172.26.81.89 'tmux attach -t node0'
```

### Training Progress
```bash
# Monitor training on master node
ssh 172.26.81.88 'tmux attach -t verl_training'

# Check training logs
ssh 172.26.81.88 'tail -f /path/to/logs/train_*.log'
```

### Dashboard Access
- Ray Dashboard: `http://MASTER_IP:8265`
- WandB Logs: Check configured WandB directory
- TensorBoard: Standard VERL tensorboard logs

## Troubleshooting

### Cluster Issues
```bash
# Restart cluster
./setup_cluster.sh restart

# Manual cleanup
./setup_cluster.sh cleanup
```

### Training Issues
```bash
# Check training status
ssh MASTER_IP 'tmux capture-pane -t verl_training -p'

# Kill training session
ssh MASTER_IP 'tmux kill-session -t verl_training'
```

### Common Problems
1. **Connection Refused**: Check if cluster is running with `./setup_cluster.sh status`
2. **Tool Server Errors**: Verify tool server URL and connectivity
3. **Dataset Not Found**: Check dataset file paths and permissions
4. **CUDA Out of Memory**: Reduce batch sizes or enable gradient checkpointing

## File Structure

```
verl/experimental/agent_loop/
├── README.md                           # This file
├── setup_cluster.sh                    # Cluster setup script
├── run_mixed_training.sh               # Training execution script
├── config/
│   └── mixed_dataset_grpo.yaml        # Base training configuration
├── code_execution_agent_loop.py       # Enhanced agent loop implementation
├── efficient_mixed_sampler.py         # Optimized mixed dataset sampler
├── global_rate_limiter.py             # Cross-process rate limiting
└── enhanced_tool_client.py            # Tool client with rate limiting
```

## Advanced Configuration

### Custom Agent Types
Edit `config/mixed_dataset_grpo.yaml` to add new agent types:

```yaml
agent_loops:
  custom_agent:
    _target_: your.custom.AgentLoop
```

### Special Token Configuration
Modify tool configuration in the training script:

```bash
actor_rollout_ref.rollout.multi_turn.tool_config.code_start_tag="<custom_code>"
actor_rollout_ref.rollout.multi_turn.tool_config.code_end_tag="</custom_code>"
```

### Rate Limiting Tuning
Adjust rate limiting parameters:

```bash
RATE_LIMIT_PER_MINUTE=60000     # Higher for more aggressive tool usage
TOOL_MAX_WORKERS=100            # More concurrent workers
TOOL_REQUEST_INTERVAL=0.005     # Faster request intervals
``` 