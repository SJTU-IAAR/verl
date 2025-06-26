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
This script provides a standalone, "heavyweight" evaluation pipeline for a trained model.
It leverages the full RayPPOTrainer to run the validation loop, including generation
and scoring with a Reward Model, to produce a complete reward log/metric summary.
This ensures that the evaluation environment is identical to the validation environment
during training.
"""

import os
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

# Reuse components from the main_ppo entrypoint
from verl.trainer.main_ppo import create_rl_dataset
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """
    Hydra entry point for the evaluation script.
    It uses the 'ppo_trainer.yaml' as the base configuration,
    which must be overridden from the command line with evaluation-specific settings.
    """
    run_standalone_eval(config)


def run_standalone_eval(config) -> None:
    """Initializes Ray and launches the main evaluation task."""
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    # Use a remote task to ensure it runs on a node with resources, not necessarily the head node.
    eval_runner = EvalTaskRunner.remote()
    ray.get(eval_runner.run.remote(config))
    ray.shutdown()
    print("\nEvaluation finished successfully. Ray has been shut down.")


@ray.remote(num_cpus=1)
class EvalTaskRunner:
    """
    A Ray actor that encapsulates the setup and execution of the evaluation task.
    This mimics the structure of TaskRunner in main_ppo.py.
    """

    def run(self, config):
        """
        Sets up the RayPPOTrainer and runs the validation step.
        """
        from verl.utils.fs import copy_to_local

        # Print the fully resolved config for debugging
        print("--- Starting Standalone Evaluation with Resolved Config ---")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        print("---------------------------------------------------------")

        # --- 1. Setup Tokenizer and Processor ---
        # The tokenizer is loaded from the actor model path, which should be the trained model.
        actor_model_path = copy_to_local(config.actor_rollout_ref.model.path)
        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(actor_model_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(actor_model_path, use_fast=True)

        # --- 2. Define Worker Roles and Resource Mapping ---
        # This setup is critical and must mirror the training setup to ensure
        # all necessary components (like RewardModel) are initialized.
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        # Define the mapping of roles to worker classes
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Setup resource pools
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Conditionally add the RewardModel worker if it's enabled in the config
        if config.reward_model.enable:
            from verl.workers.fsdp_workers import RewardModelWorker
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
            print("RewardModel is enabled and will be initialized.")
        else:
            print("Warning: RewardModel is disabled. Validation will likely fail or produce meaningless scores.")

        # The reference policy is needed if KL divergence is part of the reward
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
            print("Reference Policy is enabled and will be initialized.")

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # --- 3. Instantiate the Trainer ---
        # The trainer object orchestrates all the workers.
        # We only need the validation dataset for this script.
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
        
        from verl.utils.dataset.rl_dataset import collate_fn

        print("Initializing RayPPOTrainer for validation...")
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            val_reward_fn=val_reward_fn,
            val_dataset=val_dataset,
            collate_fn=collate_fn
            # train_dataset and reward_fn are not needed for pure validation
        )

        # --- 4. Initialize Workers and Run Validation ---
        print("Initializing workers (Actor, Reward Model, etc.)...")
        trainer.init_workers()
        
        # Manually set global_steps since we are not in a training loop.
        # This is required for naming the reward log file.
        trainer.global_steps = 1

        print("Running validation...")
        validation_metrics = trainer._validate()
        
        # --- 5. Print Results ---
        print("\n--- STANDALONE EVALUATION COMPLETE ---")
        print("Validation Metrics:")
        pprint(validation_metrics)
        print("------------------------------------")


if __name__ == "__main__":
    main() 