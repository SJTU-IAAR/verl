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
import os
from importlib.metadata import PackageNotFoundError, version


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = "vllm"
package_version = get_version(package_name)

###
# package_version = get_version(package_name)
# [SUPPORT AMD:]
# Do not call any torch.cuda* API here, or ray actor creation import class will fail.
if "ROCM_PATH" in os.environ:
    import re

    package_version = version(package_name)
    package_version = re.match(r"(\d+\.\d+\.?\d*)", package_version).group(1)
else:
    package_version = get_version(package_name)
###

if package_version <= "0.6.3":
    vllm_mode = "customized"
    from .fire_vllm_rollout import FIREvLLMRollout  # noqa: F401
    from .vllm_rollout import vLLMRollout  # noqa: F401
    from .tool_vllm_rollout import ToolEnabledVLLMRollout  # noqa: F401
else:
    vllm_mode = "spmd"
    from .vllm_rollout_spmd import vLLMAsyncRollout, vLLMRollout  # noqa: F401
    from .tool_vllm_rollout_spmd import ToolEnabledVLLMRollout  # noqa: F401
