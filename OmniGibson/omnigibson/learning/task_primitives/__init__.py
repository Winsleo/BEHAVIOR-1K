"""Public exports for the reusable task_primitives package.

The package is consumed by ``omnigibson/learning/policies.py`` and
``omnigibson/learning/dagger.py``; the re-exports below mirror the names those
entrypoints import.  Internal collaborators (``ActionContext``,
``BDDLSemanticActionPrimitives``, the grasp-primitive helpers, observers,
models, task protocol, etc.) are reached through their submodules directly.
"""

import sys
from pathlib import Path

JOYLO_ROOT = Path(__file__).resolve().parents[5] / "joylo"
# task_primitives reuses joylo configs/helpers but should still import cleanly from OmniGibson.
if JOYLO_ROOT.is_dir() and str(JOYLO_ROOT) not in sys.path:
    sys.path.insert(0, str(JOYLO_ROOT))

from omnigibson.learning.task_primitives.backends import UnifiedGraspBackend
from omnigibson.learning.task_primitives.bddl_task_planner import plan_from_goal
from omnigibson.learning.task_primitives.context import GraspExecutionConfig, create_action_context
from omnigibson.learning.task_primitives.grasp_primitives import report_grasp_debug_context
from omnigibson.learning.task_primitives.runtime import (
    apply_low_res_rgb,
    get_eval_instance_ids,
    load_task_instance,
    set_viewer_camera_to_robot,
    shutdown_simulation,
    warmup_environment,
)
from omnigibson.learning.task_primitives.tasks import BDDLSequenceTask

__all__ = [
    "BDDLSequenceTask",
    "GraspExecutionConfig",
    "UnifiedGraspBackend",
    "apply_low_res_rgb",
    "create_action_context",
    "get_eval_instance_ids",
    "load_task_instance",
    "plan_from_goal",
    "report_grasp_debug_context",
    "set_viewer_camera_to_robot",
    "shutdown_simulation",
    "warmup_environment",
]
