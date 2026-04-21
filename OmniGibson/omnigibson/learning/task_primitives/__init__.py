"""Public exports for the reusable task_primitives package."""

import sys
from pathlib import Path

JOYLO_ROOT = Path(__file__).resolve().parents[5] / "joylo"
# task_primitives reuses joylo configs/helpers but should still import cleanly from OmniGibson.
if JOYLO_ROOT.is_dir() and str(JOYLO_ROOT) not in sys.path:
    sys.path.insert(0, str(JOYLO_ROOT))

from omnigibson.learning.task_primitives.backends import CustomGraspBackend, FallbackGraspBackend, PrimitiveGraspBackend
from omnigibson.learning.task_primitives.context import ActionContext, GraspExecutionConfig, create_action_context
from omnigibson.learning.task_primitives.grasp_primitives import (
    batch_collision_check,
    check_reachability,
    compute_heuristic_score,
    compute_trajectory_complexity,
    compute_trajectory_score,
    execute_controller,
    execute_grasp_motion,
    execute_navigation_trajectory,
    filter_poses_by_room,
    find_optimal_grasp_base_pose,
    generate_candidate_poses,
    move_hand_to_pose,
    plan_navigation,
    report_grasp_debug_context,
    report_sampling_diagnostics,
    restore_trunk,
    select_optimal_pose,
)
from omnigibson.learning.task_primitives.models import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_SAMPLING_RADIUS,
    GraspBasePoseResult,
    GraspDebugSnapshot,
    GraspResult,
    PoseVisualizationData,
    PrimitiveExecutionResult,
    SamplingStrategy,
)
from omnigibson.learning.task_primitives.observers import ConsoleObserver, NullObserver, build_grasp_debug_snapshot
from omnigibson.learning.task_primitives.runtime import (
    apply_low_res_rgb,
    build_grasp_context,
    get_eval_instance_ids,
    load_env,
    load_task_instance,
    resolve_scene_object,
    set_viewer_camera_to_robot,
    shutdown_simulation,
    warmup_environment,
)
from omnigibson.learning.task_primitives.tasks import BaseTask, BDDLSequenceTask
from omnigibson.learning.task_primitives.bddl_task_planner import PrimitiveStep, plan_from_goal

__all__ = [
    "ActionContext",
    "BDDLSequenceTask",
    "BaseTask",
    "CustomGraspBackend",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_SAMPLING_RADIUS",
    "ConsoleObserver",
    "FallbackGraspBackend",
    "GraspBasePoseResult",
    "GraspDebugSnapshot",
    "GraspExecutionConfig",
    "GraspResult",
    "NullObserver",
    "PoseVisualizationData",
    "PrimitiveExecutionResult",
    "PrimitiveGraspBackend",
    "PrimitiveStep",
    "SamplingStrategy",
    "apply_low_res_rgb",
    "batch_collision_check",
    "build_grasp_debug_snapshot",
    "build_grasp_context",
    "check_reachability",
    "compute_heuristic_score",
    "compute_trajectory_complexity",
    "compute_trajectory_score",
    "create_action_context",
    "execute_controller",
    "execute_grasp_motion",
    "execute_navigation_trajectory",
    "filter_poses_by_room",
    "find_optimal_grasp_base_pose",
    "generate_candidate_poses",
    "get_eval_instance_ids",
    "load_env",
    "load_task_instance",
    "move_hand_to_pose",
    "plan_from_goal",
    "plan_navigation",
    "report_grasp_debug_context",
    "report_sampling_diagnostics",
    "resolve_scene_object",
    "restore_trunk",
    "select_optimal_pose",
    "set_viewer_camera_to_robot",
    "shutdown_simulation",
    "warmup_environment",
]