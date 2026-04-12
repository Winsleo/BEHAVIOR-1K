"""Lightweight result and debug models shared across task_primitives modules."""

import torch as th

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

NUM_EVAL_INSTANCES = 10
DEFAULT_SAMPLING_RADIUS = 1.5
DEFAULT_MAX_SAMPLES = 30
DEFAULT_IMAGE_SIZE = 224
MAX_VERBOSE_SAMPLES = 20

OPTIMAL_DIST_MIN = 0.3
OPTIMAL_DIST_MAX = 0.5
OPTIMAL_DIST_IDEAL = 0.4
TOP_K_CANDIDATES = 3


class GraspResult(Enum):
    """High-level execution outcomes used by both backends and tasks."""
    SUCCESS = "success"
    SAMPLING_FAILED = "sampling_failed"
    NAVIGATION_FAILED = "navigation_failed"
    GRASP_FAILED = "grasp_failed"


class SamplingStrategy(Enum):
    """Candidate base-pose sampling strategies for custom grasp planning."""
    RANDOM = "random"
    UNIFORM_POLAR = "uniform_polar"
    FIBONACCI_SPIRAL = "fibonacci_spiral"
    CONCENTRIC_RINGS = "concentric_rings"


@dataclass
class GraspBasePoseResult:
    pregrasp_pose: Optional[Tuple[th.Tensor, th.Tensor]]
    grasp_pose: Optional[Tuple[th.Tensor, th.Tensor]]
    base_pose_2d: Optional[th.Tensor]
    success: bool
    stats: Dict[str, int]


@dataclass
class PrimitiveExecutionResult:
    """Return value for eager backend execution."""
    success: bool
    backend: str
    result: GraspResult
    error_message: Optional[str] = None
    stats: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraspDebugSnapshot:
    """Single snapshot of the planning context emitted before execution starts."""
    object_name: str
    max_samples: int
    object_rooms: List[Any]
    robot_position: List[float]
    robot_yaw_deg: float
    is_holonomic_base: bool
    is_articulated_trunk: bool
    pregrasp_position: Optional[List[float]] = None
    grasp_position: Optional[List[float]] = None
    inferred_rooms: Optional[List[Any]] = None
    sampling_error: Optional[str] = None


@dataclass
class PoseVisualizationData:
    collision_failed_poses: Optional[List[th.Tensor]] = None
    reachability_failed_poses: Optional[List[th.Tensor]] = None
    valid_poses: Optional[List[th.Tensor]] = None
    selected_pose: Optional[th.Tensor] = None
    target_position: Optional[th.Tensor] = None
    collision_joint_positions: Optional[List[th.Tensor]] = None

    def __post_init__(self):
        if self.collision_failed_poses is None:
            self.collision_failed_poses = []
        if self.reachability_failed_poses is None:
            self.reachability_failed_poses = []
        if self.valid_poses is None:
            self.valid_poses = []


class PoseColors:
    ROBOT_CURRENT = (0.0, 0.5, 1.0, 1.0)
    COLLISION_FAILED = (1.0, 0.2, 0.2, 0.6)
    REACHABILITY_FAILED = (1.0, 0.6, 0.0, 0.6)
    VALID = (0.2, 0.8, 0.2, 0.7)
    SELECTED = (0.8, 0.0, 0.8, 1.0)
    TARGET = (1.0, 1.0, 0.0, 1.0)