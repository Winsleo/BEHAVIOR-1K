"""
DAgger (Dataset Aggregation) module for robot learning.

This module provides utilities for:
- Navigation to objects
- Grasping objects
- Environment setup and task instance loading
"""

import argparse
import csv
import json
import math
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
import yaml

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.learning.utils.eval_utils import (
    TASK_NAMES_TO_INDICES,
    generate_basic_environment_config,
)
from omnigibson.macros import gm, macros
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.geometry_utils import wrap_angle
from omnigibson.utils.ui_utils import clear_debug_drawing, draw_box
from omnigibson.utils.visualize_utils import (
    visualize_2d_pose,
    visualize_obstacles,
    visualize_robot_and_sampled_pose,
    visualize_robot_spheres_at_config,
    visualize_sampling_region,
    visualize_trajectory,
)
from gello.robots.sim_robot.og_teleop_cfg import (
    ROBOT_RESET_JOINT_POS,
    DEFAULT_TRUNK_TRANSLATE,
)
from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    load_available_tasks,
    get_task_relevant_room_types,
    infer_torso_qpos_from_trunk_translate,
)

from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.robots.sim_robot.og_teleop_utils import load_available_tasks

# ============================================================================
# Global Configuration
# ============================================================================

# Performance settings
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

# Grasp window for hard grasps
with macros.unlocked():
    macros.robots.manipulation_robot.GRASP_WINDOW = 0.75

# Constants
NUM_EVAL_INSTANCES = 10
DEFAULT_SAMPLING_RADIUS = 1.5
DEFAULT_MAX_SAMPLES = 30
DEFAULT_IMAGE_SIZE = 224
MAX_VERBOSE_SAMPLES = 20

# Pose selection constants
OPTIMAL_DIST_MIN = 0.3  # Too close may cause collision
OPTIMAL_DIST_MAX = 0.5  # Too far may not reach target
OPTIMAL_DIST_IDEAL = 0.4  # Ideal manipulation distance
TOP_K_CANDIDATES = 3  # Number of candidates for trajectory evaluation


# ============================================================================
# Data Classes
# ============================================================================

class GraspResult(Enum):
    """Result status for grasp operations."""
    SUCCESS = "success"
    SAMPLING_FAILED = "sampling_failed"
    NAVIGATION_FAILED = "navigation_failed"
    GRASP_FAILED = "grasp_failed"


@dataclass
class GraspBasePoseResult:
    """Result of finding optimal base pose for grasping."""
    pregrasp_pose: Optional[Tuple[th.Tensor, th.Tensor]]
    grasp_pose: Optional[Tuple[th.Tensor, th.Tensor]]
    base_pose_2d: Optional[th.Tensor]
    success: bool
    stats: Dict[str, int]


# ============================================================================
# Core Functions: Navigate and Grasp
# ============================================================================

def navigate_and_grasp(
    controller: StarterSemanticActionPrimitives,
    env,
    obj,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    verbose: bool = True,
    visualize: bool = False,
) -> GraspResult:
    """
    Navigate to an object and grasp it.

    This function performs the complete sequence:
    1. Sample a valid base pose near the object
    2. Plan and execute navigation to that pose
    3. Move hand to pre-grasp position
    4. Execute grasp
    5. Move hand to final grasp position

    Args:
        controller: StarterSemanticActionPrimitives instance.
        env: OmniGibson environment.
        obj: Target object to grasp.
        max_samples: Maximum sampling attempts for pose near object.
        verbose: Whether to print progress information.
        visualize: Whether to visualize the planning process.

    Returns:
        GraspResult indicating the outcome.
    """
    robot = controller.robot

    # Clear previous visualizations if enabled
    if visualize:
        clear_debug_drawing()

    # Step 1: Sample pose near object
    if verbose:
        print(f"\n{'='*50}")
        print(f"[Step 1/5] Sampling pose near object: {obj.name}")
        print(f"{'='*50}")

    # Visualize obstacles if enabled
    # if visualize:
    #     visualize_obstacles(controller._motion_generator, verbose=verbose)

    grasp_base_result = find_optimal_grasp_base_pose(
        controller=controller,
        obj=obj,
        max_samples=max_samples,
        verbose=verbose,
        visualize=visualize,
    )

    if not grasp_base_result.success:
        if verbose:
            print(f"[FAILED] Could not find optimal base pose for grasping")
        return GraspResult.SAMPLING_FAILED

    pregrasp_pose = grasp_base_result.pregrasp_pose
    grasp_pose = grasp_base_result.grasp_pose
    base_pose_2d = grasp_base_result.base_pose_2d

    # Visualize sampling region and sampled pose
    if visualize:
        visualize_robot_and_sampled_pose(
            robot=robot,
            sampled_pose=base_pose_2d,
            verbose=verbose,
        )
        for _ in range(30):
            og.sim.step()

    # Step 2: Plan navigation
    if verbose:
        print(f"\n{'='*50}")
        print(f"[Step 2/5] Planning navigation to target pose")
        print(f"{'='*50}")
    
    q_traj = plan_navigation(
        controller=controller,
        robot=robot,
        target_pose_2d=base_pose_2d,
        verbose=verbose,
        visualize=visualize,
    )

    if q_traj is None:
        return GraspResult.NAVIGATION_FAILED

    # Visualize planned trajectory
    if visualize and q_traj is not None:
        visualize_trajectory(q_traj=q_traj, robot=robot, verbose=verbose)
        for _ in range(30):
            og.sim.step()

    # Step 3: Execute navigation
    if verbose:
        print(f"\n{'='*50}")
        print(f"[Step 3/5] Executing navigation ({len(q_traj)} waypoints)")
        print(f"{'='*50}")

    for action in controller._execute_motion_plan(q_traj, low_precision=True):
        env.step(action)

    if verbose:
        print(f"[OK] Navigation complete")

    # Step 4: Move to pre-grasp pose
    if verbose:
        print(f"\n{'='*50}")
        print(f"[Step 4/5] Moving hand to pre-grasp position")
        print(f"{'='*50}")

    for action in controller._move_hand(pregrasp_pose):
        env.step(action)

    if verbose:
        print(f"[OK] Hand at pre-grasp position")

    # Step 5: Execute grasp
    if verbose:
        print(f"\n{'='*50}")
        print(f"[Step 5/5] Executing grasp")
        print(f"{'='*50}")

    # 1. Open gripper first
    for action in controller._execute_release():
        env.step(action)

    # 2. Move to grasp position
    for action in controller._move_hand(
        grasp_pose,
        ignore_objects=[obj],
        low_precision=True,
    ):
        env.step(action)

    # 3. Close gripper to grasp
    for action in controller._execute_grasp():
        env.step(action)

    # # 4. Lift object and restore to pre-grasp pose
    # # Use _move_hand_linearly_cartesian to avoid CuRobo attach_objects_to_robot
    # # (which can fail with NoneType mesh for some grasped objects)
    # for action in controller._move_hand_linearly_cartesian(
    #     pregrasp_pose,
    #     stop_on_contact=True,
    #     ignore_failure=True,
    # ):
    #     env.step(action)

    # 5. Restore trunk to initial pose for robots with articulated trunk (R1, R1Pro)
    # Use _execute_motion_plan with full joint positions so trunk controller receives the target
    # (_move_hand_direct_joint only sets arm target, trunk uses compute_no_op_action and stays put)
    if robot.is_articulated_trunk:
        if verbose:
            print(f"\n{'='*50}")
            print(f"[Step 6/6] Restoring trunk to initial pose")
            print(f"{'='*50}")
        initial_trunk = robot.reset_joint_pos[robot.trunk_control_idx]
        start_q = robot.get_joint_positions().clone()
        end_q = start_q.clone()
        end_q[robot.trunk_control_idx] = th.tensor(initial_trunk, dtype=end_q.dtype, device=end_q.device)
        # Interpolate waypoints for smooth trunk motion (avoid fast movement causing oscillation)
        q_traj = th.stack([start_q, end_q])
        q_traj = controller._motion_generator.add_linearly_interpolated_waypoints(
            traj=q_traj, max_inter_dist=0.02
        )
        for action in controller._execute_motion_plan(
            q_traj,
            stop_on_contact=True,
            ignore_failure=True,
            low_precision=True,
        ):
            env.step(action)
        if verbose:
            print(f"[OK] Stand up complete")

    if verbose:
        print(f"[OK] Grasp complete for object: {obj.name}")
        print(f"{'='*50}\n")

    return GraspResult.SUCCESS


# ============================================================================
# Grasp Base Pose Selection Helper Functions
# ============================================================================

class SamplingStrategy(Enum):
    """Sampling strategy for candidate pose generation."""
    RANDOM = "random"                    # Pure random (original behavior)
    UNIFORM_POLAR = "uniform_polar"      # Area-uniform polar sampling
    FIBONACCI_SPIRAL = "fibonacci_spiral" # Fibonacci spiral for very uniform distribution
    CONCENTRIC_RINGS = "concentric_rings" # Concentric rings with uniform angular spacing


def generate_candidate_poses(
    target_pos: th.Tensor,
    num_samples: int,
    sampling_radius: float,
    arm_workspace_offset: float = 0.0,
    min_radius: float = 0.0,
    strategy: SamplingStrategy = SamplingStrategy.FIBONACCI_SPIRAL,
) -> th.Tensor:
    """
    Generate candidate 2D poses around a target position with uniform distribution.

    Args:
        target_pos: Target position (x, y) or (x, y, z).
        num_samples: Number of candidate poses to generate.
        sampling_radius: Maximum distance from target.
        arm_workspace_offset: Offset for yaw calculation based on arm workspace.
        min_radius: Minimum distance from target (default 0.0).
        strategy: Sampling strategy to use (default: FIBONACCI_SPIRAL).

    Returns:
        Tensor of shape (num_samples, 3) with [x, y, yaw] for each candidate.
    """
    if strategy == SamplingStrategy.RANDOM:
        # Original random sampling (non-uniform in area)
        distances = th.rand(num_samples) * (sampling_radius - min_radius) + min_radius
        angles = th.rand(num_samples) * 2 * math.pi - math.pi

    elif strategy == SamplingStrategy.UNIFORM_POLAR:
        # Area-uniform polar sampling: use sqrt(rand) for distance
        # This ensures equal probability per unit area
        t = th.rand(num_samples)
        # Map t to [min_radius^2, sampling_radius^2] then sqrt
        r_squared_min = min_radius ** 2
        r_squared_max = sampling_radius ** 2
        distances = th.sqrt(t * (r_squared_max - r_squared_min) + r_squared_min)
        angles = th.rand(num_samples) * 2 * math.pi - math.pi

    elif strategy == SamplingStrategy.FIBONACCI_SPIRAL:
        # Fibonacci spiral: very uniform distribution using golden angle
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # ~137.5 degrees
        indices = th.arange(num_samples, dtype=th.float32)
        
        # Radial distribution: sqrt for area uniformity
        t = (indices + 0.5) / num_samples  # Offset by 0.5 for better centering
        r_squared_min = min_radius ** 2
        r_squared_max = sampling_radius ** 2
        distances = th.sqrt(t * (r_squared_max - r_squared_min) + r_squared_min)
        
        # Angular distribution: golden angle spiral
        angles = indices * golden_angle
        # Wrap to [-pi, pi]
        angles = th.remainder(angles + math.pi, 2 * math.pi) - math.pi

    elif strategy == SamplingStrategy.CONCENTRIC_RINGS:
        # Concentric rings with uniform angular spacing per ring
        # Number of rings based on sqrt of samples for roughly equal points per unit area
        num_rings = max(1, int(math.sqrt(num_samples)))
        samples_per_ring = num_samples // num_rings
        extra_samples = num_samples % num_rings
        
        distances_list = []
        angles_list = []
        
        for ring_idx in range(num_rings):
            # Radius for this ring (area-uniform spacing)
            t = (ring_idx + 0.5) / num_rings
            r_squared_min = min_radius ** 2
            r_squared_max = sampling_radius ** 2
            ring_radius = math.sqrt(t * (r_squared_max - r_squared_min) + r_squared_min)
            
            # Number of samples in this ring
            n_in_ring = samples_per_ring + (1 if ring_idx < extra_samples else 0)
            
            # Uniform angular spacing with random offset per ring
            angle_offset = th.rand(1).item() * 2 * math.pi
            ring_angles = th.linspace(0, 2 * math.pi, n_in_ring + 1)[:-1] + angle_offset
            ring_angles = th.remainder(ring_angles + math.pi, 2 * math.pi) - math.pi
            
            distances_list.append(th.full((n_in_ring,), ring_radius))
            angles_list.append(ring_angles)
        
        distances = th.cat(distances_list)
        angles = th.cat(angles_list)

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    # Compute x, y positions
    x = target_pos[0] + distances * th.cos(angles)
    y = target_pos[1] + distances * th.sin(angles)
    
    # Yaw: face toward target with arm workspace offset
    yaws = angles + math.pi - arm_workspace_offset

    return th.stack([x, y, yaws], dim=1)


def filter_poses_by_room(
    poses: th.Tensor,
    valid_rooms: List,
    scene_seg_map,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Filter poses to only include those in valid rooms.

    Args:
        poses: Candidate poses tensor of shape (N, 3).
        valid_rooms: List of valid room identifiers.
        scene_seg_map: Scene segmentation map for room queries.

    Returns:
        Tuple of (filtered_poses, valid_indices).
    """
    num_poses = len(poses)
    valid_mask = th.zeros(num_poses, dtype=th.bool)

    for i in range(num_poses):
        room = scene_seg_map.get_room_instance_by_point(poses[i, :2])
        valid_mask[i] = room in valid_rooms

    valid_indices = th.where(valid_mask)[0]
    return poses[valid_indices], valid_indices


def convert_poses_to_joint_positions(
    poses: th.Tensor,
    robot,
    current_joint_pos: th.Tensor,
) -> th.Tensor:
    """
    Convert 2D world poses to robot joint positions.

    Args:
        poses: 2D poses tensor of shape (N, 3) with [x, y, yaw].
        robot: Robot instance.
        current_joint_pos: Current joint positions tensor.

    Returns:
        Batch joint positions tensor of shape (N, num_joints).
    """
    num_poses = len(poses)
    batch_joint_positions = current_joint_pos.unsqueeze(0).repeat(num_poses, 1)

    if robot.is_holonomic_base:
        root_pos, root_quat = robot.root_link.get_position_orientation()

        for i, pose_2d in enumerate(poses):
            target_pos = th.tensor([pose_2d[0], pose_2d[1], root_pos[2]])
            target_quat = T.euler2quat(th.tensor([0.0, 0.0, pose_2d[2]]))

            inv_root_pos, inv_root_quat = T.invert_pose_transform(root_pos, root_quat)
            relative_pos, relative_quat = T.pose_transform(
                inv_root_pos, inv_root_quat, target_pos, target_quat
            )
            relative_euler = T.quat2euler(relative_quat)

            batch_joint_positions[i, robot.base_control_idx[0]] = relative_pos[0]
            batch_joint_positions[i, robot.base_control_idx[1]] = relative_pos[1]
            batch_joint_positions[i, robot.base_control_idx[2]] = relative_euler[2]
    else:
        batch_joint_positions[:, robot.base_control_idx] = poses

    return batch_joint_positions


def batch_collision_check(
    controller: StarterSemanticActionPrimitives,
    batch_joint_positions: th.Tensor,
    attached_obj: Optional[Dict] = None,
) -> th.Tensor:
    """
    Perform batch collision checking for joint positions.

    Args:
        controller: StarterSemanticActionPrimitives instance.
        batch_joint_positions: Joint positions tensor of shape (N, num_joints).
        attached_obj: Optional attached object for collision checking.

    Returns:
        Boolean tensor of shape (N,) indicating collision status.
    """
    return controller._motion_generator.check_collisions(
        batch_joint_positions,
        self_collision_check=False,
        skip_obstacle_update=True,
        attached_obj=attached_obj,
    ).cpu()


def check_reachability(
    controller: StarterSemanticActionPrimitives,
    poses: th.Tensor,
    batch_joint_positions: th.Tensor,
    collision_free_indices: th.Tensor,
    target_pose: Tuple[th.Tensor, th.Tensor],
    original_indices: th.Tensor,
    verbose: bool = False,
) -> Tuple[List[Tuple[th.Tensor, int]], List[th.Tensor], int]:
    """
    Check reachability for collision-free poses.

    Args:
        controller: StarterSemanticActionPrimitives instance.
        poses: Filtered poses tensor.
        batch_joint_positions: Corresponding joint positions.
        collision_free_indices: Indices of collision-free poses.
        target_pose: Target end-effector pose.
        original_indices: Original global indices.
        verbose: Whether to print progress.

    Returns:
        Tuple of (valid_poses_list, failed_poses_list, failure_count).
    """
    robot_pos, _ = controller.robot.get_position_orientation()
    robot_xy = robot_pos[:2]

    valid_poses = []
    failed_poses = []
    failure_count = 0

    for local_idx in collision_free_indices:
        candidate_pose = poses[local_idx]
        joint_pos = batch_joint_positions[local_idx]

        if controller._target_in_reach_of_robot(
            target_pose, initial_joint_pos=joint_pos, skip_obstacle_update=True
        ):
            global_idx = original_indices[local_idx].item()
            valid_poses.append((candidate_pose.clone(), global_idx))
            if verbose:
                dist = th.norm(candidate_pose[:2] - robot_xy).item()
                print(f"    Sample {global_idx}: Valid at ({candidate_pose[0].item():.2f}, "
                      f"{candidate_pose[1].item():.2f}), dist={dist:.2f}m")
        else:
            failure_count += 1
            failed_poses.append(candidate_pose.clone())
            if verbose and len(failed_poses) <= MAX_VERBOSE_SAMPLES:
                print(f"    Sample {original_indices[local_idx].item()}: Not reachable at "
                      f"({candidate_pose[0].item():.2f}, {candidate_pose[1].item():.2f})")

    return valid_poses, failed_poses, failure_count


def compute_heuristic_score(
    pose_2d: th.Tensor,
    target_xy: th.Tensor,
    robot_xy: th.Tensor,
    robot_yaw: float,
) -> float:
    """
    Compute a heuristic score for a candidate pose (lower is better).

    Args:
        pose_2d: Candidate pose [x, y, yaw].
        target_xy: Target object position [x, y].
        robot_xy: Robot current position [x, y].
        robot_yaw: Robot current yaw angle.

    Returns:
        Heuristic score (lower is better).
    """
    pose_xy = pose_2d[:2]
    pose_yaw = pose_2d[2].item()

    # Factor 1: Distance to target
    dist_to_target = th.norm(pose_xy - target_xy).item()
    if dist_to_target < OPTIMAL_DIST_MIN:
        dist_cost = (OPTIMAL_DIST_MIN - dist_to_target) * 6.0
    elif dist_to_target > OPTIMAL_DIST_MAX:
        dist_cost = (dist_to_target - OPTIMAL_DIST_MAX) * 6.0
    else:
        dist_cost = abs(dist_to_target - OPTIMAL_DIST_IDEAL) * 1.0

    # Factor 2: Facing target
    target_direction = th.atan2(
        target_xy[1] - pose_xy[1],
        target_xy[0] - pose_xy[0]
    ).item()
    facing_error = abs(wrap_angle(pose_yaw - target_direction))
    facing_cost = facing_error * 0.3

    # Factor 3: Travel distance
    travel_dist = th.norm(pose_xy - robot_xy).item()
    travel_cost = travel_dist * 1.0

    # Factor 4: Rotation change
    yaw_change = abs(wrap_angle(pose_yaw - robot_yaw))
    rotation_cost = yaw_change * 0.3

    return dist_cost + facing_cost + travel_cost + rotation_cost


def compute_trajectory_complexity(
    q_traj: th.Tensor,
    robot,
) -> Tuple[float, float]:
    """
    Compute trajectory complexity based on path length and cumulative rotation.

    Args:
        q_traj: Joint trajectory tensor.
        robot: Robot instance.

    Returns:
        Tuple of (path_length, cumulative_rotation) in meters and radians.
    """
    if q_traj is None or len(q_traj) < 2:
        return float('inf'), float('inf')

    base_idx = robot.base_idx
    path_length = 0.0
    cumulative_rotation = 0.0

    root_pos, root_quat = robot.root_link.get_position_orientation()
    prev_world_pos = None
    prev_yaw = None

    for q in q_traj:
        joint_x = q[base_idx[0]].item()
        joint_y = q[base_idx[1]].item()
        joint_yaw = q[base_idx[5]].item()

        local_pos = th.tensor([joint_x, joint_y, 0.0])
        local_quat = th.tensor([0.0, 0.0, 0.0, 1.0])
        world_pos, _ = T.pose_transform(root_pos, root_quat, local_pos, local_quat)

        if prev_world_pos is not None:
            path_length += th.norm(world_pos[:2] - prev_world_pos[:2]).item()
            cumulative_rotation += abs(wrap_angle(joint_yaw - prev_yaw))

        prev_world_pos = world_pos
        prev_yaw = joint_yaw

    return path_length, cumulative_rotation


def compute_trajectory_score(
    path_length: float,
    cumulative_rotation: float,
    dist_to_target: float,
) -> float:
    """
    Compute final trajectory-based score.

    Args:
        path_length: Total path length in meters.
        cumulative_rotation: Total rotation in radians.
        dist_to_target: Distance from pose to target object.

    Returns:
        Trajectory score (lower is better).
    """
    score = path_length + cumulative_rotation * 0.5

    # Add manipulation quality penalty
    if dist_to_target < OPTIMAL_DIST_MIN:
        score += (OPTIMAL_DIST_MIN - dist_to_target) * 3.0
    elif dist_to_target > OPTIMAL_DIST_MAX:
        score += (dist_to_target - OPTIMAL_DIST_MAX) * 3.0

    return score


def select_optimal_pose(
    valid_poses: List[Tuple[th.Tensor, int]],
    target_xy: th.Tensor,
    robot_xy: th.Tensor,
    robot_yaw: float,
    controller: StarterSemanticActionPrimitives,
    robot,
    verbose: bool = False,
) -> Tuple[th.Tensor, int, Optional[Tuple[float, float, int]]]:
    """
    Select the optimal pose using two-stage evaluation.

    Stage 1: Heuristic filtering to select top-K candidates.
    Stage 2: Trajectory-based evaluation for final selection.

    Args:
        valid_poses: List of (pose_2d, global_idx) tuples.
        target_xy: Target object position.
        robot_xy: Robot current position.
        robot_yaw: Robot current yaw.
        controller: StarterSemanticActionPrimitives instance.
        robot: Robot instance.
        verbose: Whether to print progress.

    Returns:
        Tuple of (best_pose, best_idx, trajectory_info).
        trajectory_info is (path_length, cumulative_rotation, num_waypoints) or None.
    """
    # Stage 1: Heuristic filtering
    scored_poses = [
        (pose, idx, compute_heuristic_score(pose, target_xy, robot_xy, robot_yaw))
        for pose, idx in valid_poses
    ]
    scored_poses.sort(key=lambda x: x[2])
    top_k_poses = scored_poses[:min(TOP_K_CANDIDATES, len(scored_poses))]

    if verbose:
        print(f"\n  Stage 1: Selected top-{len(top_k_poses)} candidates from {len(valid_poses)} valid poses")
        print(f"  Stage 2: Evaluating trajectory complexity...")

    # Stage 2: Trajectory evaluation
    best_pose = None
    best_idx = None
    best_traj_score = float('inf')
    best_traj_info = None

    for i, (pose_2d, global_idx, _) in enumerate(top_k_poses):
        try:
            q_traj = plan_navigation(
                controller=controller,
                robot=robot,
                target_pose_2d=pose_2d,
                verbose=False,
                visualize=False,
            )
        except Exception:
            q_traj = None

        if q_traj is None:
            if verbose:
                print(f"    Candidate {i+1}: Planning failed, skipped")
            continue

        path_length, cumulative_rotation = compute_trajectory_complexity(q_traj, robot)
        dist_to_target = th.norm(pose_2d[:2] - target_xy).item()
        traj_score = compute_trajectory_score(path_length, cumulative_rotation, dist_to_target)

        if verbose:
            print(f"    Candidate {i+1}: path={path_length:.2f}m, "
                  f"rotation={math.degrees(cumulative_rotation):.1f}deg, score={traj_score:.3f}")

        if traj_score < best_traj_score:
            best_traj_score = traj_score
            best_pose = pose_2d
            best_idx = global_idx
            best_traj_info = (path_length, cumulative_rotation, len(q_traj))

    # Fallback to heuristic best if all planning failed
    if best_pose is None:
        if verbose:
            print(f"  Warning: All trajectory planning failed, using heuristic best")
        best_pose, best_idx, _ = top_k_poses[0]
        best_traj_info = None

    return best_pose, best_idx, best_traj_info


# ============================================================================
# Main Grasp Base Pose Selection Function
# ============================================================================

def find_optimal_grasp_base_pose(
    controller: StarterSemanticActionPrimitives,
    obj,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    sampling_radius: float = DEFAULT_SAMPLING_RADIUS,
    verbose: bool = True,
    visualize: bool = False,
    seed: int = 42,
) -> GraspBasePoseResult:
    """
    Find the optimal robot base pose for grasping a target object.

    This function performs a multi-stage search to find the best base pose:
    1. Sample candidate poses around the target object
    2. Filter by room constraints (same room as object)
    3. Batch collision checking for robot base
    4. Verify arm reachability to target
    5. Two-stage optimal selection:
       - Stage 1: Heuristic filtering (distance, facing, travel cost)
       - Stage 2: Trajectory complexity evaluation (actual path length, rotation)

    The selected pose optimizes for:
    - Manipulation quality: Optimal distance to target (0.3-0.5m)
    - Movement efficiency: Minimal travel distance and rotation
    - Trajectory simplicity: Minimal path length and cumulative rotation

    Args:
        controller: StarterSemanticActionPrimitives instance.
        obj: Target object to grasp.
        max_samples: Maximum number of candidate poses to generate.
        sampling_radius: Maximum distance from object center to sample.
        verbose: Whether to print progress information.
        visualize: Whether to visualize failed poses for debugging.
        seed: Random seed for reproducibility.

    Returns:
        GraspBasePoseResult containing:
        - pregrasp_pose: End-effector pre-grasp pose
        - grasp_pose: End-effector grasp pose
        - base_pose_2d: Optimal robot base pose [x, y, yaw]
        - success: Whether a valid pose was found
        - stats: Filtering statistics (failures by category)
    """
    th.manual_seed(seed)

    robot = controller.robot
    arm = controller.arm

    stats = {
        "total_attempts": 0,
        "collision_failures": 0,
        "reachability_failures": 0,
        "room_failures": 0,
    }

    # Step 1: Sample grasp pose
    if verbose:
        print(f"  Sampling grasp pose for object...")

    try:
        eef_pose, grasp_pose = controller._sample_grasp_pose(obj)
        if visualize:
            visualize_sampling_region(
                obj=obj,
                eef_pose=eef_pose,
                radius=DEFAULT_SAMPLING_RADIUS,
                verbose=verbose,
            )
        if verbose:
            print(f"  Pre-grasp position: {eef_pose[0].tolist()}")
            print(f"  Grasp position: {grasp_pose[0].tolist()}")
    except Exception as e:
        if verbose:
            print(f"  [ERROR] Grasp pose sampling failed: {e}")
        return GraspBasePoseResult(
            pregrasp_pose=None, grasp_pose=None, base_pose_2d=None,
            success=False, stats=stats,
        )

    target_pose = eef_pose
    target_xy = target_pose[0][:2]

    # Step 2: Determine valid rooms
    obj_rooms = obj.in_rooms if obj.in_rooms else [
        robot.scene._seg_map.get_room_instance_by_point(target_pose[0][:2])
    ]
    if verbose:
        print(f"  Object room(s): {obj_rooms}")

    # Step 3: Generate candidate poses
    avg_arm_workspace_range = th.mean(robot.arm_workspace_range[arm])
    if verbose:
        print(f"  Sampling distance range: [0.00, {sampling_radius:.2f}]m")
        print(f"  Average arm workspace range: {avg_arm_workspace_range:.3f}")
        print(f"  Generating {max_samples} candidate poses...")

    controller._motion_generator.update_obstacles()

    candidate_poses = generate_candidate_poses(
        target_pos=target_pose[0],
        num_samples=max_samples,
        sampling_radius=sampling_radius,
        arm_workspace_offset=avg_arm_workspace_range.item(),
    )

    # Step 4: Filter by room
    room_valid_poses, room_valid_indices = filter_poses_by_room(
        poses=candidate_poses,
        valid_rooms=obj_rooms,
        scene_seg_map=robot.scene._seg_map,
    )
    stats["room_failures"] = max_samples - len(room_valid_indices)

    if verbose:
        print(f"  Room filter: {len(room_valid_indices)}/{max_samples} passed")

    if len(room_valid_indices) == 0:
        if verbose:
            print(f"\n  [FAILED] All samples failed room check")
        return GraspBasePoseResult(
            pregrasp_pose=None, grasp_pose=None, base_pose_2d=None,
            success=False, stats=stats,
        )

    # Step 5: Convert to joint positions
    current_joint_pos = robot.get_joint_positions()
    batch_joint_positions = convert_poses_to_joint_positions(
        poses=room_valid_poses,
        robot=robot,
        current_joint_pos=current_joint_pos,
    )

    # Step 6: Batch collision check
    obj_in_hand = controller._get_obj_in_hand()
    attached_obj = (
        {robot.eef_link_names[arm]: obj_in_hand.root_link}
        if obj_in_hand else None
    )

    if verbose:
        print(f"  Running batch collision check for {len(room_valid_indices)} candidates...")

    collision_results = batch_collision_check(
        controller=controller,
        batch_joint_positions=batch_joint_positions,
        attached_obj=attached_obj,
    )

    collision_free_indices = th.where(~collision_results)[0]
    stats["collision_failures"] = int(collision_results.sum().item())

    if verbose:
        print(f"  Collision filter: {len(collision_free_indices)}/{len(room_valid_indices)} passed")

    # Collect failed poses for visualization
    collision_failed_poses = [
        room_valid_poses[i].clone() for i in range(len(room_valid_poses))
        if collision_results[i].item()
    ]
    collision_failed_joint_positions = [
        batch_joint_positions[i].clone() for i in range(len(batch_joint_positions))
        if collision_results[i].item()
    ]

    if len(collision_free_indices) == 0:
        stats["total_attempts"] = max_samples
        if verbose:
            print(f"\n  [FAILED] All samples failed collision check")
            print(f"    Collision failures: {stats['collision_failures']}")
            print(f"    Room failures: {stats['room_failures']}")
        if visualize:
            _visualize_failed_poses(
                robot, collision_failed_poses, [],
                max_display=MAX_VERBOSE_SAMPLES,
                motion_generator=controller._motion_generator,
                collision_joint_positions=collision_failed_joint_positions,
            )
        return GraspBasePoseResult(
            pregrasp_pose=None, grasp_pose=None, base_pose_2d=None,
            success=False, stats=stats,
        )

    # Step 7: Check reachability
    if verbose:
        print(f"  Checking reachability for {len(collision_free_indices)} collision-free candidates...")

    valid_poses, reachability_failed_poses, reachability_failures = check_reachability(
        controller=controller,
        poses=room_valid_poses,
        batch_joint_positions=batch_joint_positions,
        collision_free_indices=collision_free_indices,
        target_pose=target_pose,
        original_indices=room_valid_indices,
        verbose=verbose,
    )
    stats["reachability_failures"] = reachability_failures

    # Step 8: Select optimal pose
    if valid_poses:
        robot_pos, robot_quat = robot.get_position_orientation()
        robot_xy = robot_pos[:2]
        robot_yaw = T.quat2euler(robot_quat)[2].item()

        best_pose, best_idx, best_traj_info = select_optimal_pose(
            valid_poses=valid_poses,
            target_xy=target_xy,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            controller=controller,
            robot=robot,
            verbose=verbose,
        )

        # Compute final stats
        best_travel_dist = th.norm(best_pose[:2] - robot_xy).item()
        best_yaw_change = abs(wrap_angle(best_pose[2].item() - robot_yaw))
        best_dist_to_target = th.norm(best_pose[:2] - target_xy).item()
        target_direction = th.atan2(
            target_xy[1] - best_pose[1],
            target_xy[0] - best_pose[0]
        ).item()
        best_facing_error = abs(wrap_angle(best_pose[2].item() - target_direction))

        stats["total_attempts"] = max_samples

        if verbose:
            print(f"\n  [SUCCESS] Selected optimal pose based on trajectory complexity")
            print(f"    Collision failures: {stats['collision_failures']}")
            print(f"    Reachability failures: {stats['reachability_failures']}")
            print(f"    Room failures: {stats['room_failures']}")
            print(f"    Selected pose: ({best_pose[0].item():.3f}, {best_pose[1].item():.3f}, "
                  f"{best_pose[2].item():.3f})")
            print(f"    Distance to target: {best_dist_to_target:.3f}m "
                  f"(optimal: {OPTIMAL_DIST_MIN:.1f}-{OPTIMAL_DIST_MAX:.1f}m)")
            print(f"    Straight-line: {best_travel_dist:.3f}m, End rotation: "
                  f"{math.degrees(best_yaw_change):.1f}deg, Facing error: "
                  f"{math.degrees(best_facing_error):.1f}deg")
            if best_traj_info:
                print(f"    Actual trajectory: path={best_traj_info[0]:.3f}m, "
                      f"cumulative rotation={math.degrees(best_traj_info[1]):.1f}deg, "
                      f"waypoints={best_traj_info[2]}")

        # Visualize all poses if requested
        if visualize:
            viz_data = PoseVisualizationData(
                collision_failed_poses=collision_failed_poses,
                reachability_failed_poses=reachability_failed_poses,
                valid_poses=[pose for pose, _ in valid_poses],
                selected_pose=best_pose,
                target_position=target_xy,
                collision_joint_positions=collision_failed_joint_positions,
            )
            visualize_all_candidate_poses(
                robot=robot,
                viz_data=viz_data,
                max_display_per_category=MAX_VERBOSE_SAMPLES,
                motion_generator=controller._motion_generator,
                show_collision_spheres=False,
            )

        return GraspBasePoseResult(
            pregrasp_pose=eef_pose,
            grasp_pose=grasp_pose,
            base_pose_2d=best_pose,
            success=True,
            stats=stats,
        )

    # Sampling failed
    stats["total_attempts"] = max_samples
    if verbose:
        print(f"\n  [FAILED] No valid pose found after {max_samples} attempts")
        print(f"    Collision failures: {stats['collision_failures']}")
        print(f"    Reachability failures: {stats['reachability_failures']}")
        print(f"    Room failures: {stats['room_failures']}")

    if visualize:
        viz_data = PoseVisualizationData(
            collision_failed_poses=collision_failed_poses,
            reachability_failed_poses=reachability_failed_poses,
            valid_poses=[],
            selected_pose=None,
            target_position=target_xy,
            collision_joint_positions=collision_failed_joint_positions,
        )
        visualize_all_candidate_poses(
            robot=robot,
            viz_data=viz_data,
            max_display_per_category=MAX_VERBOSE_SAMPLES,
            motion_generator=controller._motion_generator,
            show_collision_spheres=True,
        )

    return GraspBasePoseResult(
        pregrasp_pose=None,
        grasp_pose=None,
        base_pose_2d=None,
        success=False,
        stats=stats,
    )


def plan_navigation(
    controller: StarterSemanticActionPrimitives,
    robot,
    target_pose_2d: th.Tensor,
    verbose: bool = True,
    visualize: bool = False,
) -> Optional[th.Tensor]:
    """
    Plan navigation trajectory to a target 2D pose.

    Args:
        controller: StarterSemanticActionPrimitives instance.
        robot: Robot object.
        target_pose_2d: Target 2D pose (x, y, yaw).
        verbose: Whether to print progress information.
        visualize: Whether to visualize debug info when planning fails.

    Returns:
        Joint trajectory tensor if successful, None otherwise.
    """
    # Convert 2D pose to 3D pose
    pose_3d = controller._get_robot_pose_from_2d_pose(target_pose_2d)
    base_link_pos = robot.links[robot.base_footprint_link_name].get_position_orientation()[0]
    pose_3d = (pose_3d[0].clone(), pose_3d[1])
    pose_3d[0][2] = base_link_pos[2]
    target_pos = {robot.base_footprint_link_name: pose_3d[0]}
    target_quat = {robot.base_footprint_link_name: pose_3d[1]}

    if verbose:
        print(f"  Target position: {pose_3d[0].tolist()}")
        print(f"  Target orientation: {pose_3d[1].tolist()}")

    # Plan motion - catch exception if planning fails
    # Enable graph search for base navigation to handle obstacles
    q_traj = None
    planning_error_msg = None
    try:
        q_traj = controller._plan_joint_motion(
            target_pos=target_pos,
            target_quat=target_quat,
            embodiment_selection=CuRoboEmbodimentSelection.BASE,
            skip_obstacle_update=True
        )
    except ActionPrimitiveError as e:
        planning_error_msg = str(e)

    if q_traj is not None:
        if verbose:
            print(f"  [OK] Planned trajectory with {len(q_traj)} waypoints")
    else:
        # Planning failed - provide debug information
        if verbose:
            print(f"\n  [FAILED] Navigation planning failed")
            if planning_error_msg:
                print(f"  Error: {planning_error_msg}")
            robot_pos, robot_quat = robot.get_position_orientation()
            robot_yaw = T.quat2euler(robot_quat)[2].item()
            print(f"  Debug info:")
            print(f"    Robot current position: ({robot_pos[0].item():.3f}, {robot_pos[1].item():.3f}, {robot_pos[2].item():.3f})")
            print(f"    Robot current yaw: {math.degrees(robot_yaw):.1f} deg")
            print(f"    Target position: ({target_pose_2d[0].item():.3f}, {target_pose_2d[1].item():.3f})")
            print(f"    Target yaw: {math.degrees(target_pose_2d[2].item()):.1f} deg")
            distance = math.sqrt((target_pose_2d[0].item() - robot_pos[0].item())**2 + 
                                  (target_pose_2d[1].item() - robot_pos[1].item())**2)
            print(f"    Distance to target: {distance:.3f}m")
            print(f"  Possible causes:")
            print(f"    - Obstacles blocking the path")
            print(f"    - Target position is in collision")
            print(f"    - IK solution not found for base motion")
        
        if visualize:
            print(f"\n  Visualizing navigation failure...")
            # Visualize current robot pose and target pose
            visualize_robot_and_sampled_pose(
                robot=robot,
                sampled_pose=target_pose_2d,
                verbose=False,
            )
            # Visualize obstacles
            visualize_obstacles(
                motion_generator=controller._motion_generator,
                verbose=True,
            )
            # Allow user to inspect
            og.sim.enable_viewer_camera_teleoperation()
            print(f"  Press Ctrl+C to exit visualization...")
            try:
                while True:
                    og.sim.step()
            except KeyboardInterrupt:
                print(f"  Visualization ended by user.")

    return q_traj


@dataclass
class PoseVisualizationData:
    """Data container for pose visualization."""
    collision_failed_poses: List[th.Tensor] = None
    reachability_failed_poses: List[th.Tensor] = None
    valid_poses: List[th.Tensor] = None
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


# Visualization color scheme
class PoseColors:
    """Color constants for pose visualization (RGBA)."""
    ROBOT_CURRENT = (0.0, 0.5, 1.0, 1.0)      # Blue - robot's current pose
    COLLISION_FAILED = (1.0, 0.2, 0.2, 0.6)   # Red - collision failures
    REACHABILITY_FAILED = (1.0, 0.6, 0.0, 0.6) # Orange - reachability failures
    VALID = (0.2, 0.8, 0.2, 0.7)              # Green - valid poses
    SELECTED = (0.8, 0.0, 0.8, 1.0)           # Purple - selected optimal pose
    TARGET = (1.0, 1.0, 0.0, 1.0)             # Yellow - target object position


def visualize_all_candidate_poses(
    robot,
    viz_data: PoseVisualizationData,
    z_height: float = 0.1,
    max_display_per_category: int = 20,
    motion_generator=None,
    show_collision_spheres: bool = True,
) -> None:
    """
    Visualize all candidate poses with different colors for each category.

    Color scheme:
    - Blue (large arrow): Robot's current pose
    - Red (small arrows): Collision failures
    - Orange (small arrows): Reachability failures  
    - Green (medium arrows): Valid poses (passed all checks)
    - Purple (large arrow + box): Selected optimal pose
    - Yellow (box): Target object position

    Args:
        robot: Robot object for visualizing current robot pose.
        viz_data: PoseVisualizationData containing all pose categories.
        z_height: Base Z-coordinate for visualization.
        max_display_per_category: Maximum poses to display per category.
        motion_generator: Optional CuRoboMotionGenerator for sphere visualization.
        show_collision_spheres: Whether to show collision spheres for first failure.
    """
    counts = {
        "collision_failed": 0,
        "reachability_failed": 0,
        "valid": 0,
    }

    # 1. Visualize target position (yellow box)
    if viz_data.target_position is not None:
        target_xy = viz_data.target_position[:2]
        draw_box(
            center=[target_xy[0].item(), target_xy[1].item(), z_height],
            extents=[0.15, 0.15, 0.02],
            color=PoseColors.TARGET,
            size=3.0,
        )

    # 2. Visualize robot's current pose (blue, large arrow)
    robot_pos, robot_quat = robot.get_position_orientation()
    robot_yaw = T.quat2euler(robot_quat)[2].item()
    robot_2d_pose = th.tensor([robot_pos[0], robot_pos[1], robot_yaw])
    visualize_2d_pose(
        robot_2d_pose, 
        z_height=z_height + 0.05, 
        arrow_length=0.5,
        color=PoseColors.ROBOT_CURRENT, 
        verbose=False
    )

    # 3. Visualize collision failed poses (red, small arrows)
    collision_to_show = viz_data.collision_failed_poses[:max_display_per_category]
    for i, pose in enumerate(collision_to_show):
        visualize_2d_pose(
            pose,
            z_height=z_height,
            arrow_length=0.2,
            color=PoseColors.COLLISION_FAILED,
            verbose=False,
        )
    counts["collision_failed"] = len(collision_to_show)

    # 4. Visualize reachability failed poses (orange, small arrows)
    reachability_to_show = viz_data.reachability_failed_poses[:max_display_per_category]
    for i, pose in enumerate(reachability_to_show):
        visualize_2d_pose(
            pose,
            z_height=z_height + 0.01,
            arrow_length=0.2,
            color=PoseColors.REACHABILITY_FAILED,
            verbose=False,
        )
    counts["reachability_failed"] = len(reachability_to_show)

    # 5. Visualize valid poses (green, medium arrows)
    valid_to_show = viz_data.valid_poses[:max_display_per_category]
    for i, pose in enumerate(valid_to_show):
        visualize_2d_pose(
            pose,
            z_height=z_height + 0.02,
            arrow_length=0.35,
            color=PoseColors.VALID,
            verbose=False,
        )
    counts["valid"] = len(valid_to_show)

    # 6. Visualize selected optimal pose (purple, large arrow + highlighted box)
    if viz_data.selected_pose is not None:
        selected = viz_data.selected_pose
        # Draw highlighted box at selected position
        draw_box(
            center=[selected[0].item(), selected[1].item(), z_height + 0.03],
            extents=[0.2, 0.2, 0.02],
            color=PoseColors.SELECTED,
            size=3.0,
        )
        # Draw large arrow for selected pose
        visualize_2d_pose(
            selected,
            z_height=z_height + 0.04,
            arrow_length=0.5,
            color=PoseColors.SELECTED,
            verbose=False,
        )

    # 7. Optionally show collision spheres for first collision failure
    if (show_collision_spheres and motion_generator is not None and 
        viz_data.collision_joint_positions and len(viz_data.collision_joint_positions) > 0):
        print(f"\n  Visualizing collision spheres for first collision failure...")
        visualize_robot_spheres_at_config(
            motion_generator=motion_generator,
            joint_positions=viz_data.collision_joint_positions[0],
            verbose=True,
        )

    # Print legend
    print(f"\n  === Pose Visualization Legend ===")
    print(f"    Blue arrow      : Robot current pose")
    print(f"    Yellow box      : Target object position")
    print(f"    Red arrows   ({counts['collision_failed']:3d}): Collision failures")
    print(f"    Orange arrows({counts['reachability_failed']:3d}): Reachability failures")
    print(f"    Green arrows ({counts['valid']:3d}): Valid poses")
    if viz_data.selected_pose is not None:
        print(f"    Purple arrow    : Selected optimal pose")
    
    total_shown = counts['collision_failed'] + counts['reachability_failed'] + counts['valid']
    total_actual = (len(viz_data.collision_failed_poses) + 
                   len(viz_data.reachability_failed_poses) + 
                   len(viz_data.valid_poses))
    if total_shown < total_actual:
        print(f"\n    (Showing {total_shown}/{total_actual} poses, limit={max_display_per_category} per category)")


def _visualize_failed_poses(
    robot,
    collision_poses: List[th.Tensor],
    reachability_poses: List[th.Tensor],
    z_height: float = 0.1,
    max_display: int = 10,
    motion_generator=None,
    collision_joint_positions: Optional[List[th.Tensor]] = None,
) -> None:
    """
    Visualize failed candidate poses for debugging (legacy interface).

    This is a simplified wrapper around visualize_all_candidate_poses for
    backward compatibility.

    Args:
        robot: Robot object for visualizing current robot pose.
        collision_poses: List of 2D poses (x, y, yaw) that failed collision check.
        reachability_poses: List of 2D poses (x, y, yaw) that failed reachability check.
        z_height: Z-coordinate for visualization.
        max_display: Maximum number of poses to display per category.
        motion_generator: Optional CuRoboMotionGenerator for sphere visualization.
        collision_joint_positions: Optional list of joint positions for collision failures.
    """
    viz_data = PoseVisualizationData(
        collision_failed_poses=collision_poses,
        reachability_failed_poses=reachability_poses,
        collision_joint_positions=collision_joint_positions,
    )
    
    visualize_all_candidate_poses(
        robot=robot,
        viz_data=viz_data,
        z_height=z_height,
        max_display_per_category=max_display,
        motion_generator=motion_generator,
        show_collision_spheres=(collision_joint_positions is not None),
    )
    og.sim.enable_viewer_camera_teleoperation()
    print(f"\n  Press Ctrl+C to exit visualization...")
    try:
        while True:
            og.sim.step()
    except KeyboardInterrupt:
        print(f"\n  Visualization ended by user.")


# ============================================================================
# Utility Functions
# ============================================================================

def get_2d_pose_from_3d_pose(pose_3d: Tuple[th.Tensor, th.Tensor]) -> th.Tensor:
    """
    Extract 2D pose (x, y, yaw) from a 3D pose.

    Args:
        pose_3d: Tuple of (position, quaternion).

    Returns:
        2D pose tensor (x, y, yaw).
    """
    pos, quat = pose_3d
    pos = th.as_tensor(pos)
    quat = th.as_tensor(quat)
    yaw = T.z_angle_from_quat(quat).squeeze()
    return th.stack([pos[0], pos[1], yaw])


# ============================================================================
# Environment Setup Functions
# ============================================================================

def load_task_instance(env, robot, instance_id: int, test_hidden: bool = False) -> None:
    """
    Load a specific task instance state, mirroring eval.py logic.

    Args:
        env: OmniGibson environment.
        robot: Robot object.
        instance_id: Task instance ID.
        test_hidden: Whether to use hidden test instances.
    """
    scene_model = env.task.scene_name
    tro_filename = env.task.get_cached_activity_scene_filename(
        scene_model=scene_model,
        activity_name=env.task.activity_name,
        activity_definition_id=env.task.activity_definition_id,
        activity_instance_id=instance_id,
    )

    if test_hidden:
        tro_file_path = os.path.join(
            gm.DATA_PATH,
            "2025-challenge-test-instances",
            env.task.activity_name,
            f"{tro_filename}-tro_state.json",
        )
    else:
        task_instance_root = get_task_instance_path(scene_model)
        if task_instance_root is None:
            raise FileNotFoundError(f"Task instance path not found for scene: {scene_model}")
        tro_file_path = os.path.join(
            task_instance_root,
            f"json/{scene_model}_task_{env.task.activity_name}_instances/{tro_filename}-tro_state.json",
        )

    with open(tro_file_path, "r") as f:
        tro_state = recursively_convert_to_torch(json.load(f))

    for tro_key, state in tro_state.items():
        if tro_key == "robot_poses":
            presampled_robot_poses = state
            presampled_robot_poses = {k.lower(): v for k, v in presampled_robot_poses.items()}
            robot_pos = presampled_robot_poses[robot.model][0]["position"]
            robot_quat = presampled_robot_poses[robot.model][0]["orientation"]
            robot.set_position_orientation(robot_pos, robot_quat)
            env.scene.write_task_metadata(key=tro_key, data=state)
        else:
            env.task.object_scope[tro_key].load_state(state, serialized=False)

    # Let physics settle
    for _ in range(25):
        og.sim.step_physics()
        for entity in env.task.object_scope.values():
            if not entity.is_system and entity.exists:
                entity.keep_still()

    env.scene.update_initial_file()
    env.scene.reset()


def apply_low_res_rgb(env, image_size: int = DEFAULT_IMAGE_SIZE) -> None:
    """
    Apply low-resolution settings for all robot vision sensors.

    Args:
        env: OmniGibson environment.
        image_size: Target image resolution.
    """
    robot = env.robots[0]
    for sensor in robot.sensors.values():
        if hasattr(sensor, "image_height") and hasattr(sensor, "image_width"):
            sensor.image_height = image_size
            sensor.image_width = image_size
    env.load_observation_space()


def set_viewer_camera_to_robot(env, distance: float = 3.0, height: float = 2.0) -> None:
    """
    Position the viewer camera behind and above the robot, looking at it.

    Args:
        env: OmniGibson environment.
        distance: Distance behind the robot.
        height: Height above the robot.
    """
    if gm.HEADLESS or og.sim.viewer_camera is None:
        return

    robot = env.robots[0]
    robot_pos, robot_quat = robot.get_position_orientation()
    robot_pos = th.as_tensor(robot_pos)
    robot_quat = th.as_tensor(robot_quat)

    # Robot forward direction (+X in local frame)
    robot_rot_mat = T.quat2mat(robot_quat)
    robot_forward = robot_rot_mat[:, 0]

    # Camera position: behind the robot and elevated
    cam_pos = robot_pos.clone()
    cam_pos[0] -= robot_forward[0] * distance
    cam_pos[1] -= robot_forward[1] * distance
    cam_pos[2] += height

    # Look at the robot (slightly above base)
    look_dir = robot_pos - cam_pos
    look_dir[2] += 0.8
    look_dir = look_dir / th.norm(look_dir)

    up = th.tensor([0.0, 0.0, 1.0], dtype=look_dir.dtype, device=look_dir.device)
    forward = look_dir
    right = th.cross(forward, up)
    right_norm = th.norm(right)
    if right_norm < 1e-6:
        up = th.tensor([0.0, 1.0, 0.0], dtype=look_dir.dtype, device=look_dir.device)
        right = th.cross(forward, up)
    right = right / th.norm(right)
    up = th.cross(right, forward)
    up = up / th.norm(up)

    # Camera convention: +X right, +Y up, -Z forward
    rot_mat = th.stack([right, up, -forward], dim=1)
    cam_quat = T.mat2quat(rot_mat)

    og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_quat)


def get_eval_instance_ids(
    task_name: str,
    eval_instance_ids: Optional[List[int]] = None,
) -> List[int]:
    """
    Get evaluation instance IDs from test_instances.csv.

    Args:
        task_name: Name of the task.
        eval_instance_ids: Optional list of specific instance indices.

    Returns:
        List of task instance IDs.
    """
    if eval_instance_ids is None:
        eval_instance_ids = list(range(NUM_EVAL_INSTANCES))
    eval_instance_ids = list(eval_instance_ids)

    assert set(eval_instance_ids).issubset(
        set(range(NUM_EVAL_INSTANCES))
    ), f"eval_instance_ids must be in range({NUM_EVAL_INSTANCES})"

    task_instance_csv_path = os.path.join(
        gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
    )
    with open(task_instance_csv_path, "r") as f:
        lines = list(csv.reader(f))[1:]

    task_idx = TASK_NAMES_TO_INDICES[task_name]
    assert (
        lines[task_idx][1] == task_name
    ), f"Task name from args {task_name} does not match task name from csv {lines[task_idx][1]}"

    test_instances = lines[task_idx][2].strip().split(",")
    return [int(test_instances[i]) for i in eval_instance_ids]


def load_env(task_name: str = "turning_on_radio", headless: bool = False):
    """
    Load the environment and robot configuration.

    Args:
        task_name: Name of the task.
        headless: Whether to run in headless mode.

    Returns:
        OmniGibson environment.
    """
    # Disable transition rules
    for rule in DISABLED_TRANSITION_RULES:
        rule.ENABLED = False

    # Validate task
    available_tasks = load_available_tasks()
    assert task_name in available_tasks, f"Got invalid task name: {task_name}"

    # Get human stats for max_steps
    task_idx = TASK_NAMES_TO_INDICES[task_name]
    human_stats = {"length": []}

    episodes_path = os.path.join(
        gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"
    )
    if os.path.exists(episodes_path):
        with open(episodes_path, "r") as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                human_stats["length"].append(episode["length"])

    avg_length = (
        sum(human_stats["length"]) / len(human_stats["length"])
        if human_stats["length"]
        else 2500
    )

    # Generate configs
    task_cfg = available_tasks[task_name][0]
    cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)

    # Robot config
    robot_cfg_path = os.path.join(Path(__file__).parents[1], "configs", "r1pro_primitives.yaml")
    curobo_cfg = OmegaConf.load(robot_cfg_path)
    cfg["robots"] = OmegaConf.to_container(curobo_cfg.robots, resolve=True)
    # Merge task-specific fields (position, orientation, reset_joint_pos)
    robot_type = cfg["robots"][0].get("type", "R1Pro")
    joint_pos = ROBOT_RESET_JOINT_POS[robot_type].clone()
    joint_pos[-4:] = 0.05  # Fingers MUST start open
    joint_pos[6:10] = infer_torso_qpos_from_trunk_translate(DEFAULT_TRUNK_TRANSLATE)
    cfg["robots"][0]["position"] = task_cfg["robot_start_position"]
    cfg["robots"][0]["orientation"] = task_cfg["robot_start_orientation"]
    cfg["robots"][0]["reset_joint_pos"] = joint_pos
    # Max steps
    cfg["task"]["termination_config"]["max_steps"] = int(avg_length * 2)
    cfg["task"]["include_obs"] = False

    # Headless mode
    gm.HEADLESS = headless

    # Create environment
    print(f"Loading environment for task: {task_name}")
    env = og.Environment(configs=cfg)

    # Apply settings
    apply_low_res_rgb(env=env, image_size=DEFAULT_IMAGE_SIZE)

    return env


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for testing navigate and grasp functionality."""
    try:
        parser = argparse.ArgumentParser(description="Navigate to object and grasp it")
        parser.add_argument("--task_name", type=str, default="turning_on_radio",
                            help="Name of the task to load")
        parser.add_argument("--headless", action="store_true", default=False,
                            help="Run in headless mode (no GUI)")
        parser.add_argument("--eval_instance_ids", type=str, default=None,
                            help="Comma-separated list of instance IDs to evaluate")
        parser.add_argument("--object_name", type=str, default="radio_89",
                            help="Name of object to grasp")
        parser.add_argument("--visualize", action="store_true", default=False,
                            help="Enable visualization of obstacles, poses, and trajectories")
        parser.add_argument("--max_samples", type=int, default=50,
                            help="Maximum sampling attempts for pose near object")

        args = parser.parse_args()

        # Parse instance IDs
        eval_instance_ids = None
        if args.eval_instance_ids:
            eval_instance_ids = [
                int(x) for x in args.eval_instance_ids.split(",") if x.strip() != ""
            ]

        instance_ids = get_eval_instance_ids(
            task_name=args.task_name,
            eval_instance_ids=eval_instance_ids,
        )

        # Load environment
        env = load_env(task_name=args.task_name, headless=args.headless)
        scene = env.scene
        robot = env.robots[0]

        # Load task instance
        if instance_ids:
            instance_id = instance_ids[1]
            load_task_instance(env=env, robot=robot, instance_id=instance_id, test_hidden=False)
        set_viewer_camera_to_robot(env)

        print(f"\nEnvironment loaded successfully.")
        print(f"Robot: {robot.name} ({type(robot).__name__})")
        print(f"Robot pose: {robot.get_position_orientation()}")

        # Let physics settle
        for _ in range(30):
            og.sim.step()

        # Get target object
        grasp_obj = scene.object_registry("name", args.object_name)
        if grasp_obj is None:
            print(f"[ERROR] Object '{args.object_name}' not found in scene")
            return

        print(f"\nTarget object: {grasp_obj.name}")
        print(f"Object position: {grasp_obj.get_position()}")

        # Create controller
        controller = StarterSemanticActionPrimitives(
            env, robot, enable_head_tracking=False, curobo_batch_size=1
        )
        # Execute navigate and grasp
        result = navigate_and_grasp(
            controller=controller,
            env=env,
            obj=grasp_obj,
            max_samples=args.max_samples,
            verbose=True,
            visualize=args.visualize,
        )

        print(f"\n{'='*50}")
        print(f"Final result: {result.value}")
        print(f"{'='*50}")
        print(f"Simulation running. Press Ctrl+C to stop...")
        try:
            while True:
                og.sim.step()
        except KeyboardInterrupt:
            print(f"\nStopped by user.")

    except Exception:
        traceback.print_exc()
    finally:
        try:
            time.sleep(5)
            og.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
