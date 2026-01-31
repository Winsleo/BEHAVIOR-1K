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
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.learning.utils.eval_utils import (
    TASK_NAMES_TO_INDICES,
    generate_basic_environment_config,
)
from omnigibson.macros import gm, macros
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.ui_utils import clear_debug_drawing, draw_box
from omnigibson.utils.visualize_utils import (
    visualize_2d_pose,
    visualize_obstacles,
    visualize_robot_and_sampled_pose,
    visualize_robot_spheres_at_config,
    visualize_sampling_region,
    visualize_trajectory,
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
DEFAULT_MAX_SAMPLES = 100
DEFAULT_IMAGE_SIZE = 224
MAX_VERBOSE_SAMPLES = 20


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
class SampledPoseResult:
    """Result of pose sampling near an object."""
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

    sample_result = sample_pose_near_object(
        controller=controller,
        obj=obj,
        max_samples=max_samples,
        verbose=verbose,
        visualize=visualize,
    )

    if not sample_result.success:
        if verbose:
            print(f"[FAILED] Could not find valid pose near object")
        return GraspResult.SAMPLING_FAILED

    pregrasp_pose = sample_result.pregrasp_pose
    grasp_pose = sample_result.grasp_pose
    base_pose_2d = sample_result.base_pose_2d

    # Visualize sampling region and sampled pose
    if visualize:
        visualize_robot_and_sampled_pose(
            robot=robot,
            sampled_pose=base_pose_2d,
            verbose=verbose,
        )

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
    )

    if q_traj is None:
        if verbose:
            print(f"[FAILED] Navigation planning failed")
        return GraspResult.NAVIGATION_FAILED

    # Visualize planned trajectory
    if visualize and q_traj is not None:
        visualize_trajectory(q_traj=q_traj, robot=robot, verbose=verbose)

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

    for action in controller._execute_grasp():
        env.step(action)

    # Move to final grasp pose
    for action in controller._move_hand(
        grasp_pose,
        motion_constraint=[1, 1, 1, 1, 1, 0],
        stop_on_ag=True,
        ignore_objects=[obj],
    ):
        env.step(action)

    if verbose:
        print(f"[OK] Grasp complete for object: {obj.name}")
        print(f"{'='*50}\n")

    return GraspResult.SUCCESS


def sample_pose_near_object(
    controller: StarterSemanticActionPrimitives,
    obj,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    sampling_radius: float = DEFAULT_SAMPLING_RADIUS,
    verbose: bool = True,
    visualize: bool = False,
    seed: int = 42,
) -> SampledPoseResult:
    """
    Sample a valid 2D base pose near an object for grasping.

    This function finds a collision-free position where:
    - The robot base does not collide with obstacles
    - The target object is reachable by the robot arm
    - The position is in the same room as the object

    Uses batch collision checking for efficiency.

    Args:
        controller: StarterSemanticActionPrimitives instance.
        obj: Target object.
        max_samples: Maximum number of sampling attempts.
        sampling_radius: Maximum distance from object to sample.
        verbose: Whether to print progress information.
        visualize: Whether to visualize failed poses.
        seed: Random seed for reproducibility.

    Returns:
        SampledPoseResult containing pose information and statistics.
    """
    # Set random seed for reproducibility
    th.manual_seed(seed)

    robot = controller.robot
    arm = controller.arm

    # Initialize statistics
    stats = {
        "total_attempts": 0,
        "collision_failures": 0,
        "reachability_failures": 0,
        "room_failures": 0,
    }

    # Sample grasp pose for the object
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
        return SampledPoseResult(
            pregrasp_pose=None,
            grasp_pose=None,
            base_pose_2d=None,
            success=False,
            stats=stats,
        )

    target_pose = eef_pose

    # Get object's room(s)
    obj_rooms = obj.in_rooms if obj.in_rooms else [
        robot.scene._seg_map.get_room_instance_by_point(target_pose[0][:2])
    ]

    if verbose:
        print(f"  Object room(s): {obj_rooms}")

    # Sampling parameters
    distance_lo, distance_hi = 0.0, sampling_radius
    yaw_lo, yaw_hi = -math.pi, math.pi
    avg_arm_workspace_range = th.mean(robot.arm_workspace_range[arm])

    if verbose:
        print(f"  Sampling distance range: [{distance_lo:.2f}, {distance_hi:.2f}]m")
        print(f"  Average arm workspace range: {avg_arm_workspace_range:.3f}")

    # Update obstacles before sampling
    controller._motion_generator.update_obstacles()

    # ========================================================================
    # Batch generation of all candidate 2D poses
    # ========================================================================
    if verbose:
        print(f"  Generating {max_samples} candidate poses...")

    # Generate all random samples at once
    distances = th.rand(max_samples) * (distance_hi - distance_lo) + distance_lo
    yaws = th.rand(max_samples) * (yaw_hi - yaw_lo) + yaw_lo

    # Compute all candidate 2D poses: (max_samples, 3) -> [x, y, yaw]
    candidate_2d_poses = th.stack([
        target_pose[0][0] + distances * th.cos(yaws),
        target_pose[0][1] + distances * th.sin(yaws),
        yaws + math.pi - avg_arm_workspace_range,
    ], dim=1)  # Shape: (max_samples, 3)

    # ========================================================================
    # Room filter (must be done per-sample due to map query)
    # ========================================================================
    room_valid_mask = th.zeros(max_samples, dtype=th.bool)
    for i in range(max_samples):
        candidate_room = robot.scene._seg_map.get_room_instance_by_point(candidate_2d_poses[i, :2])
        room_valid_mask[i] = candidate_room in obj_rooms

    room_valid_indices = th.where(room_valid_mask)[0]
    stats["room_failures"] = max_samples - len(room_valid_indices)

    if verbose:
        print(f"  Room filter: {len(room_valid_indices)}/{max_samples} passed")

    if len(room_valid_indices) == 0:
        if verbose:
            print(f"\n  [FAILED] All samples failed room check")
        return SampledPoseResult(
            pregrasp_pose=None,
            grasp_pose=None,
            base_pose_2d=None,
            success=False,
            stats=stats,
        )

    # ========================================================================
    # Batch collision check
    # ========================================================================
    current_joint_pos = robot.get_joint_positions()
    room_valid_poses = candidate_2d_poses[room_valid_indices]  # (N_valid, 3)

    # Build batch joint positions
    batch_joint_positions = current_joint_pos.unsqueeze(0).repeat(len(room_valid_indices), 1)
    batch_joint_positions[:, robot.base_control_idx] = room_valid_poses

    obj_in_hand = controller._get_obj_in_hand()
    attached_obj = (
        {robot.eef_link_names[arm]: obj_in_hand.root_link}
        if obj_in_hand else None
    )

    if verbose:
        print(f"  Running batch collision check for {len(room_valid_indices)} candidates...")

    collision_results = controller._motion_generator.check_collisions(
        batch_joint_positions,
        self_collision_check=False,
        skip_obstacle_update=True,
        attached_obj=attached_obj,
    ).cpu()  # Shape: (N_valid,)

    collision_free_mask = ~collision_results
    collision_free_local_indices = th.where(collision_free_mask)[0]
    stats["collision_failures"] = int(collision_results.sum().item())

    if verbose:
        print(f"  Collision filter: {len(collision_free_local_indices)}/{len(room_valid_indices)} passed")

    # Collect collision failed poses and joint positions for visualization
    collision_failed_poses = []
    collision_failed_joint_positions = []
    for i in range(len(room_valid_poses)):
        if collision_results[i].item():
            collision_failed_poses.append(room_valid_poses[i].clone())
            collision_failed_joint_positions.append(batch_joint_positions[i].clone())

    if len(collision_free_local_indices) == 0:
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
        return SampledPoseResult(
            pregrasp_pose=None,
            grasp_pose=None,
            base_pose_2d=None,
            success=False,
            stats=stats,
        )

    # ========================================================================
    # Reachability check (sequential, as IK solving is typically not batched)
    # ========================================================================
    reachability_failed_poses = []

    if verbose:
        print(f"  Checking reachability for {len(collision_free_local_indices)} collision-free candidates...")

    for local_idx in collision_free_local_indices:
        candidate_2d_pose = room_valid_poses[local_idx]
        joint_pos = batch_joint_positions[local_idx]

        if controller._target_in_reach_of_robot(
            target_pose, initial_joint_pos=joint_pos, skip_obstacle_update=True
        ):
            # Success! Found a valid pose
            global_idx = room_valid_indices[local_idx].item()
            stats["total_attempts"] = global_idx + 1

            if verbose:
                print(f"\n  [SUCCESS] Found valid pose (sample {global_idx})")
                print(f"    Collision failures: {stats['collision_failures']}")
                print(f"    Reachability failures: {stats['reachability_failures']}")
                print(f"    Room failures: {stats['room_failures']}")
                print(f"    Valid pose: ({candidate_2d_pose[0].item():.3f}, {candidate_2d_pose[1].item():.3f}, {candidate_2d_pose[2].item():.3f})")

            return SampledPoseResult(
                pregrasp_pose=eef_pose,
                grasp_pose=grasp_pose,
                base_pose_2d=candidate_2d_pose,
                success=True,
                stats=stats,
            )
        else:
            stats["reachability_failures"] += 1
            reachability_failed_poses.append(candidate_2d_pose.clone())
            if verbose and len(reachability_failed_poses) <= MAX_VERBOSE_SAMPLES:
                print(f"    Sample {room_valid_indices[local_idx].item()}: Not reachable at ({candidate_2d_pose[0].item():.2f}, {candidate_2d_pose[1].item():.2f})")

    # Sampling failed
    stats["total_attempts"] = max_samples
    if verbose:
        print(f"\n  [FAILED] No valid pose found after {max_samples} attempts")
        print(f"    Collision failures: {stats['collision_failures']}")
        print(f"    Reachability failures: {stats['reachability_failures']}")
        print(f"    Room failures: {stats['room_failures']}")

    # Visualize failed poses if requested
    if visualize:
        _visualize_failed_poses(
            robot, collision_failed_poses, reachability_failed_poses,
            max_display=MAX_VERBOSE_SAMPLES,
            motion_generator=controller._motion_generator,
            collision_joint_positions=collision_failed_joint_positions,
        )

    return SampledPoseResult(
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
) -> Optional[th.Tensor]:
    """
    Plan navigation trajectory to a target 2D pose.

    Args:
        controller: StarterSemanticActionPrimitives instance.
        robot: Robot object.
        target_pose_2d: Target 2D pose (x, y, yaw).
        verbose: Whether to print progress information.

    Returns:
        Joint trajectory tensor if successful, None otherwise.
    """
    # Convert 2D pose to 3D pose
    pose_3d = controller._get_robot_pose_from_2d_pose(target_pose_2d)
    target_pos = {robot.base_footprint_link_name: pose_3d[0]}
    target_quat = {robot.base_footprint_link_name: pose_3d[1]}

    if verbose:
        print(f"  Target position: {pose_3d[0].tolist()}")
        print(f"  Target orientation: {pose_3d[1].tolist()}")

    # Plan motion
    q_traj = controller._plan_joint_motion(
        target_pos=target_pos,
        target_quat=target_quat,
        embodiment_selection=CuRoboEmbodimentSelection.BASE,
        skip_obstacle_update=True,
    )

    if q_traj is not None and verbose:
        print(f"  [OK] Planned trajectory with {len(q_traj)} waypoints")

    return q_traj


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
    Visualize failed candidate poses for debugging.

    Uses visualize_2d_pose to show position and orientation (with arrows).
    Red = collision failures, Green = reachability failures.
    Optionally shows collision spheres for the first collision failure.

    Args:
        robot: Robot object for visualizing current robot pose.
        collision_poses: List of 2D poses (x, y, yaw) that failed collision check.
        reachability_poses: List of 2D poses (x, y, yaw) that failed reachability check.
        z_height: Z-coordinate for visualization.
        max_display: Maximum number of poses to display per category.
        motion_generator: Optional CuRoboMotionGenerator for sphere visualization.
        collision_joint_positions: Optional list of joint positions for collision failures.
    """
    # Visualize robot's current pose (blue)
    robot_pos, robot_quat = robot.get_position_orientation()
    robot_yaw = T.quat2euler(robot_quat)[2].item()
    robot_2d_pose = th.tensor([robot_pos[0], robot_pos[1], robot_yaw])
    visualize_2d_pose(robot_2d_pose, z_height=z_height, color=(0.0, 0.5, 1.0, 1.0), verbose=False)

    # Red: collision failures (with arrows showing orientation)
    for i, pose in enumerate(collision_poses[:max_display]):
        visualize_2d_pose(
            pose,
            z_height=z_height + 0.01 * i,  # Slight offset to avoid overlap
            arrow_length=0.3,
            color=(1.0, 0.0, 0.0, 0.8),  # Red
            verbose=False,
        )

    # Green: reachability failures (with arrows showing orientation)
    for i, pose in enumerate(reachability_poses[:max_display]):
        visualize_2d_pose(
            pose,
            z_height=z_height + 0.01 * (i + max_display),  # Offset after collision poses
            arrow_length=0.3,
            color=(0.0, 1.0, 0.0, 0.8),  # Green
            verbose=False,
        )

    # Visualize collision spheres for first collision failure (shows WHY it's a collision)
    if motion_generator is not None and collision_joint_positions and len(collision_joint_positions) > 0:
        print(f"\n  Visualizing collision spheres for first collision failure...")
        print(f"  (Red boxes = robot collision spheres at sampled position)")
        visualize_robot_spheres_at_config(
            motion_generator=motion_generator,
            joint_positions=collision_joint_positions[0],
            color=(1.0, 0.3, 0.3, 0.5),  # Semi-transparent red
            verbose=True,
        )

    print(f"\n  Visualization legend:")
    print(f"    Blue arrow = Robot current pose")
    print(f"    Red arrows ({len(collision_poses[:max_display])}) = Collision failures")
    print(f"    Green arrows ({len(reachability_poses[:max_display])}) = Reachability failures")
    if motion_generator is not None and collision_joint_positions:
        print(f"    Red boxes = Robot collision spheres at first collision failure")

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()
    # Keep simulation running for visualization inspection
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
            robot_pos = presampled_robot_poses[robot.model_name][0]["position"]
            robot_quat = presampled_robot_poses[robot.model_name][0]["orientation"]
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
    robot_cfg_path = os.path.join(og.example_config_path, "r1_primitives.yaml")
    with open(robot_cfg_path, "r") as f:
        robot_cfg = yaml.safe_load(f)
    robot_config = robot_cfg["robots"][0]

    if task_cfg is not None:
        robot_config["position"] = task_cfg["robot_start_position"]
        robot_config["orientation"] = task_cfg["robot_start_orientation"]

    cfg["robots"] = [robot_config]

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
