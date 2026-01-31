"""
Visualization utilities for robot poses, trajectories, and collision spheres.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.utils.ui_utils import draw_box, draw_line

# ============================================================================
# Constants
# ============================================================================

DEFAULT_Z_HEIGHT = 0.1
DEFAULT_ARROW_LENGTH = 0.5
DEFAULT_LINE_SIZE = 3.0
DEFAULT_BOX_SIZE = 2.0
ARROW_HEAD_RATIO = 0.3
ARROW_HEAD_ANGLE = math.pi / 6  # 30 degrees
WAYPOINT_STEP_DIVISOR = 20

# Color presets (RGBA)
COLOR_BLUE = (0.0, 0.5, 1.0, 1.0)
COLOR_GREEN = (0.0, 1.0, 0.0, 1.0)
COLOR_RED = (1.0, 0.0, 0.0, 1.0)
COLOR_PURPLE = (1.0, 0.0, 1.0, 1.0)
COLOR_SPHERE = (0.0, 0.5, 1.0, 0.6)

# Type aliases
ColorType = Tuple[float, float, float, float]
PoseType = Union[th.Tensor, np.ndarray, List[float]]


# ============================================================================
# Helper Functions
# ============================================================================

def _to_scalar(val: Union[th.Tensor, np.ndarray, float]) -> float:
    """
    Convert a tensor, numpy array, or scalar to a Python float.

    Args:
        val: Input value (tensor, numpy array, or scalar).

    Returns:
        Python float value.
    """
    if isinstance(val, th.Tensor):
        return val.item()
    elif isinstance(val, np.ndarray):
        return float(val)
    return float(val)


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_robot_spheres(
    motion_generator: CuRoboMotionGenerator,
    robot: BaseRobot,
    emb_sel: CuRoboEmbodimentSelection = CuRoboEmbodimentSelection.DEFAULT,
    verbose: bool = True,
) -> None:
    """
    Visualize the robot's sphere representation used by CuRobo for collision detection.

    Args:
        motion_generator: CuRoboMotionGenerator instance.
        robot: Robot object.
        emb_sel: Embodiment selection for CuRobo.
        verbose: Whether to print debug information.
    """
    import omnigibson.lazy as lazy

    # Get current joint state
    q = robot.get_joint_positions()
    cu_js = lazy.curobo.types.state.JointState(
        position=motion_generator.tensor_args.to_device(q.unsqueeze(0)),
        joint_names=motion_generator.robot_joint_names,
    ).get_ordered_joint_state(motion_generator.mg[emb_sel].kinematics.joint_names)

    # Get robot's sphere representation
    spheres = motion_generator.mg[emb_sel].kinematics.get_robot_as_spheres(cu_js.position)

    if verbose:
        print("=== Robot Sphere Representation ===")
        print(f"Number of spheres: {len(spheres[0]) if spheres else 0}")

    # Visualize spheres using blue boxes
    if spheres and len(spheres[0]) > 0:
        for sphere in spheres[0]:
            center = sphere.pose[:3].cpu().numpy()
            radius = _to_scalar(sphere.radius)

            # Approximate sphere with a box
            draw_box(
                center=center.tolist(),
                extents=[radius * 2, radius * 2, radius * 2],
                color=COLOR_SPHERE,
                size=1.0,
            )


def visualize_robot_spheres_at_config(
    motion_generator: CuRoboMotionGenerator,
    joint_positions: th.Tensor,
    color: ColorType = COLOR_RED,
    emb_sel: CuRoboEmbodimentSelection = CuRoboEmbodimentSelection.DEFAULT,
    verbose: bool = True,
) -> None:
    """
    Visualize the robot's collision spheres at a specific joint configuration.

    This is useful for debugging why a sampled pose is considered a collision.
    The spheres show the robot's full body (base, trunk, arms) at the given config.

    Args:
        motion_generator: CuRoboMotionGenerator instance.
        joint_positions: Joint positions tensor (1D or 2D with batch dim).
        color: RGBA color for the spheres.
        emb_sel: Embodiment selection for CuRobo.
        verbose: Whether to print debug information.
    """
    import omnigibson.lazy as lazy

    # Ensure 2D tensor
    if joint_positions.dim() == 1:
        joint_positions = joint_positions.unsqueeze(0)

    # Create joint state
    cu_js = lazy.curobo.types.state.JointState(
        position=motion_generator.tensor_args.to_device(joint_positions),
        joint_names=motion_generator.robot_joint_names,
    ).get_ordered_joint_state(motion_generator.mg[emb_sel].kinematics.joint_names)

    # Get robot's sphere representation at this configuration
    spheres = motion_generator.mg[emb_sel].kinematics.get_robot_as_spheres(cu_js.position)

    if verbose:
        print(f"=== Robot Spheres at Config ===")
        print(f"Number of spheres: {len(spheres[0]) if spheres else 0}")

    # Visualize spheres
    if spheres and len(spheres[0]) > 0:
        for sphere in spheres[0]:
            center = sphere.pose[:3].cpu().numpy()
            radius = _to_scalar(sphere.radius)

            draw_box(
                center=center.tolist(),
                extents=[radius * 2, radius * 2, radius * 2],
                color=color,
                size=1.0,
            )

        if verbose:
            # Print bounding box info
            all_centers = th.stack([s.pose[:3] for s in spheres[0]])
            min_bound = all_centers.min(dim=0).values.cpu()
            max_bound = all_centers.max(dim=0).values.cpu()
            print(f"Bounding box: ({min_bound[0]:.2f}, {min_bound[1]:.2f}, {min_bound[2]:.2f}) to ({max_bound[0]:.2f}, {max_bound[1]:.2f}, {max_bound[2]:.2f})")


def visualize_obstacles(
    motion_generator: CuRoboMotionGenerator,
    save_path: Optional[str] = None,
    emb_sel: CuRoboEmbodimentSelection = CuRoboEmbodimentSelection.DEFAULT,
    verbose: bool = True,
) -> None:
    """
    Visualize obstacles detected by CuRobo.

    Args:
        motion_generator: CuRoboMotionGenerator instance.
        save_path: Optional path to save obstacle mesh (.obj file).
        emb_sel: Embodiment selection for CuRobo.
        verbose: Whether to print obstacle information.
    """
    # Update obstacles
    motion_generator.update_obstacles()

    # Get world model
    world_model = motion_generator.mg[emb_sel].world_model

    # Get all obstacle bounding boxes and visualize
    mesh_world = world_model.get_mesh_world(merge_meshes=False)

    if verbose:
        print("=== CuRobo Obstacle Information ===")
        print(f"Number of meshes: {len(mesh_world.mesh) if mesh_world.mesh else 0}")

    if mesh_world.mesh:
        for i, mesh in enumerate(mesh_world.mesh):
            # Get mesh bounding box
            vertices = th.tensor(mesh.vertices) if hasattr(mesh, 'vertices') else None
            if vertices is not None and len(vertices) > 0:
                min_bound = vertices.min(dim=0).values
                max_bound = vertices.max(dim=0).values
                center = (min_bound + max_bound) / 2
                extents = max_bound - min_bound

                # Draw bounding box (yellow for obstacles)
                draw_box(
                    center=center.tolist(),
                    extents=extents.tolist(),
                    color=(1.0, 0.8, 0.0, 0.5),
                    size=1.0,
                )

    # Save to file
    if save_path:
        robot = motion_generator.robot
        q = robot.get_joint_positions().unsqueeze(0)
        motion_generator.save_visualization(q, save_path)
        if verbose:
            print(f"Obstacle mesh saved to: {save_path}")


def visualize_sampling_region(
    obj,
    eef_pose: Tuple[th.Tensor, th.Tensor],
    z_height: float = DEFAULT_Z_HEIGHT,
    radius: float = 2.5,
    verbose: bool = True,
) -> None:
    """
    Visualize the sampling region: a circular area centered on the grasp pose.

    Args:
        obj: Target object.
        eef_pose: End-effector target pose (position, orientation).
        z_height: Z-coordinate height for visualization.
        radius: Sampling radius (default 2.5m).
        verbose: Whether to print region information.
    """
    center = eef_pose[0]

    # Draw sampling region boundary (circle approximated with line segments)
    num_points = 24
    for i in range(num_points):
        angle1 = 2 * math.pi * i / num_points
        angle2 = 2 * math.pi * (i + 1) / num_points

        p1 = (
            _to_scalar(center[0]) + radius * math.cos(angle1),
            _to_scalar(center[1]) + radius * math.sin(angle1),
            z_height,
        )
        p2 = (
            _to_scalar(center[0]) + radius * math.cos(angle2),
            _to_scalar(center[1]) + radius * math.sin(angle2),
            z_height,
        )

        draw_line(p1, p2, color=(1.0, 0.5, 0.0, 0.8), size=DEFAULT_BOX_SIZE)

    # Draw center point (grasp target)
    draw_box(
        center=[_to_scalar(center[0]), _to_scalar(center[1]), z_height],
        extents=[0.1, 0.1, 0.02],
        color=COLOR_RED,
        size=DEFAULT_BOX_SIZE,
    )

    # Draw object position
    obj_pos = obj.get_position_orientation()[0]
    draw_box(
        center=[_to_scalar(obj_pos[0]), _to_scalar(obj_pos[1]), z_height + 0.02],
        extents=[0.15, 0.15, 0.02],
        color=COLOR_GREEN,
        size=DEFAULT_BOX_SIZE,
    )

    if verbose:
        print("=== Sampling Region ===")
        print(f"  Center (grasp target): ({_to_scalar(center[0]):.3f}, {_to_scalar(center[1]):.3f})")
        print(f"  Object position: ({_to_scalar(obj_pos[0]):.3f}, {_to_scalar(obj_pos[1]):.3f})")
        print(f"  Sampling radius: {radius}m")


def visualize_2d_pose(
    pose_2d: PoseType,
    z_height: float = DEFAULT_Z_HEIGHT,
    arrow_length: float = DEFAULT_ARROW_LENGTH,
    color: ColorType = COLOR_RED,
    verbose: bool = True,
) -> None:
    """
    Visualize a 2D pose returned by _sample_pose_near_object.

    Draws an arrow indicating position and orientation, with a box at the base position.

    Args:
        pose_2d: (x, y, yaw) tensor, numpy array, or list.
        z_height: Z-coordinate height for visualization.
        arrow_length: Length of the direction arrow.
        color: RGBA color tuple.
        verbose: Whether to print pose information.
    """
    x = _to_scalar(pose_2d[0])
    y = _to_scalar(pose_2d[1])
    yaw = _to_scalar(pose_2d[2])

    # Start position
    start_pos = (x, y, z_height)

    # Calculate arrow endpoint based on yaw (orientation)
    end_x = x + arrow_length * math.cos(yaw)
    end_y = y + arrow_length * math.sin(yaw)
    end_pos = (end_x, end_y, z_height)

    # Draw main direction arrow
    draw_line(start_pos, end_pos, color=color, size=DEFAULT_LINE_SIZE)

    # Draw arrow head
    arrow_head_length = arrow_length * ARROW_HEAD_RATIO
    left_head_x = end_x - arrow_head_length * math.cos(yaw - ARROW_HEAD_ANGLE)
    left_head_y = end_y - arrow_head_length * math.sin(yaw - ARROW_HEAD_ANGLE)
    right_head_x = end_x - arrow_head_length * math.cos(yaw + ARROW_HEAD_ANGLE)
    right_head_y = end_y - arrow_head_length * math.sin(yaw + ARROW_HEAD_ANGLE)

    draw_line(end_pos, (left_head_x, left_head_y, z_height), color=color, size=DEFAULT_LINE_SIZE)
    draw_line(end_pos, (right_head_x, right_head_y, z_height), color=color, size=DEFAULT_LINE_SIZE)

    # Draw position marker box
    draw_box(center=(x, y, z_height), extents=(0.15, 0.15, 0.02), color=color, size=DEFAULT_BOX_SIZE)

    if verbose:
        print(f"Visualized pose: x={x:.3f}, y={y:.3f}, yaw={math.degrees(yaw):.1f} deg")


def visualize_robot_and_sampled_pose(
    robot: BaseRobot,
    sampled_pose: PoseType,
    z_height: float = DEFAULT_Z_HEIGHT,
    verbose: bool = True,
) -> None:
    """
    Visualize both the robot's current position and a sampled target navigation pose.

    The robot's current pose is shown in blue, the target pose in green,
    and a red line connects them.

    Args:
        robot: Robot object.
        sampled_pose: (x, y, yaw) tensor from _sample_pose_near_object.
        z_height: Z-coordinate height for visualization.
        verbose: Whether to print pose and distance information.
    """
    # Get robot's current position and orientation
    robot_pos, robot_quat = robot.get_position_orientation()
    robot_yaw = T.quat2euler(robot_quat)[2].item()
    robot_2d_pose = th.tensor([robot_pos[0], robot_pos[1], robot_yaw])

    if verbose:
        print(f"Robot 3D position: {robot_pos}, 3D orientation: {robot_quat}")

    # Draw robot's current pose (blue)
    visualize_2d_pose(robot_2d_pose, z_height=z_height, color=COLOR_BLUE, verbose=False)

    if verbose:
        print(f"Robot current 2D pose: x={robot_pos[0]:.3f}, y={robot_pos[1]:.3f}, yaw={math.degrees(robot_yaw):.1f} deg")

    # Draw sampled target pose (green, slightly elevated to avoid overlap)
    visualize_2d_pose(sampled_pose, z_height=z_height + 0.01, color=COLOR_GREEN, verbose=False)

    if verbose:
        target_x = _to_scalar(sampled_pose[0])
        target_y = _to_scalar(sampled_pose[1])
        target_yaw = _to_scalar(sampled_pose[2])
        print(f"Target 2D pose: x={target_x:.3f}, y={target_y:.3f}, yaw={math.degrees(target_yaw):.1f} deg")

    # Draw connecting line from robot to target (red)
    robot_xy = (robot_pos[0].item(), robot_pos[1].item(), z_height)
    target_xy = (_to_scalar(sampled_pose[0]), _to_scalar(sampled_pose[1]), z_height)
    draw_line(robot_xy, target_xy, color=(1.0, 0.0, 0.0, 0.8), size=DEFAULT_BOX_SIZE)

    # Calculate and display distance
    if verbose:
        distance = math.sqrt(
            (_to_scalar(sampled_pose[0]) - robot_pos[0].item()) ** 2
            + (_to_scalar(sampled_pose[1]) - robot_pos[1].item()) ** 2
        )
        print(f"Distance from robot to target: {distance:.3f}m")


def visualize_trajectory(
    q_traj: Optional[th.Tensor],
    robot: BaseRobot,
    color: ColorType = COLOR_GREEN,
    show_waypoints: bool = True,
    z_height: float = DEFAULT_Z_HEIGHT,
    verbose: bool = True,
) -> None:
    """
    Visualize a planned joint trajectory.

    Draws the base trajectory path with optional waypoint markers.
    Start point is shown in green, end point in red.

    Args:
        q_traj: Joint trajectory tensor of shape (N, num_joints).
        robot: Robot object.
        color: RGBA color for trajectory lines.
        show_waypoints: Whether to display waypoint markers.
        z_height: Z-coordinate height for visualization.
        verbose: Whether to print trajectory information.
    """
    if q_traj is None or len(q_traj) == 0:
        if verbose:
            print("No trajectory to visualize")
        return

    if verbose:
        print("=== Trajectory Visualization ===")
        print(f"Number of waypoints: {len(q_traj)}")

    # Get base control indices
    base_idx = robot.base_control_idx

    # Extract base positions (x, y) from trajectory
    # base_idx typically contains indices for [x, y, z, roll, pitch, yaw]
    positions = [
        (q[base_idx[0]].item(), q[base_idx[1]].item(), z_height)
        for q in q_traj
    ]

    # Draw trajectory lines
    for i in range(len(positions) - 1):
        draw_line(positions[i], positions[i + 1], color=color, size=DEFAULT_LINE_SIZE)

    # Draw waypoint markers
    if show_waypoints:
        # Only show subset of points to avoid clutter
        step = max(1, len(positions) // WAYPOINT_STEP_DIVISOR)
        for i in range(0, len(positions), step):
            draw_box(
                center=positions[i],
                extents=(0.05, 0.05, 0.05),
                color=COLOR_PURPLE,
                size=DEFAULT_BOX_SIZE,
            )

    # Highlight start and end points
    if len(positions) >= 2:
        # Start point (green)
        draw_box(center=positions[0], extents=(0.1, 0.1, 0.1), color=COLOR_GREEN, size=DEFAULT_LINE_SIZE)
        # End point (red)
        draw_box(center=positions[-1], extents=(0.1, 0.1, 0.1), color=COLOR_RED, size=DEFAULT_LINE_SIZE)

    if verbose:
        print(f"Start: {positions[0]}")
        print(f"End: {positions[-1]}")
