import math
import torch as th

import omnigibson.utils.transform_utils as T

from typing import Dict, Generator, List, Optional, Tuple

from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
from omnigibson.learning.task_primitives.models import (
    DEFAULT_MIN_RADIUS,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_SAMPLING_RADIUS,
    GraspBasePoseResult,
    GraspDebugSnapshot,
    MAX_VERBOSE_SAMPLES,
    OPTIMAL_DIST_IDEAL,
    OPTIMAL_DIST_MAX,
    OPTIMAL_DIST_MIN,
    PoseVisualizationData,
    SamplingStrategy,
    TOP_K_CANDIDATES,
)
from omnigibson.learning.task_primitives.observers import build_grasp_debug_snapshot
from omnigibson.utils.geometry_utils import wrap_angle


def report_grasp_debug_context(context, obj, max_samples: int) -> GraspDebugSnapshot:
    snapshot = build_grasp_debug_snapshot(context, obj, max_samples)
    context.observer.report_grasp_debug_snapshot(snapshot)
    return snapshot


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def iterate_navigation_trajectory(context, q_traj) -> Generator[th.Tensor, None, None]:
    yield from context.controller._execute_motion_plan(q_traj, low_precision=True)


def iterate_restore_trunk_actions(context) -> Generator[th.Tensor, None, None]:
    robot = context.robot
    if not robot.is_articulated_trunk:
        return
    initial_trunk = robot.reset_joint_pos[robot.trunk_control_idx]
    start_q = robot.get_joint_positions().clone()
    end_q = start_q.clone()
    end_q[robot.trunk_control_idx] = th.as_tensor(initial_trunk, dtype=end_q.dtype, device=end_q.device)
    q_traj = th.stack([start_q, end_q])
    q_traj = context.controller._motion_generator.add_linearly_interpolated_waypoints(traj=q_traj, max_inter_dist=0.02)
    yield from context.controller._execute_motion_plan(
        q_traj,
        stop_on_contact=True,
        ignore_failure=True,
        low_precision=True,
    )


def generate_candidate_poses(
    target_pos: th.Tensor,
    num_samples: int,
    sampling_radius: float,
    arm_workspace_offset: float = 0.0,
    min_radius: float = 0.0,
    strategy: SamplingStrategy = SamplingStrategy.FIBONACCI_SPIRAL,
) -> th.Tensor:
    if strategy == SamplingStrategy.RANDOM:
        distances = th.rand(num_samples) * (sampling_radius - min_radius) + min_radius
        angles = th.rand(num_samples) * 2 * math.pi - math.pi
    elif strategy == SamplingStrategy.UNIFORM_POLAR:
        t = th.rand(num_samples)
        r_squared_min = min_radius ** 2
        r_squared_max = sampling_radius ** 2
        distances = th.sqrt(t * (r_squared_max - r_squared_min) + r_squared_min)
        angles = th.rand(num_samples) * 2 * math.pi - math.pi
    elif strategy == SamplingStrategy.FIBONACCI_SPIRAL:
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        indices = th.arange(num_samples, dtype=th.float32)
        t = (indices + 0.5) / num_samples
        r_squared_min = min_radius ** 2
        r_squared_max = sampling_radius ** 2
        distances = th.sqrt(t * (r_squared_max - r_squared_min) + r_squared_min)
        angles = indices * golden_angle
        angles = th.remainder(angles + math.pi, 2 * math.pi) - math.pi
    elif strategy == SamplingStrategy.CONCENTRIC_RINGS:
        num_rings = max(1, int(math.sqrt(num_samples)))
        samples_per_ring = num_samples // num_rings
        extra_samples = num_samples % num_rings
        distances_list = []
        angles_list = []
        for ring_idx in range(num_rings):
            t = (ring_idx + 0.5) / num_rings
            r_squared_min = min_radius ** 2
            r_squared_max = sampling_radius ** 2
            ring_radius = math.sqrt(t * (r_squared_max - r_squared_min) + r_squared_min)
            n_in_ring = samples_per_ring + (1 if ring_idx < extra_samples else 0)
            angle_offset = th.rand(1).item() * 2 * math.pi
            ring_angles = th.linspace(0, 2 * math.pi, n_in_ring + 1)[:-1] + angle_offset
            ring_angles = th.remainder(ring_angles + math.pi, 2 * math.pi) - math.pi
            distances_list.append(th.full((n_in_ring,), ring_radius))
            angles_list.append(ring_angles)
        distances = th.cat(distances_list)
        angles = th.cat(angles_list)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    x = target_pos[0] + distances * th.cos(angles)
    y = target_pos[1] + distances * th.sin(angles)
    yaws = angles + math.pi - arm_workspace_offset
    return th.stack([x, y, yaws], dim=1)


def filter_poses_by_room(poses: th.Tensor, valid_rooms: List, scene_seg_map) -> Tuple[th.Tensor, th.Tensor]:
    num_poses = len(poses)
    valid_mask = th.zeros(num_poses, dtype=th.bool)
    for i in range(num_poses):
        room = scene_seg_map.get_room_instance_by_point(poses[i, :2])
        valid_mask[i] = room in valid_rooms
    valid_indices = th.where(valid_mask)[0]
    return poses[valid_indices], valid_indices


def convert_poses_to_joint_positions(poses: th.Tensor, robot, current_joint_pos: th.Tensor) -> th.Tensor:
    num_poses = len(poses)
    batch_joint_positions = current_joint_pos.unsqueeze(0).repeat(num_poses, 1)
    if robot.is_holonomic_base:
        root_pos, root_quat = robot.root_link.get_position_orientation()
        for i, pose_2d in enumerate(poses):
            target_pos = th.tensor([pose_2d[0], pose_2d[1], root_pos[2]])
            target_quat = T.euler2quat(th.tensor([0.0, 0.0, pose_2d[2]]))
            inv_root_pos, inv_root_quat = T.invert_pose_transform(root_pos, root_quat)
            relative_pos, relative_quat = T.pose_transform(inv_root_pos, inv_root_quat, target_pos, target_quat)
            relative_euler = T.quat2euler(relative_quat)
            batch_joint_positions[i, robot.base_control_idx[0]] = relative_pos[0]
            batch_joint_positions[i, robot.base_control_idx[1]] = relative_pos[1]
            batch_joint_positions[i, robot.base_control_idx[2]] = relative_euler[2]
    else:
        batch_joint_positions[:, robot.base_control_idx] = poses
    return batch_joint_positions


def batch_collision_check(controller, batch_joint_positions: th.Tensor, attached_obj: Optional[Dict] = None) -> th.Tensor:
    return controller._motion_generator.check_collisions(
        batch_joint_positions,
        self_collision_check=False,
        skip_obstacle_update=True,
        attached_obj=attached_obj,
    ).cpu()


def check_reachability(
    controller,
    poses: th.Tensor,
    batch_joint_positions: th.Tensor,
    collision_free_indices: th.Tensor,
    target_pose: Tuple[th.Tensor, th.Tensor],
    original_indices: th.Tensor,
    verbose: bool = False,
) -> Tuple[List[Tuple[th.Tensor, int]], List[th.Tensor], int]:
    robot_pos, _ = controller.robot.get_position_orientation()
    robot_xy = robot_pos[:2]
    valid_poses = []
    failed_poses = []
    failure_count = 0
    for local_idx in collision_free_indices:
        candidate_pose = poses[local_idx]
        joint_pos = batch_joint_positions[local_idx]
        if controller._target_in_reach_of_robot(target_pose, initial_joint_pos=joint_pos, skip_obstacle_update=True):
            global_idx = original_indices[local_idx].item()
            valid_poses.append((candidate_pose.clone(), global_idx))
            if verbose:
                dist = th.norm(candidate_pose[:2] - robot_xy).item()
                print(
                    f"    Sample {global_idx}: Valid at ({candidate_pose[0].item():.2f}, "
                    f"{candidate_pose[1].item():.2f}), dist={dist:.2f}m"
                )
        else:
            failure_count += 1
            failed_poses.append(candidate_pose.clone())
            if verbose and len(failed_poses) <= MAX_VERBOSE_SAMPLES:
                print(
                    f"    Sample {original_indices[local_idx].item()}: Not reachable at "
                    f"({candidate_pose[0].item():.2f}, {candidate_pose[1].item():.2f})"
                )
    return valid_poses, failed_poses, failure_count


def compute_heuristic_score(pose_2d: th.Tensor, target_xy: th.Tensor, robot_xy: th.Tensor, robot_yaw: float) -> float:
    pose_xy = pose_2d[:2]
    pose_yaw = pose_2d[2].item()
    dist_to_target = th.norm(pose_xy - target_xy).item()
    if dist_to_target < OPTIMAL_DIST_MIN:
        dist_cost = (OPTIMAL_DIST_MIN - dist_to_target) * 6.0
    elif dist_to_target > OPTIMAL_DIST_MAX:
        dist_cost = (dist_to_target - OPTIMAL_DIST_MAX) * 6.0
    else:
        dist_cost = abs(dist_to_target - OPTIMAL_DIST_IDEAL) * 1.0

    target_direction = th.atan2(target_xy[1] - pose_xy[1], target_xy[0] - pose_xy[0]).item()
    facing_error = abs(wrap_angle(pose_yaw - target_direction))
    facing_cost = facing_error * 0.3
    travel_dist = th.norm(pose_xy - robot_xy).item()
    travel_cost = travel_dist * 1.0
    yaw_change = abs(wrap_angle(pose_yaw - robot_yaw))
    rotation_cost = yaw_change * 0.3
    return dist_cost + facing_cost + travel_cost + rotation_cost


def compute_trajectory_complexity(q_traj: th.Tensor, robot) -> Tuple[float, float]:
    if q_traj is None or len(q_traj) < 2:
        return float("inf"), float("inf")
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


def compute_trajectory_score(path_length: float, cumulative_rotation: float, dist_to_target: float) -> float:
    score = path_length + cumulative_rotation * 0.5
    if dist_to_target < OPTIMAL_DIST_MIN:
        score += (OPTIMAL_DIST_MIN - dist_to_target) * 3.0
    elif dist_to_target > OPTIMAL_DIST_MAX:
        score += (dist_to_target - OPTIMAL_DIST_MAX) * 3.0
    return score


def plan_navigation(controller, robot, target_pose_2d: th.Tensor, observer=None, verbose: bool = True, visualize: bool = False):
    pose_3d = controller._get_robot_pose_from_2d_pose(target_pose_2d)
    base_link_pos = robot.links[robot.base_footprint_link_name].get_position_orientation()[0]
    pose_3d = (pose_3d[0].clone(), pose_3d[1])
    pose_3d[0][2] = base_link_pos[2]
    target_pos = {robot.base_footprint_link_name: pose_3d[0]}
    target_quat = {robot.base_footprint_link_name: pose_3d[1]}
    if verbose:
        observer.log(f"  Target position: {pose_3d[0].tolist()}")
        observer.log(f"  Target orientation: {pose_3d[1].tolist()}")
    q_traj = None
    planning_error_msg = None
    try:
        q_traj = controller._plan_joint_motion(
            target_pos=target_pos,
            target_quat=target_quat,
            embodiment_selection=CuRoboEmbodimentSelection.BASE,
            skip_obstacle_update=True,
        )
    except ActionPrimitiveError as exc:
        planning_error_msg = str(exc)

    if q_traj is not None:
        if verbose:
            observer.log(f"  [OK] Planned trajectory with {len(q_traj)} waypoints")
    else:
        if verbose:
            observer.log("\n  [FAILED] Navigation planning failed")
            if planning_error_msg:
                observer.log(f"  Error: {planning_error_msg}")
        if visualize:
            observer.show_navigation_failure(True, robot, target_pose_2d, controller._motion_generator)
    return q_traj


def select_optimal_pose(valid_poses, target_xy, robot_xy, robot_yaw, controller, robot, observer=None, verbose=False):
    scored_poses = [(pose, idx, compute_heuristic_score(pose, target_xy, robot_xy, robot_yaw)) for pose, idx in valid_poses]
    scored_poses.sort(key=lambda x: x[2])
    top_k_poses = scored_poses[: min(TOP_K_CANDIDATES, len(scored_poses))]
    if verbose:
        observer.log(f"\n  Stage 1: Selected top-{len(top_k_poses)} candidates from {len(valid_poses)} valid poses")
        observer.log("  Stage 2: Evaluating trajectory complexity...")
    best_pose = None
    best_idx = None
    best_traj = None
    best_traj_score = float("inf")
    best_traj_info = None
    for i, (pose_2d, global_idx, _) in enumerate(top_k_poses):
        try:
            q_traj = plan_navigation(controller=controller, robot=robot, target_pose_2d=pose_2d, observer=observer, verbose=False, visualize=False)
        except Exception:
            q_traj = None
        if q_traj is None:
            if verbose:
                observer.log(f"    Candidate {i+1}: Planning failed, skipped")
            continue
        path_length, cumulative_rotation = compute_trajectory_complexity(q_traj, robot)
        dist_to_target = th.norm(pose_2d[:2] - target_xy).item()
        traj_score = compute_trajectory_score(path_length, cumulative_rotation, dist_to_target)
        if verbose:
            observer.log(
                f"    Candidate {i+1}: path={path_length:.2f}m, "
                f"rotation={math.degrees(cumulative_rotation):.1f}deg, score={traj_score:.3f}"
            )
        if traj_score < best_traj_score:
            best_traj_score = traj_score
            best_pose = pose_2d
            best_idx = global_idx
            best_traj = q_traj
            best_traj_info = (path_length, cumulative_rotation, len(q_traj))
    if best_pose is None:
        if verbose:
            observer.log("  Warning: All trajectory planning failed, using heuristic best")
        best_pose, best_idx, _ = top_k_poses[0]
        best_traj = None
        best_traj_info = None
    return best_pose, best_idx, best_traj, best_traj_info


def find_optimal_grasp_base_pose(controller, obj, max_samples, observer, sampling_radius=DEFAULT_SAMPLING_RADIUS, verbose=True, visualize=False, seed=42):
    th.manual_seed(seed)
    robot = controller.robot
    arm = controller.arm
    stats = {
        "total_attempts": 0,
        "collision_failures": 0,
        "reachability_failures": 0,
        "room_failures": 0,
    }
    if verbose:
        observer.log("  Sampling grasp pose for object...")
    try:
        eef_pose, grasp_pose = controller._sample_grasp_pose(obj)
        observer.show_sampling_region(visualize, obj, eef_pose, DEFAULT_SAMPLING_RADIUS, verbose)
        if verbose:
            observer.log(f"  Pre-grasp position: {eef_pose[0].tolist()}")
            observer.log(f"  Grasp position: {grasp_pose[0].tolist()}")
    except Exception as exc:
        if verbose:
            observer.log(f"  [ERROR] Grasp pose sampling failed: {exc}")
        return GraspBasePoseResult(None, None, None, False, stats)

    target_pose = eef_pose
    target_xy = target_pose[0][:2]
    obj_rooms = obj.in_rooms if obj.in_rooms else [robot.scene._seg_map.get_room_instance_by_point(target_pose[0][:2])]
    if verbose:
        observer.log(f"  Object room(s): {obj_rooms}")

    avg_arm_workspace_range = th.mean(robot.arm_workspace_range[arm])
    if verbose:
        observer.log(f"  Sampling distance range: [0.00, {sampling_radius:.2f}]m")
        observer.log(f"  Average arm workspace range: {avg_arm_workspace_range:.3f}")
        observer.log(f"  Generating {max_samples} candidate poses...")

    controller._motion_generator.update_obstacles()
    candidate_poses = generate_candidate_poses(
        target_pos=target_pose[0],
        num_samples=max_samples,
        sampling_radius=sampling_radius,
        arm_workspace_offset=avg_arm_workspace_range.item(),
        min_radius=DEFAULT_MIN_RADIUS,
    )
    room_valid_poses, room_valid_indices = filter_poses_by_room(candidate_poses, obj_rooms, robot.scene._seg_map)
    stats["room_failures"] = max_samples - len(room_valid_indices)
    if verbose:
        observer.log(f"  Room filter: {len(room_valid_indices)}/{max_samples} passed")
    if len(room_valid_indices) == 0:
        stats["total_attempts"] = max_samples
        if verbose:
            observer.log("\n  [FAILED] All samples failed room check")
            observer.report_sampling_diagnostics(stats)
        return GraspBasePoseResult(None, None, None, False, stats)

    current_joint_pos = robot.get_joint_positions()
    batch_joint_positions = convert_poses_to_joint_positions(room_valid_poses, robot, current_joint_pos)
    obj_in_hand = controller._get_obj_in_hand()
    attached_obj = {robot.eef_link_names[arm]: obj_in_hand.root_link} if obj_in_hand else None
    if verbose:
        observer.log(f"  Running batch collision check for {len(room_valid_indices)} candidates...")
    collision_results = batch_collision_check(controller, batch_joint_positions, attached_obj)
    collision_free_indices = th.where(~collision_results)[0]
    stats["collision_failures"] = int(collision_results.sum().item())
    if verbose:
        observer.log(f"  Collision filter: {len(collision_free_indices)}/{len(room_valid_indices)} passed")

    collision_failed_poses = [room_valid_poses[i].clone() for i in range(len(room_valid_poses)) if collision_results[i].item()]
    collision_failed_joint_positions = [batch_joint_positions[i].clone() for i in range(len(batch_joint_positions)) if collision_results[i].item()]

    if len(collision_free_indices) == 0:
        stats["total_attempts"] = max_samples
        if verbose:
            observer.log("\n  [FAILED] All samples failed collision check")
            observer.report_sampling_diagnostics(stats)
        if visualize:
            observer.show_candidate_poses(
                True,
                robot,
                PoseVisualizationData(
                    collision_failed_poses=collision_failed_poses,
                    reachability_failed_poses=[],
                    collision_joint_positions=collision_failed_joint_positions,
                ),
                controller._motion_generator,
                True,
                MAX_VERBOSE_SAMPLES,
            )
        return GraspBasePoseResult(None, None, None, False, stats)

    if verbose:
        observer.log(f"  Checking reachability for {len(collision_free_indices)} collision-free candidates...")
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

    if valid_poses:
        robot_pos, robot_quat = robot.get_position_orientation()
        robot_xy = robot_pos[:2]
        robot_yaw = T.quat2euler(robot_quat)[2].item()
        best_pose, _, best_traj, _ = select_optimal_pose(
            valid_poses, target_xy, robot_xy, robot_yaw, controller, robot, observer=observer, verbose=verbose
        )
        stats["total_attempts"] = max_samples
        if verbose:
            observer.log("\n  [SUCCESS] Selected optimal pose based on trajectory complexity")
            observer.report_sampling_diagnostics(stats)
        if visualize:
            observer.show_candidate_poses(
                True,
                robot,
                PoseVisualizationData(
                    collision_failed_poses=collision_failed_poses,
                    reachability_failed_poses=reachability_failed_poses,
                    valid_poses=[pose for pose, _ in valid_poses],
                    selected_pose=best_pose,
                    target_position=target_xy,
                    collision_joint_positions=collision_failed_joint_positions,
                ),
                controller._motion_generator,
                False,
                MAX_VERBOSE_SAMPLES,
            )
        return GraspBasePoseResult(eef_pose, grasp_pose, best_pose, True, stats, q_traj=best_traj)

    stats["total_attempts"] = max_samples
    if verbose:
        observer.log(f"\n  [FAILED] No valid pose found after {max_samples} attempts")
        observer.report_sampling_diagnostics(stats)
    if visualize:
        observer.show_candidate_poses(
            True,
            robot,
            PoseVisualizationData(
                collision_failed_poses=collision_failed_poses,
                reachability_failed_poses=reachability_failed_poses,
                valid_poses=[],
                selected_pose=None,
                target_position=target_xy,
                collision_joint_positions=collision_failed_joint_positions,
            ),
            controller._motion_generator,
            True,
            MAX_VERBOSE_SAMPLES,
        )
    return GraspBasePoseResult(None, None, None, False, stats)


def find_optimal_base_pose(
    controller,
    target_eef_pose,
    obj,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    sampling_radius: float = DEFAULT_SAMPLING_RADIUS,
    plan_with_open_gripper: bool = False,
    skip_obstacle_update: bool = False,
    max_rounds: int = 3,
    min_radius: float = DEFAULT_MIN_RADIUS,
    seed: int = 42,
    return_traj: bool = False,
):
    """Find an optimal base pose near *target_eef_pose* using multi-round sampling.

    Round 0 uses a Fibonacci spiral for uniform coverage; subsequent rounds
    use random sampling with an increased sample count and a different seed
    for diversity.  Each round filters by room → collision → reachability,
    scores surviving candidates heuristically, and pre-validates navigation
    on the top-K before returning.

    Args:
        return_traj: If True, also return the navigation trajectory validated
            during pre-validation.  curobo's planner is stochastic, so reusing
            this trajectory avoids the risk of a failing second plan on the
            same inputs.  Default False preserves the original signature.

    Returns:
        If return_traj is False: a (x, y, yaw) tensor, or None.
        If return_traj is True: a tuple (pose, q_traj), or (None, None).
    """
    robot = controller.robot
    arm = controller.arm
    target_xy = target_eef_pose[0][:2]
    obj_rooms = (
        obj.in_rooms
        if obj.in_rooms
        else [robot.scene._seg_map.get_room_instance_by_point(target_xy)]
    )
    avg_arm_workspace_range = th.mean(robot.arm_workspace_range[arm])

    if not skip_obstacle_update:
        controller._motion_generator.update_obstacles()

    if plan_with_open_gripper:
        current_joint_pos = controller._get_joint_position_with_fingers_at_limit("upper")
    else:
        current_joint_pos = robot.get_joint_positions()

    obj_in_hand = controller._get_obj_in_hand()
    attached_obj = (
        {robot.eef_link_names[arm]: obj_in_hand.root_link} if obj_in_hand else None
    )

    robot_pos, robot_quat = robot.get_position_orientation()
    robot_xy = robot_pos[:2]
    robot_yaw = T.quat2euler(robot_quat)[2].item()

    for round_idx in range(max_rounds):
        n = max_samples * (round_idx + 1)
        strategy = (
            SamplingStrategy.FIBONACCI_SPIRAL
            if round_idx == 0
            else SamplingStrategy.RANDOM
        )
        th.manual_seed(seed + round_idx * 1000)

        candidates = generate_candidate_poses(
            target_pos=target_eef_pose[0],
            num_samples=n,
            sampling_radius=sampling_radius,
            arm_workspace_offset=avg_arm_workspace_range.item(),
            min_radius=min_radius,
            strategy=strategy,
        )

        # ---- Room filter ----
        room_valid, room_indices = filter_poses_by_room(
            candidates, obj_rooms, robot.scene._seg_map
        )
        if len(room_indices) == 0:
            continue

        # ---- Batch collision check ----
        batch_jp = convert_poses_to_joint_positions(
            room_valid, robot, current_joint_pos
        )
        collision_results = batch_collision_check(controller, batch_jp, attached_obj)
        collision_free = th.where(~collision_results)[0]
        if len(collision_free) == 0:
            continue

        # ---- Reachability check ----
        valid_poses: List[th.Tensor] = []
        for idx in collision_free:
            jp = batch_jp[idx]
            if controller._target_in_reach_of_robot(
                target_eef_pose,
                initial_joint_pos=jp,
                skip_obstacle_update=True,
            ):
                valid_poses.append(room_valid[idx])
        if not valid_poses:
            continue

        # ---- Heuristic scoring ----
        scored = sorted(
            valid_poses,
            key=lambda p: compute_heuristic_score(
                p, target_xy, robot_xy, robot_yaw
            ),
        )

        # ---- Navigation pre-validation on top candidates ----
        top_k = min(TOP_K_CANDIDATES + 2, len(scored))
        for pose in scored[:top_k]:
            q_traj = plan_navigation(
                controller=controller,
                robot=robot,
                target_pose_2d=pose,
                verbose=False,
            )
            if q_traj is not None:
                return (pose, q_traj) if return_traj else pose

        # Try remaining candidates
        for pose in scored[top_k:]:
            q_traj = plan_navigation(
                controller=controller,
                robot=robot,
                target_pose_2d=pose,
                verbose=False,
            )
            if q_traj is not None:
                return (pose, q_traj) if return_traj else pose

    return (None, None) if return_traj else None