"""Observer implementations for logging and visualization during primitive execution."""

import math

import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch as th

from omnigibson.learning.task_primitives.models import GraspDebugSnapshot, PoseColors, PoseVisualizationData
from omnigibson.utils.ui_utils import clear_debug_drawing, draw_box
from omnigibson.utils.visualize_utils import (
    visualize_2d_pose,
    visualize_obstacles,
    visualize_robot_and_sampled_pose,
    visualize_robot_spheres_at_config,
    visualize_sampling_region,
    visualize_trajectory,
)


class NullObserver:
    """Drop-in observer that disables all side effects."""
    def section(self, title: str) -> None:
        pass

    def log(self, message: str) -> None:
        pass

    def report_sampling_diagnostics(self, stats) -> None:
        pass

    def report_primitive_failure(self, exc) -> None:
        pass

    def report_grasp_debug_snapshot(self, snapshot: GraspDebugSnapshot) -> None:
        pass

    def maybe_clear_debug_drawing(self, enabled: bool) -> None:
        pass

    def show_sampling_region(self, enabled: bool, obj, eef_pose, radius: float, verbose: bool) -> None:
        pass

    def show_navigation_failure(self, enabled: bool, robot, sampled_pose, motion_generator) -> None:
        pass

    def show_candidate_poses(self, enabled: bool, robot, viz_data: PoseVisualizationData, motion_generator, show_collision_spheres: bool, max_display_per_category: int) -> None:
        pass

    def show_sampled_pose(self, enabled: bool, robot, sampled_pose, verbose: bool) -> None:
        pass

    def show_trajectory(self, enabled: bool, q_traj, robot, verbose: bool) -> None:
        pass


class ConsoleObserver(NullObserver):
    """Observer that prints diagnostics and optionally drives debug visualizations."""
    def section(self, title: str) -> None:
        print(f"\n{'=' * 50}")
        print(title)
        print(f"{'=' * 50}")

    def log(self, message: str) -> None:
        print(message)

    def report_sampling_diagnostics(self, stats) -> None:
        total_attempts = stats.get("total_attempts", 0)
        room_failures = stats.get("room_failures", 0)
        collision_failures = stats.get("collision_failures", 0)
        reachability_failures = stats.get("reachability_failures", 0)
        room_passes = max(total_attempts - room_failures, 0)
        collision_passes = max(room_passes - collision_failures, 0)
        reachable_passes = max(collision_passes - reachability_failures, 0)
        self.log("  Sampling diagnostics:")
        self.log(f"    total_attempts={total_attempts}")
        self.log(f"    room_passes={room_passes}, room_failures={room_failures}")
        self.log(f"    collision_passes={collision_passes}, collision_failures={collision_failures}")
        self.log(f"    reachable_passes={reachable_passes}, reachability_failures={reachability_failures}")

    def report_primitive_failure(self, exc) -> None:
        self.log(f"[FAILED] Primitive grasp failed after {len(exc.exceptions)} attempts")
        for idx, error in enumerate(exc.exceptions):
            self.log(
                f"  Attempt {idx}: reason={error.reason.name}, "
                f"message={str(error)}, metadata={error.metadata}"
            )

    def report_grasp_debug_snapshot(self, snapshot: GraspDebugSnapshot) -> None:
        self.log("\n" + "=" * 60)
        self.log("Grasp debug context")
        self.log("=" * 60)
        self.log(f"Object: {snapshot.object_name}")
        self.log(f"Requested max_samples: {snapshot.max_samples}")
        self.log(f"Object in_rooms: {snapshot.object_rooms}")
        self.log(f"Robot position: {snapshot.robot_position}")
        self.log(f"Robot yaw: {snapshot.robot_yaw_deg:.1f} deg")
        self.log(f"Robot holonomic base: {snapshot.is_holonomic_base}")
        self.log(f"Robot articulated trunk: {snapshot.is_articulated_trunk}")
        if snapshot.sampling_error:
            self.log(f"Pre-grasp sampling failed before execution: {snapshot.sampling_error}")
        else:
            self.log(f"Pre-grasp position: {snapshot.pregrasp_position}")
            self.log(f"Grasp position: {snapshot.grasp_position}")
            self.log(f"Inferred target rooms: {snapshot.inferred_rooms}")
        self.log("=" * 60)

    def maybe_clear_debug_drawing(self, enabled: bool) -> None:
        if enabled:
            clear_debug_drawing()

    def show_sampling_region(self, enabled: bool, obj, eef_pose, radius: float, verbose: bool) -> None:
        if enabled:
            visualize_sampling_region(obj=obj, eef_pose=eef_pose, radius=radius, verbose=verbose)

    def show_navigation_failure(self, enabled: bool, robot, sampled_pose, motion_generator) -> None:
        if not enabled:
            return
        self.log("\n  Visualizing navigation failure...")
        # Hand control back to the viewer so failed plans can be inspected interactively.
        visualize_robot_and_sampled_pose(robot=robot, sampled_pose=sampled_pose, verbose=False)
        visualize_obstacles(motion_generator=motion_generator, verbose=True)
        og.sim.enable_viewer_camera_teleoperation()
        try:
            while True:
                og.sim.step()
        except KeyboardInterrupt:
            pass

    def show_candidate_poses(self, enabled: bool, robot, viz_data: PoseVisualizationData, motion_generator, show_collision_spheres: bool, max_display_per_category: int) -> None:
        if not enabled:
            return
        z_height = 0.1
        if viz_data.target_position is not None:
            target_xy = viz_data.target_position[:2]
            draw_box(
                center=[target_xy[0].item(), target_xy[1].item(), z_height],
                extents=[0.15, 0.15, 0.02],
                color=PoseColors.TARGET,
                size=3.0,
            )
        robot_pos, robot_quat = robot.get_position_orientation()
        robot_yaw = T.quat2euler(robot_quat)[2].item()
        robot_2d_pose = th.tensor([robot_pos[0], robot_pos[1], robot_yaw])
        visualize_2d_pose(robot_2d_pose, z_height=z_height + 0.05, arrow_length=0.5, color=PoseColors.ROBOT_CURRENT, verbose=False)
        for pose in viz_data.collision_failed_poses[:max_display_per_category]:
            visualize_2d_pose(pose, z_height=z_height, arrow_length=0.2, color=PoseColors.COLLISION_FAILED, verbose=False)
        for pose in viz_data.reachability_failed_poses[:max_display_per_category]:
            visualize_2d_pose(pose, z_height=z_height + 0.01, arrow_length=0.2, color=PoseColors.REACHABILITY_FAILED, verbose=False)
        for pose in viz_data.valid_poses[:max_display_per_category]:
            visualize_2d_pose(pose, z_height=z_height + 0.02, arrow_length=0.35, color=PoseColors.VALID, verbose=False)
        if viz_data.selected_pose is not None:
            selected = viz_data.selected_pose
            draw_box(
                center=[selected[0].item(), selected[1].item(), z_height + 0.03],
                extents=[0.2, 0.2, 0.02],
                color=PoseColors.SELECTED,
                size=3.0,
            )
            visualize_2d_pose(selected, z_height=z_height + 0.04, arrow_length=0.5, color=PoseColors.SELECTED, verbose=False)
        if show_collision_spheres and motion_generator is not None and viz_data.collision_joint_positions:
            visualize_robot_spheres_at_config(
                motion_generator=motion_generator,
                joint_positions=viz_data.collision_joint_positions[0],
                verbose=True,
            )

    def show_sampled_pose(self, enabled: bool, robot, sampled_pose, verbose: bool) -> None:
        if enabled:
            visualize_robot_and_sampled_pose(robot=robot, sampled_pose=sampled_pose, verbose=verbose)
            for _ in range(30):
                og.sim.step()

    def show_trajectory(self, enabled: bool, q_traj, robot, verbose: bool) -> None:
        if enabled:
            visualize_trajectory(q_traj=q_traj, robot=robot, verbose=verbose)
            for _ in range(30):
                og.sim.step()


def build_grasp_debug_snapshot(context, obj, max_samples: int) -> GraspDebugSnapshot:
    """Capture the planning inputs once so failures can be reproduced from logs alone."""
    robot = context.robot
    controller = context.controller
    robot_pos, robot_quat = robot.get_position_orientation()
    robot_yaw = T.quat2euler(robot_quat)[2].item()
    try:
        eef_pose, grasp_pose = controller._sample_grasp_pose(obj)
        inferred_rooms = obj.in_rooms if obj.in_rooms else [
            robot.scene._seg_map.get_room_instance_by_point(eef_pose[0][:2])
        ]
        return GraspDebugSnapshot(
            object_name=obj.name,
            max_samples=max_samples,
            object_rooms=list(obj.in_rooms),
            robot_position=[round(x, 3) for x in robot_pos.tolist()],
            robot_yaw_deg=math.degrees(robot_yaw),
            is_holonomic_base=robot.is_holonomic_base,
            is_articulated_trunk=robot.is_articulated_trunk,
            pregrasp_position=[round(x, 3) for x in eef_pose[0].tolist()],
            grasp_position=[round(x, 3) for x in grasp_pose[0].tolist()],
            inferred_rooms=inferred_rooms,
        )
    except Exception as exc:
        return GraspDebugSnapshot(
            object_name=obj.name,
            max_samples=max_samples,
            object_rooms=list(obj.in_rooms),
            robot_position=[round(x, 3) for x in robot_pos.tolist()],
            robot_yaw_deg=math.degrees(robot_yaw),
            is_holonomic_base=robot.is_holonomic_base,
            is_articulated_trunk=robot.is_articulated_trunk,
            sampling_error=str(exc),
        )
