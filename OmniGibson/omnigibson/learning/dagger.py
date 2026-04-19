"""
DAgger entrypoint built on reusable task primitives with LeRobot trajectory recording.
"""

import argparse
import json
import logging
import os
import traceback
import types

import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch as th
from gello.utils.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.utils.og_teleop_utils import load_available_tasks
from omnigibson.controllers import ControllerView, JointController, HolonomicBaseJointController, MultiFingerGripperController
from omnigibson.envs.lerobot_data_wrapper import LeRobotDataWrapper
from omnigibson.utils.geometry_utils import wrap_angle
from omnigibson.learning.task_primitives import (
    CustomGraspBackend,
    FallbackGraspBackend,
    GraspAndPlaceInsideTask,
    GraspExecutionConfig,
    NavigateAndGraspTask,
    PrimitiveGraspBackend,
    create_action_context,
    report_grasp_debug_context,
    shutdown_simulation,
)
from omnigibson.learning.task_primitives.registry import resolve_task_spec
from omnigibson.learning.task_primitives.runtime import (
    apply_low_res_rgb,
    get_eval_instance_ids,
    load_task_instance,
    set_viewer_camera_to_robot,
    warmup_environment,
)
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    TASK_NAMES_TO_INDICES,
    generate_basic_environment_config,
)
from omnigibson.learning.utils.robot_config_utils import build_r1pro_primitives_robot_config
from omnigibson.macros import gm


BACKEND_FACTORIES = {
    "custom": lambda: CustomGraspBackend(),
    "primitive": lambda: PrimitiveGraspBackend(),
    "both": lambda: FallbackGraspBackend(CustomGraspBackend(), PrimitiveGraspBackend()),
}


def parse_args():
    parser = argparse.ArgumentParser(description="DAgger data collection with LeRobot recording")
    parser.add_argument("--task_name", type=str, default="turning_on_radio", help="Name of the task to load")
    parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode (no GUI)")
    parser.add_argument(
        "--eval_instance_ids",
        type=str,
        default=None,
        help="Comma-separated list of instance IDs to evaluate",
    )
    parser.add_argument("--object_name", type=str, default=None, help="Name of object to grasp (overrides registry)")
    parser.add_argument("--target_name", type=str, default=None, help="Name of target for place-inside tasks")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Absolute path for the LeRobot dataset output directory (e.g. '/data/dagger/turning_on_radio')",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable visualization of obstacles, poses, and trajectories",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum sampling attempts for pose near object",
    )
    parser.add_argument(
        "--grasp_mode",
        type=str,
        choices=["custom", "primitive", "both"],
        default="custom",
        help="Grasp execution mode backend",
    )
    parser.add_argument(
        "--primitive_attempts",
        type=int,
        default=5,
        help="Number of retries for the built-in GRASP primitive backend",
    )
    return parser.parse_args()


def parse_eval_instance_ids(raw_value: str | None):
    if not raw_value:
        return None
    return [int(value) for value in raw_value.split(",") if value.strip()]


logger = logging.getLogger("dagger")


def _patch_q_to_action(robot):
    """
    Monkey-patch robot.q_to_action to support MultiFingerGripperController.

    The original q_to_action asserts all controllers are JointController, which fails
    when grippers use MultiFingerGripperController. This patch handles gripper controllers
    by converting target joint positions into a binary open/close command.
    """

    def patched_q_to_action(self, q):
        action = []
        for name, (group_key, _) in self.controllers.items():
            command = q[ControllerView.get_dof_idx(group_key)]

            if ControllerView.is_controller_type(group_key, MultiFingerGripperController):
                gripper_dof_idx = ControllerView.get_dof_idx(group_key)
                lower = self.joint_lower_limits[gripper_dof_idx]
                upper = self.joint_upper_limits[gripper_dof_idx]
                midpoint = (lower + upper) / 2.0
                mean_target = command.mean()
                mean_mid = midpoint.mean()
                binary_cmd = th.tensor([1.0], device=command.device, dtype=command.dtype) if mean_target >= mean_mid else th.tensor([-1.0], device=command.device, dtype=command.dtype)
                action.append(binary_cmd)
            elif ControllerView.is_controller_type(group_key, HolonomicBaseJointController):
                base_joint_pos = self.get_joint_positions()[self.base_idx]
                cur_rz_joint_pos = base_joint_pos[5]
                delta_q = wrap_angle(command[2] - cur_rz_joint_pos)
                body_pos = base_joint_pos[:3]
                body_quat = T.mat2quat(T.euler_intrinsic2mat(base_joint_pos[3:6]))
                canonical_pos = th.tensor([command[0], command[1], body_pos[2]], dtype=th.float32, device=command.device)
                local_pos = T.relative_pose_transform(
                    canonical_pos, th.tensor([0.0, 0.0, 0.0, 1.0], device=command.device), body_pos, body_quat
                )[0]
                command = th.tensor([local_pos[0], local_pos[1], delta_q], device=local_pos.device)
                action.append(ControllerView.reverse_preprocess_command(group_key, command))
            else:
                assert (
                    ControllerView.is_controller_type(group_key, JointController)
                    and not ControllerView.get_use_delta_commands(group_key)
                ), f"Controller [{name}] should be a JointController with use_delta_commands=False!"
                action.append(ControllerView.reverse_preprocess_command(group_key, command))

        action = th.cat(action, dim=0)
        assert (
            action.shape[0] == self.action_dim
        ), f"Action should have dimension {self.action_dim}, got {action.shape[0]}"
        return action

    robot.q_to_action = types.MethodType(patched_q_to_action, robot)
    logger.info("Patched robot.q_to_action to support MultiFingerGripperController.")


def load_env_with_obs(task_name: str, headless: bool):
    """Create an environment with proprio + rgb observations and 23-dim actions for LeRobot recording."""
    for rule in DISABLED_TRANSITION_RULES:
        rule.ENABLED = False

    available_tasks = load_available_tasks()
    if task_name not in available_tasks:
        raise ValueError(f"Got invalid task name: {task_name}")

    task_idx = TASK_NAMES_TO_INDICES[task_name]
    human_stats = {"length": []}
    episodes_path = os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl")
    if os.path.exists(episodes_path):
        with open(episodes_path, "r") as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 10000 == task_idx:
                human_stats["length"].append(episode["length"])
    avg_length = sum(human_stats["length"]) / len(human_stats["length"]) if human_stats["length"] else 2500

    task_cfg = available_tasks[task_name][0]
    cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
    cfg["robots"] = [
        build_r1pro_primitives_robot_config(
            task_cfg,
            obs_modalities=["proprio", "rgb"],
            proprio_obs=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
        )
    ]
    # Replace gripper controllers with MultiFingerGripperController (binary mode)
    # so action_dim = 23 (1 DOF per gripper instead of 2), matching the websocket policy.
    robot_cfg = cfg["robots"][0]
    for gripper_key in ("gripper_left", "gripper_right"):
        robot_cfg["controller_config"][gripper_key] = {
            "name": "MultiFingerGripperController",
            "motor_type": "position",
            "command_output_limits": "default",
            "mode": "binary",
            "limit_tolerance": 0.001,
        }
    cfg["task"]["termination_config"]["max_steps"] = int(avg_length * 2)
    cfg["task"]["include_obs"] = False
    # LeRobotDataWrapper expects flattened obs keys like "robot_r1::sensor::modality"
    cfg["env"]["flatten_obs_space"] = True
    gm.HEADLESS = headless

    print(f"Loading environment for task: {task_name}")
    env = og.Environment(configs=cfg)
    apply_low_res_rgb(env)
    return env


def main():
    try:
        args = parse_args()
        task_name = args.task_name

        # --- Create environment with proper obs modalities ---
        env = load_env_with_obs(task_name=task_name, headless=args.headless)
        robot = env.robots[0]
        # Patch q_to_action so CuRobo works with MultiFingerGripperController
        _patch_q_to_action(robot)

        # --- Wrap with LeRobotDataWrapper for recording ---
        # LeRobotDataWrapper.create_dataset iterates env.external_sensors;
        # default to empty dict when no external sensors are configured.
        if env.external_sensors is None:
            env._external_sensors = {}
        # LeRobotDataWrapper stores data at {root_dir}/{output_path}.
        # Split the user-provided absolute path so data lands exactly where requested.
        abs_output = os.path.abspath(args.output_path)
        root_dir = os.path.dirname(abs_output)
        repo_id = os.path.basename(abs_output)
        wrapped_env = LeRobotDataWrapper(
            env=env,
            output_path=repo_id,
            root_dir=root_dir,
            only_successes=False,
            task_name=task_name.replace("_", " "),
        )

        # --- Resolve task instances ---
        instance_ids = get_eval_instance_ids(
            task_name=task_name,
            eval_instance_ids=parse_eval_instance_ids(args.eval_instance_ids),
        )
        print(f"Will run {len(instance_ids)} instance(s): {instance_ids}")

        # --- Resolve task spec ---
        spec = resolve_task_spec(task_name, object_name=args.object_name, target_name=args.target_name)
        backend = BACKEND_FACTORIES[args.grasp_mode]()

        config = GraspExecutionConfig(
            max_samples=args.max_samples,
            primitive_attempts=args.primitive_attempts,
            verbose=True,
            visualize=args.visualize,
        )

        # --- Loop over instances ---
        for idx, instance_id in enumerate(instance_ids):
            print(f"\n{'='*60}")
            print(f"Instance {idx+1}/{len(instance_ids)}: id={instance_id}")
            print(f"{'='*60}")

            try:
                # Reset env and record initial obs via wrapper
                wrapped_env.reset()

                # Load the specific task instance (modifies scene state)
                load_task_instance(env=wrapped_env, robot=robot, instance_id=instance_id)
                set_viewer_camera_to_robot(wrapped_env)
                warmup_environment()

                # Re-capture initial obs after task instance loading
                # (the reset obs is stale since load_task_instance changed the scene)
                obs, info = wrapped_env.env.get_obs()
                wrapped_env.current_obs = obs
                assert len(wrapped_env.current_traj_history) > 0, "reset() should have populated current_traj_history"
                wrapped_env.current_traj_history[-1]["obs"] = wrapped_env._process_obs(obs, info)

                # Create action context with the wrapped env
                context = create_action_context(
                    env=wrapped_env,
                    config=config,
                    enable_head_tracking=False,
                    curobo_batch_size=1,
                )

                # Debug snapshot
                obj_name = args.object_name or spec.object_name
                obj = context.scene.object_registry("name", obj_name)
                if obj is not None:
                    report_grasp_debug_context(context, obj, max_samples=args.max_samples)

                # Build and run task
                if spec.task_type == "navigate_and_grasp":
                    task = NavigateAndGraspTask(object_name=spec.object_name, backend=backend)
                elif spec.task_type == "grasp_and_place_inside":
                    task = GraspAndPlaceInsideTask(
                        object_name=spec.object_name,
                        target_name=spec.target_name,
                        backend=backend,
                    )
                else:
                    raise ValueError(f"Unsupported task type: {spec.task_type}")

                result = task.run(context)

                if result.success:
                    print(f"[OK] Instance {instance_id}: backend={result.backend}")
                else:
                    print(f"[FAILED] Instance {instance_id}: backend={result.backend}, result={result.result.value}")
                    if result.error_message:
                        print(f"  Error: {result.error_message}")
            except Exception:
                traceback.print_exc()
                print(f"[SKIP] Instance {instance_id} failed unexpectedly, continuing...")
                continue

        # --- Finalize dataset ---
        wrapped_env.save_data()
        print(f"\nLeRobot dataset saved to: {abs_output}")
        print(f"Total episodes recorded: {wrapped_env.traj_count}")

    except Exception:
        traceback.print_exc()
    finally:
        shutdown_simulation()


if __name__ == "__main__":
    main()
