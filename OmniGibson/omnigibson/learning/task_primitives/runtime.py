"""Runtime helpers for launching a task_primitives scene outside the eval loop."""

import csv
import json
import os
import time
from pathlib import Path
from typing import Optional

import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch as th

from gello.utils.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.utils.og_teleop_utils import load_available_tasks
from omnigibson.learning.task_primitives.context import GraspExecutionConfig, create_action_context
from omnigibson.learning.task_primitives.models import DEFAULT_IMAGE_SIZE, NUM_EVAL_INSTANCES
from omnigibson.learning.utils.robot_config_utils import (
    build_r1pro_primitives_robot_config,
    resolve_presampled_robot_pose,
)
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES, generate_basic_environment_config
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.bddl_utils import is_system_bddl_inst
from omnigibson.utils.python_utils import recursively_convert_to_torch


def apply_low_res_rgb(env, image_size: int = DEFAULT_IMAGE_SIZE) -> None:
    """Shrink robot camera outputs so manual debugging starts faster."""
    robot = env.robots[0]
    for sensor in robot.sensors.values():
        if hasattr(sensor, "image_height") and hasattr(sensor, "image_width"):
            sensor.image_height = image_size
            sensor.image_width = image_size
    # Refresh physics sim views before reloading the observation space.
    # After Environment.__init__() completes, the physics sim view can become
    # stale (e.g. from scene.reset() during env.reset()). If the robot uses
    # proprio observations, load_observation_space() will query joint positions
    # which requires a valid physics sim view.
    og.sim.update_handles()
    env.load_observation_space()


def set_viewer_camera_to_robot(env, distance: float = 3.0, height: float = 2.0) -> None:
    """Place the viewer behind the robot for quick interactive inspection."""
    if gm.HEADLESS or og.sim.viewer_camera is None:
        return
    robot = env.robots[0]
    robot_pos, robot_quat = robot.get_position_orientation()
    robot_pos = th.as_tensor(robot_pos)
    robot_quat = th.as_tensor(robot_quat)
    robot_rot_mat = T.quat2mat(robot_quat)
    robot_forward = robot_rot_mat[:, 0]
    cam_pos = robot_pos.clone()
    cam_pos[0] -= robot_forward[0] * distance
    cam_pos[1] -= robot_forward[1] * distance
    cam_pos[2] += height
    look_dir = robot_pos - cam_pos
    look_dir[2] += 0.8
    look_dir = look_dir / th.norm(look_dir)
    up = th.tensor([0.0, 0.0, 1.0], dtype=look_dir.dtype, device=look_dir.device)
    forward = look_dir
    right = th.cross(forward, up, dim=0)
    right_norm = th.norm(right)
    if right_norm < 1e-6:
        up = th.tensor([0.0, 1.0, 0.0], dtype=look_dir.dtype, device=look_dir.device)
        right = th.cross(forward, up, dim=0)
    right = right / th.norm(right)
    up = th.cross(right, forward, dim=0)
    up = up / th.norm(up)
    rot_mat = th.stack([right, up, -forward], dim=1)
    cam_quat = T.mat2quat(rot_mat)
    og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_quat)


def get_eval_instance_ids(task_name: str, eval_instance_ids: Optional[list[int]] = None) -> list[int]:
    if eval_instance_ids is None:
        eval_instance_ids = list(range(NUM_EVAL_INSTANCES))
    eval_instance_ids = list(eval_instance_ids)
    assert set(eval_instance_ids).issubset(set(range(NUM_EVAL_INSTANCES))), f"eval_instance_ids must be in range({NUM_EVAL_INSTANCES})"
    task_instance_csv_path = os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv")
    with open(task_instance_csv_path, "r") as file:
        lines = list(csv.reader(file))[1:]
    task_idx = TASK_NAMES_TO_INDICES[task_name]
    assert lines[task_idx][1] == task_name, f"Task name from args {task_name} does not match task name from csv {lines[task_idx][1]}"
    test_instances = lines[task_idx][2].strip().split(",")
    return [int(test_instances[i]) for i in eval_instance_ids]


def load_task_instance(env, robot, instance_id: int, test_hidden: bool = False) -> None:
    """Load a presampled task instance and settle the scene before execution."""
    scene_model = env.task.scene_name
    tro_filename = env.task.get_cached_activity_scene_filename(
        scene_model=scene_model,
        activity_name=env.task.activity_name,
        activity_definition_id=env.task.activity_definition_id,
        activity_instance_id=instance_id,
    )
    if test_hidden:
        tro_file_path = os.path.join(gm.DATA_PATH, "2025-challenge-test-instances", env.task.activity_name, f"{tro_filename}-tro_state.json")
    else:
        tro_file_path = os.path.join(
            get_task_instance_path(
                scene_model,
                f"{scene_model}_task_{env.task.activity_name}_instances/{tro_filename}-tro_state",
            )
        )
    with open(tro_file_path, "r") as file:
        tro_state = recursively_convert_to_torch(json.load(file))
    for tro_key, state in tro_state.items():
        if tro_key == "robot_poses":
            robot_poses = resolve_presampled_robot_pose(state, robot.model)
            robot_pos = robot_poses[0]["position"]
            robot_quat = robot_poses[0]["orientation"]
            robot.set_position_orientation(robot_pos, robot_quat)
            env.scene.write_task_metadata(key=tro_key, data=state)
        else:
            env.task.object_scope[tro_key].load_state(state, serialized=False)
    for _ in range(25):
        og.sim.step_physics()
        for inst, entity in env.task.object_scope.items():
            if not is_system_bddl_inst(inst) and entity is not None:
                entity.keep_still()
    env.scene.update_initial_file()
    env.scene.reset()


def load_env(task_name: str = "turning_on_radio", headless: bool = False):
    """Create a standalone environment configured the same way as the expert runtime."""
    for rule in DISABLED_TRANSITION_RULES:
        rule.ENABLED = False
    available_tasks = load_available_tasks()
    assert task_name in available_tasks, f"Got invalid task name: {task_name}"
    task_idx = TASK_NAMES_TO_INDICES[task_name]
    human_stats = {"length": []}
    episodes_path = os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl")
    if os.path.exists(episodes_path):
        with open(episodes_path, "r") as file:
            episodes = [json.loads(line) for line in file]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                human_stats["length"].append(episode["length"])
    avg_length = sum(human_stats["length"]) / len(human_stats["length"]) if human_stats["length"] else 2500
    task_cfg = available_tasks[task_name][0]
    cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
    # Reuse the same robot config builder as eval so both entrypoints stay behaviorally aligned.
    cfg["robots"] = [build_r1pro_primitives_robot_config(task_cfg)]
    cfg["task"]["termination_config"]["max_steps"] = int(avg_length * 2)
    cfg["task"]["include_obs"] = False
    gm.HEADLESS = headless
    print(f"Loading environment for task: {task_name}")
    env = og.Environment(configs=cfg)
    apply_low_res_rgb(env=env, image_size=DEFAULT_IMAGE_SIZE)
    return env


def resolve_scene_object(scene, object_name: str, task=None):
    if task is not None and object_name in task.object_scope:
        return task.object_scope[object_name]
    obj = scene.object_registry("name", object_name)
    if obj is None:
        raise ValueError(f"Object '{object_name}' not found in scene")
    return obj


def warmup_environment(step_count: int = 30) -> None:
    for _ in range(step_count):
        og.sim.step()


def build_grasp_context(task_name: str, headless: bool, eval_instance_ids: Optional[list[int]], config: GraspExecutionConfig):
    """Create a ready-to-run ActionContext for a chosen task instance."""
    env = load_env(task_name=task_name, headless=headless)
    robot = env.robots[0]
    instance_ids = get_eval_instance_ids(task_name=task_name, eval_instance_ids=eval_instance_ids)
    if instance_ids:
        instance_id = instance_ids[0]
        load_task_instance(env=env, robot=robot, instance_id=instance_id, test_hidden=False)
    set_viewer_camera_to_robot(env)
    warmup_environment()
    return create_action_context(env=env, config=config, enable_head_tracking=False, curobo_batch_size=1)


def shutdown_simulation(delay_seconds: float = 5.0) -> None:
    """Best-effort shutdown helper for dagger.py and other manual entrypoints."""
    try:
        time.sleep(delay_seconds)
        og.shutdown()
    except Exception:
        pass