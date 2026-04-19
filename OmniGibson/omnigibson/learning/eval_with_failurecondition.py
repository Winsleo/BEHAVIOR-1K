import csv
import cv2
import hydra
import json
import logging
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import os
import sys
import torch as th
import traceback
from av.container import Container
from av.stream import Stream
from gello.utils.og_teleop_utils import (
    augment_rooms,
    load_available_tasks,
    get_task_relevant_room_types,
)
from gello.utils.og_teleop_cfg import DISABLED_TRANSITION_RULES
from hydra.utils import instantiate
from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    generate_basic_environment_config,
    flatten_obs_dict,
    TASK_NAMES_TO_INDICES,
)
from omnigibson.learning.utils.robot_config_utils import (
    build_r1pro_primitives_robot_config,
    resolve_presampled_robot_pose,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    write_video,
)
from omnigibson.learning.policies import TaskPrimitivesExpertPolicy
from omnigibson.controllers import ControllerView, JointController, HolonomicBaseJointController, MultiFingerGripperController
from omnigibson.macros import gm, create_module_macros, macros
from omnigibson.metrics import MetricBase, AgentMetric, TaskMetric
from omnigibson.robots import Robot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.bddl_utils import is_system_bddl_inst
from omnigibson.utils.geometry_utils import wrap_angle
from omnigibson.utils.python_utils import recursively_convert_to_torch
from pathlib import Path
from signal import signal, SIGINT
from typing import Any, Tuple, List

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 10

# set global variables to boost performance
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

# Set grasp window to larger value to account for hard grasps
with macros.unlocked():
    macros.robots.manipulation_robot.GRASP_WINDOW = 0.75


# create module logger
logger = logging.getLogger("evaluator_with_failurecondition")
logger.setLevel(20)  # info

# Default expert-policy hyper-parameters (mirrors task_primitives_expert.yaml).
EXPERT_DEFAULTS = dict(
    grasp_mode="custom",
    max_samples=50,
    primitive_attempts=5,
    object_name=None,
    task_object_overrides={
        "turning_on_radio": "radio_89",
        "picking_up_trash": "can__of__soda.n.01_2",
    },
    target_name=None,
    task_target_overrides={
        "picking_up_trash": "ashcan.n.01_1",
    },
    verbose=False,
    visualize=False,
    action_mode="velocity",
)

def _patch_q_to_action(robot):
    """
    Monkey-patch robot.q_to_action to support MultiFingerGripperController.

    The original q_to_action asserts all controllers are JointController, which fails
    when grippers use MultiFingerGripperController. This patch handles gripper controllers
    by converting target joint positions into a binary open/close command.
    """
    import types

    original_q_to_action = robot.q_to_action

    def patched_q_to_action(self, q):
        action = []
        for name, (group_key, _) in self.controllers.items():
            command = q[ControllerView.get_dof_idx(group_key)]

            if ControllerView.is_controller_type(group_key, MultiFingerGripperController):
                # Convert N-dim joint positions to 1-dim binary command.
                # grasping_direction == "lower" means lower joint limit = closed grasp.
                gripper_dof_idx = ControllerView.get_dof_idx(group_key)
                lower = self.joint_lower_limits[gripper_dof_idx]
                upper = self.joint_upper_limits[gripper_dof_idx]
                midpoint = (lower + upper) / 2.0
                # If target is below midpoint → closing → negative command; above → opening → positive
                mean_target = command.mean()
                mean_mid = midpoint.mean()
                binary_cmd = th.tensor([1.0]) if mean_target >= mean_mid else th.tensor([-1.0])
                action.append(binary_cmd)
            elif ControllerView.is_controller_type(group_key, HolonomicBaseJointController):
                base_joint_pos = self.get_joint_positions()[self.base_idx]
                cur_rz_joint_pos = base_joint_pos[5]
                delta_q = wrap_angle(command[2] - cur_rz_joint_pos)
                body_pos = base_joint_pos[:3]
                body_quat = T.mat2quat(T.euler_intrinsic2mat(base_joint_pos[3:6]))
                canonical_pos = th.tensor([command[0], command[1], body_pos[2]], dtype=th.float32)
                local_pos = T.relative_pose_transform(
                    canonical_pos, th.tensor([0.0, 0.0, 0.0, 1.0]), body_pos, body_quat
                )[0]
                command = th.tensor([local_pos[0], local_pos[1], delta_q])
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

def set_viewer_camera_to_robot(env, distance: float = 3.0, height: float = 2.0) -> None:
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
    right = th.cross(forward, up)
    right_norm = th.norm(right)
    if right_norm < 1e-6:
        up = th.tensor([0.0, 1.0, 0.0], dtype=look_dir.dtype, device=look_dir.device)
        right = th.cross(forward, up)
    right = right / th.norm(right)
    up = th.cross(right, forward)
    up = up / th.norm(up)

    rot_mat = th.stack([right, up, -forward], dim=1)
    cam_quat = T.mat2quat(rot_mat)

    og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_quat)


class Evaluator:
    """
    Evaluator with failure-condition fallback.

    Extends the standard evaluation loop so that when the primary policy times out
    (done=True, success=False), a TaskPrimitivesExpertPolicy takes over and keeps
    generating actions until the task succeeds or the extended timeout fires.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()

        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        self.robot = self.load_robot()
        # Patch q_to_action so CuRobo (used by expert policy) works with MultiFingerGripperController
        _patch_q_to_action(self.robot)
        self.policy = self.load_policy()
        self.metrics = self.load_metrics()

        # --- Failure-condition fallback: expert policy ---
        self.expert_policy = self._load_expert_policy()
        self._use_expert = False
        # Remember the original max_steps so we can restore it on reset
        self._original_max_steps = self.env.task._termination_conditions["timeout"]._max_steps

        self.reset()
        # manually reset environment episode number
        self.env._current_episode = 0
        self._video_writer = None

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        """
        Read the environment config file and create the environment.
        The config file is located in the configs/envs directory.
        """
        # Disable a subset of transition rules for data collection
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False
        # Load config file
        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"
        # Now, get human stats of the task
        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for k in self.human_stats.keys():
                    self.human_stats[k].append(episode[k])
        # take a mean
        for k in self.human_stats.keys():
            self.human_stats[k] = sum(self.human_stats[k]) / len(self.human_stats[k])

        # Load the seed instance by default
        task_cfg = available_tasks[task_name][0]
        robot_type = self.cfg.robot.type
        assert robot_type == "R1Pro", f"Got invalid robot type: {robot_type}, only R1Pro is supported."
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            build_r1pro_primitives_robot_config(
                task_cfg,
                robot_name="robot_r1",
                obs_modalities=["proprio", "rgb"],
                proprio_obs=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
                controller_overrides=self.cfg.robot.controllers,
            )
        ]
        # Replace (not merge) gripper configs with MultiFingerGripperController so the
        # websocket policy's 23-dim actions are accepted.  We do this AFTER
        # build_r1pro_primitives_robot_config so that leftover JointController-only
        # kwargs (use_delta_commands, use_impedances) are wiped out entirely.
        robot_cfg = cfg["robots"][0]
        for gripper_key in ("gripper_left", "gripper_right"):
            robot_cfg["controller_config"][gripper_key] = {
                "name": "MultiFingerGripperController",
                "motor_type": "position",
                "command_output_limits": "default",
                "mode": "binary",
                "limit_tolerance": 0.001,
            }
        if self.cfg.max_steps is None:
            logger.info(
                f"Setting timeout to be 2x the average length of human demos: {int(self.human_stats['length'] * 2)}"
            )
            cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
        else:
            logger.info(f"Setting timeout to be {self.cfg.max_steps} steps through config.")
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
        cfg["task"]["include_obs"] = False
        env = og.Environment(configs=cfg)
        # instantiate env wrapper
        env = instantiate(env_wrapper, env=env)
        return env

    def load_robot(self) -> Robot:
        """
        Loads and returns the robot instance from the environment.
        Returns:
            Robot: The robot instance loaded from the environment.
        """
        robot = self.env.scene.object_registry("name", "robot_r1")
        return robot

    def load_policy(self) -> Any:
        """
        Loads and returns the policy instance.
        """
        policy = instantiate(self.cfg.model)
        if hasattr(policy, "setup"):
            policy.setup(env=self.env, robot=self.robot, task_name=self.cfg.task.name)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded policy: {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def load_metrics(self) -> List[MetricBase]:
        """
        Load agent and task metrics.
        """
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]

    def _load_expert_policy(self) -> TaskPrimitivesExpertPolicy:
        """Instantiate and set up the TaskPrimitivesExpertPolicy used as fallback on failure."""
        expert = TaskPrimitivesExpertPolicy(**EXPERT_DEFAULTS)
        expert.setup(env=self.env, robot=self.robot, task_name=self.cfg.task.name)
        logger.info("TaskPrimitivesExpertPolicy loaded as fallback for failure recovery.")
        return expert

    def step(self) -> Tuple[bool, bool]:
        """
        Performs a single step with failure-condition fallback.

        When the primary policy is active, it generates actions normally.
        If the episode terminates due to timeout (done=True, success=False),
        the expert policy (TaskPrimitivesExpertPolicy) takes over:
          - The timeout budget is doubled so the expert has room to act.
          - The episode done/success flags are cleared so env.step() continues.
          - The expert generates actions until the task succeeds or the
            extended timeout fires.

        Returns:
            Tuple[bool, bool]:
                - terminated (bool): Whether the episode has truly terminated.
                - truncated (bool): Whether the episode was truncated.
        """
        # =====================================================================
        # Expert-policy branch: already switched after a prior timeout
        # =====================================================================
        if self._use_expert:
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            logger.info("Using expert policy...")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            action = self.expert_policy.forward(obs=self.obs)
            env_action = (
                self.expert_policy.to_env_action(action)
                if hasattr(self.expert_policy, "to_env_action")
                else action
            )
            obs, _, terminated, truncated, info = self.env.step(env_action, n_render_iterations=1)
            self.obs = self._preprocess_obs(obs)

            if terminated or truncated:
                self.n_trials += 1
                if info["done"]["success"]:
                    self.n_success_trials += 1

            for metric in self.metrics:
                metric.step_callback(self.env)
            return terminated, truncated

        # =====================================================================
        # Primary-policy branch
        # =====================================================================
        self.robot_action = self.policy.forward(obs=self.obs)
        env_action = (
            self.policy.to_env_action(self.robot_action)
            if hasattr(self.policy, "to_env_action")
            else self.robot_action
        )
        obs, _, terminated, truncated, info = self.env.step(env_action, n_render_iterations=1)
        self.obs = self._preprocess_obs(obs)

        if terminated or truncated:
            if info["done"]["success"]:
                # Primary policy succeeded — count and finish normally.
                self.n_trials += 1
                self.n_success_trials += 1
                for metric in self.metrics:
                    metric.step_callback(self.env)
                return terminated, truncated

            # ------------------------------------------------------------------
            # Timeout failure detected — hand over to TaskPrimitivesExpertPolicy
            # ------------------------------------------------------------------
            logger.info(
                "Primary policy timed out at step %d — switching to TaskPrimitivesExpertPolicy.",
                self.env._current_step,
            )
            self._use_expert = True

            # Extend the timeout so the expert has a full budget to work with.
            timeout_cond = self.env.task._termination_conditions["timeout"]
            original_max = timeout_cond._max_steps
            timeout_cond._max_steps = original_max * 2
            logger.info("Extended timeout from %d to %d steps.", original_max, timeout_cond._max_steps)

            # Clear the done/success flags so env.step() keeps working.
            self.env.task._done = False
            self.env.task._success = False

            # Reset the expert policy state for a fresh run.
            self.expert_policy.reset()

            # Do NOT count as a finished trial yet — the expert continues.
            for metric in self.metrics:
                metric.step_callback(self.env)
            return False, False

        # Normal mid-episode step (not done yet).
        for metric in self.metrics:
            metric.step_callback(self.env)
        return terminated, truncated

    @property
    def video_writer(self) -> Tuple[Container, Stream]:
        """
        Returns the video writer for the current evaluation step.
        """
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: Tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            (container, stream) = self._video_writer
            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)
            # Close the container
            container.close()
        self._video_writer = video_writer

    def load_task_instance(self, instance_id: int, test_hidden: bool = False) -> None:
        """
        Loads the configuration for a specific task instance.

        Args:
            instance_id (int): The ID of the task instance to load.
            test_hidden (bool): [Interal use only] Whether to load the hidden test instance.
        """
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        if test_hidden:
            tro_file_path = os.path.join(
                gm.DATA_PATH,
                "2025-challenge-test-instances",
                self.env.task.activity_name,
                f"{tro_filename}-tro_state.json",
            )
        else:
            tro_file_path = os.path.join(
                get_task_instance_path(
                    scene_model,
                    f"{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state",
                )
            )
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                robot_poses = resolve_presampled_robot_pose(tro_state, self.robot.model)
                robot_pos = robot_poses[0]["position"]
                robot_quat = robot_poses[0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)
                # Write robot poses to scene metadata
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        # Try to ensure that all task-relevant objects are stable
        # They should already be stable from the sampled instance, but there is some issue where loading the state
        # causes some jitter (maybe for small mass / thin objects?)
        for _ in range(25):
            og.sim.step_physics()
            for inst, entity in self.env.task.object_scope.items():
                if not is_system_bddl_inst(inst) and entity is not None:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()

    def _preprocess_obs(self, obs: dict) -> dict:
        """
        Preprocess the observation dictionary before passing it to the policy.
        Args:
            obs (dict): The observation dictionary to preprocess.

        Returns:
            dict: The preprocessed observation dictionary.
        """
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        # The first time we query for camera parameters, it will return all zeros
        # For this case, we use camera.get_position_orientation() instead.
        # The reason we are not using camera.get_position_orientation() by defualt is because it will always return the most recent camera poses
        # However, since og render is somewhat "async", it takes >= 3 render calls per step to actually get the up-to-date camera renderings
        # Since we are using n_render_iterations=1 for speed concern, we need the correct corresponding camera poses instead of the most update-to-date one.
        # Thus, we use camera parameters which are guaranteed to be in sync with the visual observations.
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = self.robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        # append task id to obs
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        return obs

    def _write_video(self) -> None:
        """
        Write the current robot observations to video.
        """
        if ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb" not in self.obs:
            return
        # concatenate obs
        left_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        head_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy(),
            (448, 448),
        )
        write_video(
            np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def reset(self) -> None:
        """
        Reset the environment, policy, expert policy state, and compute metrics.
        """
        # Restore the original timeout before env reset (it may have been doubled).
        self.env.task._termination_conditions["timeout"]._max_steps = self._original_max_steps
        self._use_expert = False
        self.expert_policy.reset()

        self.obs = self._preprocess_obs(self.env.reset()[0])
        # run metric start callbacks
        for metric in self.metrics:
            metric.start_callback(self.env)
        self.policy.reset()
        self.n_success_trials, self.n_trials = 0, 0

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {self.n_success_trials}")
        logger.info(f"Total trials: {self.n_trials}")
        if self.n_trials > 0:
            logger.info(f"Success rate: {self.n_success_trials / self.n_trials}")
        logger.info("=" * 50)
        logger.info("")
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        self.video_writer = None
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


if __name__ == "__main__":
    register_omegaconf_resolvers()
    # open yaml from task path
    with hydra.initialize_config_dir(f"{Path(getsourcefile(lambda: 0)).parents[0]}/configs", version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)
    # set headless mode
    gm.HEADLESS = config.headless
    # set video path
    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)
    assert not (
        config.eval_on_train_instances and config.test_hidden
    ), "Cannot eval on train instances and test hidden instances simultaneously."
    if config.test_hidden:
        logger.info("You are evaluating on hidden test instances! This is for internal use only.")
    # get run instances
    if config.eval_on_train_instances:
        logger.info(
            "You are evaluating on training instances, set eval_on_train_instances to False for test instances."
        )
        task_idx = TASK_NAMES_TO_INDICES[config.task.name]
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        instances_to_run = []
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                instances_to_run.append(str(int((episode["episode_index"] // 10) % 1e3)))
        if config.eval_instance_ids:
            assert set(config.eval_instance_ids).issubset(
                set(range(m.NUM_TRAIN_INSTANCES))
            ), f"eval instance ids must be in range({m.NUM_TRAIN_INSTANCES})"
            instances_to_run = [instances_to_run[i] for i in config.eval_instance_ids]
    elif config.test_hidden:
        instances_to_run = (
            config.eval_instance_ids if config.eval_instance_ids is not None else set(range(m.NUM_EVAL_INSTANCES))
        )
        assert set(instances_to_run).issubset(
            set(range(m.NUM_EVAL_INSTANCES))
        ), f"eval instance ids must be in range({m.NUM_EVAL_INSTANCES})"
    else:
        instances_to_run = (
            config.eval_instance_ids if config.eval_instance_ids is not None else set(range(m.NUM_EVAL_INSTANCES))
        )
        assert set(instances_to_run).issubset(
            set(range(m.NUM_EVAL_INSTANCES))
        ), f"eval instance ids must be in range({m.NUM_EVAL_INSTANCES})"
        # load csv file
        task_instance_csv_path = os.path.join(
            gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
        )
        with open(task_instance_csv_path, "r") as f:
            lines = list(csv.reader(f))[1:]
        assert (
            lines[TASK_NAMES_TO_INDICES[config.task.name]][1] == config.task.name
        ), f"Task name from config {config.task.name} does not match task name from csv {lines[TASK_NAMES_TO_INDICES[config.task.name]][1]}"
        test_instances = lines[TASK_NAMES_TO_INDICES[config.task.name]][2].strip().split(",")
        instances_to_run = [int(test_instances[i]) for i in instances_to_run]
    # establish metrics
    metrics = {}
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    with Evaluator(config) as evaluator:
        logger.info("Starting evaluation with failure-condition fallback...")

        for idx in instances_to_run:
            evaluator.reset()
            evaluator.load_task_instance(idx, test_hidden=config.test_hidden)
            set_viewer_camera_to_robot(evaluator.env)
            logger.info(f"Starting task instance {idx} for evaluation...")
            for epi in range(m.NUM_EVAL_EPISODES):
                evaluator.reset()
                done = False
                if config.write_video:
                    video_name = str(video_path) + f"/{config.task.name}_{idx}_{epi}.mp4"
                    evaluator.video_writer = create_video_writer(
                        fpath=video_name,
                        resolution=(448, 672),
                    )
                # run metric start callbacks
                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)
                while not done:
                    terminated, truncated = evaluator.step()
                    if terminated or truncated:
                        done = True
                    if config.write_video:
                        evaluator._write_video()
                    if evaluator.env._current_step % 1000 == 0:
                        logger.info(f"Current step: {evaluator.env._current_step}")
                # run metric end callbacks
                for metric in evaluator.metrics:
                    metric.end_callback(evaluator.env)
                logger.info(f"Evaluation finished at step {evaluator.env._current_step}.")
                logger.info(f"Evaluation exit state: terminated={terminated}, truncated={truncated}")
                logger.info(f"Used expert fallback: {evaluator._use_expert}")
                logger.info(f"Total trials: {evaluator.n_trials}")
                logger.info(f"Total success trials: {evaluator.n_success_trials}")
                # gather metric results and write to file
                for metric in evaluator.metrics:
                    metrics.update(metric.gather_results())
                with open(metrics_path / f"{config.task.name}_{idx}_{epi}.json", "w") as f:
                    json.dump(metrics, f)
                # reset video writer
                if config.write_video:
                    evaluator.video_writer = None
                    logger.info(f"Saved video to {video_name}")
                else:
                    logger.warning("No observations were recorded.")
