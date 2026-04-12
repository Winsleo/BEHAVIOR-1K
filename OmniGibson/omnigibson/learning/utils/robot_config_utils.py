"""Helpers for building the eval/runtime robot config from a single source of truth."""

from pathlib import Path
from typing import Any, Mapping, Optional

from omegaconf import OmegaConf

from gello.utils.og_teleop_cfg import DEFAULT_TRUNK_TRANSLATE, ROBOT_CONFIGS
from gello.utils.og_teleop_utils import infer_torso_qpos_from_trunk_translate


def resolve_presampled_robot_pose(presampled_robot_poses: Mapping[str, Any], robot_model: str) -> Any:
    """Resolve a robot pose entry even when the task metadata uses a different model key casing."""
    if robot_model in presampled_robot_poses:
        return presampled_robot_poses[robot_model]

    normalized_poses = {key.lower(): value for key, value in presampled_robot_poses.items()}
    normalized_model = robot_model.lower()
    if normalized_model in normalized_poses:
        return normalized_poses[normalized_model]

    aliases = {
        "r1pro": ("r1pro", "r1_pro", "r1-pro"),
        "r1": ("r1",),
        "fetch": ("fetch",),
        "stretch": ("stretch",),
        "tiago": ("tiago",),
    }
    for alias in aliases.get(normalized_model, (normalized_model,)):
        if alias in normalized_poses:
            return normalized_poses[alias]

    available_models = ", ".join(sorted(presampled_robot_poses.keys()))
    raise KeyError(
        f"Presampled robot pose for model '{robot_model}' was not found. "
        f"Available robot_poses keys: {available_models}"
    )


def _merge_controller_overrides(robot_config: dict[str, Any], controller_overrides: Mapping[str, Any]) -> None:
    """Apply per-controller overrides without discarding the base primitives config."""
    controller_config = robot_config.setdefault("controller_config", {})
    for controller_name, override in controller_overrides.items():
        existing = controller_config.get(controller_name)
        if isinstance(existing, dict) and isinstance(override, Mapping):
            existing.update(dict(override))
        else:
            controller_config[controller_name] = dict(override) if isinstance(override, Mapping) else override


def build_r1pro_primitives_robot_config(
    task_cfg: Mapping[str, Any],
    *,
    robot_name: str = "robot_r1",
    obs_modalities: Optional[list[str]] = None,
    proprio_obs: Optional[list[str]] = None,
    controller_overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the R1Pro config used by both eval and task_primitives runtime.

    The base config comes from the primitives yaml, while the reset pose is aligned with
    the joylo teleop defaults so both entrypoints start from the same posture.
    """
    robot_cfg_path = Path(__file__).resolve().parents[2] / "configs" / "r1pro_primitives.yaml"
    robot_doc = OmegaConf.load(robot_cfg_path)
    robot_config = OmegaConf.to_container(robot_doc.robots[0], resolve=True)

    robot_type = str(robot_config.get("type", "R1Pro")).lower()
    teleop_config = ROBOT_CONFIGS[robot_type]
    joint_pos = teleop_config.reset_joint_pos.clone()
    # Force both grippers open so grasp planning starts from a consistent state.
    n_fingers = teleop_config.finger_joints_per_arm * 2
    joint_pos[-n_fingers:] = 0.05
    trunk_start = 6
    trunk_end = trunk_start + teleop_config.torso_joint_count
    # Rebuild the trunk pose from the shared teleop default instead of trusting stale yaml values.
    joint_pos[trunk_start:trunk_end] = infer_torso_qpos_from_trunk_translate(DEFAULT_TRUNK_TRANSLATE, teleop_config)

    robot_config["name"] = robot_name
    robot_config["position"] = task_cfg["robot_start_position"]
    robot_config["orientation"] = task_cfg["robot_start_orientation"]
    robot_config["reset_joint_pos"] = joint_pos

    if obs_modalities is not None:
        robot_config["obs_modalities"] = list(obs_modalities)
    if proprio_obs is not None:
        robot_config["proprio_obs"] = list(proprio_obs)
    if controller_overrides:
        _merge_controller_overrides(robot_config, controller_overrides)

    return robot_config