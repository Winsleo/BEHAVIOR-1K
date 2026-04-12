"""Adapters for exposing one action semantic externally while executing another internally."""

import torch as th

from omnigibson.controllers import ControllerView, HolonomicBaseJointController


class VelocityActionAdapter:
    """Expose position-controller commands as velocity-style actions and back again."""

    def __init__(self, robot, dt: float | None = None):
        self.robot = robot
        control_freq = getattr(robot, "_control_freq", None)
        self.dt = dt if dt is not None else (1.0 / control_freq if control_freq else 1.0)

    def to_velocity_action(self, action: th.Tensor) -> th.Tensor:
        """Convert a position-style action into a per-step velocity command."""
        action = action.detach().cpu()
        semantic_action = action.clone()
        current_joint_pos = self.robot.get_joint_positions().detach().cpu()

        for controller_name in self.robot.controller_order:
            group_key, _ = self.robot.controllers[controller_name]
            action_idx = self.robot.controller_action_idx[controller_name]
            motor_type = ControllerView.get_motor_type(group_key)
            if motor_type == "velocity":
                continue

            if ControllerView.is_controller_type(group_key, HolonomicBaseJointController):
                # Base position actions already represent a one-step local displacement.
                semantic_action[action_idx] = action[action_idx] / self.dt
                continue

            dof_idx = ControllerView.get_dof_idx(group_key)
            current = current_joint_pos[dof_idx]
            target = action[action_idx]
            if target.numel() == 1 and current.numel() > 1:
                semantic_action[action_idx] = ((target.item() - current.mean().item()) / self.dt)
            elif target.numel() == current.numel():
                semantic_action[action_idx] = (target - current) / self.dt

        return semantic_action

    def to_position_action(self, action: th.Tensor) -> th.Tensor:
        """Integrate a velocity-style action back into the position command expected by the env."""
        action = action.detach().cpu()
        position_action = action.clone()
        current_joint_pos = self.robot.get_joint_positions().detach().cpu()
        position_lower, position_upper = self.robot.control_limits["position"]
        position_lower = position_lower.detach().cpu()
        position_upper = position_upper.detach().cpu()

        for controller_name in self.robot.controller_order:
            group_key, _ = self.robot.controllers[controller_name]
            action_idx = self.robot.controller_action_idx[controller_name]
            motor_type = ControllerView.get_motor_type(group_key)
            if motor_type == "velocity":
                continue

            if ControllerView.is_controller_type(group_key, HolonomicBaseJointController):
                # For holonomic bases, the position-mode command is just the local displacement for this step.
                position_action[action_idx] = action[action_idx] * self.dt
                continue

            dof_idx = ControllerView.get_dof_idx(group_key)
            current = current_joint_pos[dof_idx]
            lower = position_lower[dof_idx]
            upper = position_upper[dof_idx]
            velocity_cmd = action[action_idx]
            if velocity_cmd.numel() == 1 and current.numel() > 1:
                target = (current + velocity_cmd.item() * self.dt).clip(lower, upper)
                position_action[action_idx] = target.mean()
            elif velocity_cmd.numel() == current.numel():
                position_action[action_idx] = (current + velocity_cmd * self.dt).clip(lower, upper)

        return position_action