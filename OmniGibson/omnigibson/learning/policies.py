import logging
import torch as th
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, ActionPrimitiveErrorGroup
from omnigibson.learning.task_primitives import (
    GraspExecutionConfig,
    UnifiedGraspBackend,
    create_action_context,
)
from omnigibson.learning.task_primitives.bddl_task_planner import plan_from_goal
from omnigibson.learning.task_primitives.tasks import BDDLSequenceTask
from omnigibson.learning.utils.array_tensor_utils import torch_to_numpy
from omnigibson.learning.utils.action_adapter import VelocityActionAdapter
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
from typing import Optional


__all__ = [
    "LocalPolicy",
    "TaskPrimitivesExpertPolicy",
    "WebsocketPolicy",
]


def _make_grasp_backend(grasp_mode: str):
    """Return the unified backend.

    ``grasp_mode`` is accepted for backwards compatibility with older configs
    (``custom`` / ``primitive`` / ``both``) but ignored — the unified backend
    already contains the explicit pipeline and the apply_ref fallback, so every
    mode resolves to the same implementation.
    """
    return UnifiedGraspBackend()


class LocalPolicy:
    """
    Local policy that directly queries action from policy,
        outputs zero delta action if policy is None.
    """

    def __init__(self, *args, action_dim: Optional[int] = None, **kwargs) -> None:
        self.policy = None  # To be set later
        self.action_dim = action_dim

    def act(self, obs: dict) -> th.Tensor:
        return self.forward(obs)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        """
        Directly return a zero action tensor of the specified action dimension.
        """
        if self.policy is not None:
            return self.policy.act(obs).detach().cpu()
        else:
            assert self.action_dim is not None
            return th.zeros(self.action_dim, dtype=th.float32)

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()


class WebsocketPolicy:
    """
    Websocket policy for controlling the robot over a websocket connection.
    """

    def __init__(
        self,
        *args,
        host: Optional[str] = None,
        port: Optional[int] = None,
        allow_reconnect: bool = False,
        **kwargs,
    ) -> None:
        logging.info(f"Creating websocket client policy with host: {host}, port: {port}")
        self.last_action = None
        self.policy = None
        self._allow_reconnect = allow_reconnect
        if host is not None or port is not None:
            self.policy = WebsocketClientPolicy(host=host, port=port, allow_reconnect=allow_reconnect)

    def update_host(self, host: str, port: int) -> None:
        self.policy = WebsocketClientPolicy(host=host, port=port, allow_reconnect=self._allow_reconnect)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        if "need_new_action" in obs and not obs["need_new_action"] and self.last_action is not None:
            return self.last_action
        self.last_action = self.policy.act(obs).detach().cpu()
        return self.last_action

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        self.last_action = None


class TaskPrimitivesExpertPolicy:
    """Run a task primitive as a step-wise policy that eval.py can consume."""

    def __init__(
        self,
        *args,
        action_dim: Optional[int] = None,
        grasp_mode: str = "custom",
        max_samples: int = 50,
        primitive_attempts: int = 5,
        verbose: bool = False,
        visualize: bool = False,
        action_mode: str = "position",
        **kwargs,
    ) -> None:
        self.action_dim = action_dim
        self.grasp_mode = grasp_mode
        self.max_samples = max_samples
        self.primitive_attempts = primitive_attempts
        self.verbose = verbose
        self.visualize = visualize
        if action_mode not in {"position", "velocity"}:
            raise ValueError(f"Unsupported action_mode: {action_mode}")
        self.action_mode = action_mode
        self.context = None
        self.task_name = None
        self._backend = None
        self._task = None
        self._action_iter = None
        self._last_action = None
        self._last_position_action = None
        self._completed = False
        self._failed = False
        self._action_adapter = None
        self._robot = None

    def setup(self, env, robot=None, task_name: Optional[str] = None) -> None:
        """Bind the policy to the current env and build the execution context once."""
        self._robot = robot or env.robots[0]
        config = GraspExecutionConfig(
            max_samples=self.max_samples,
            primitive_attempts=self.primitive_attempts,
            verbose=self.verbose,
            visualize=self.visualize,
        )
        self.context = create_action_context(env=env, config=config, enable_head_tracking=False, curobo_batch_size=1)
        self.task_name = task_name or getattr(env.task, "activity_name", None)
        self._backend = _make_grasp_backend(self.grasp_mode)
        # Always trust the instantiated robot over a stale config value.
        self.action_dim = self._robot.action_dim
        if self.action_mode == "velocity":
            # Keep primitives running in position space internally, but expose velocity-style actions outside.
            self._action_adapter = VelocityActionAdapter(self._robot)
        self.reset()

    def _zero_action(self) -> th.Tensor:
        assert self.action_dim is not None, "TaskPrimitivesExpertPolicy requires action_dim"
        return th.zeros(self.action_dim, dtype=th.float32)

    def _normalize_action_dim(self, action: th.Tensor) -> th.Tensor:
        if self.action_dim is None or len(action) == self.action_dim:
            return action
        if len(action) > self.action_dim:
            return action[: self.action_dim]
        normalized = self._zero_action()
        normalized[: len(action)] = action
        return normalized

    def _build_task(self):
        """Auto-decompose the task from BDDL goal conditions."""
        behavior_task = getattr(self.context.env, "task", None)
        if behavior_task is None or not hasattr(behavior_task, "compiled_task"):
            raise ValueError(
                f"Task '{self.task_name}' environment has no BehaviorTask with compiled BDDL "
                "conditions. Ensure initialize_activity() was called."
            )
        steps = plan_from_goal(behavior_task)
        if not steps:
            raise ValueError(
                f"BDDL decomposition for task '{self.task_name}' produced no actionable steps. "
                "The task's goal predicates may not be supported by the current primitive set."
            )
        logging.info(f"Auto-decomposed '{self.task_name}' into {len(steps)} primitive steps from BDDL goals")
        return BDDLSequenceTask(steps=steps, backend=self._backend)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        if self.context is None:
            raise RuntimeError("TaskPrimitivesExpertPolicy.setup(env, robot, task_name) must be called before forward().")
        if "need_new_action" in obs and not obs["need_new_action"] and self._last_action is not None:
            return self._last_action
        if self._completed or self._failed:
            self._last_position_action = self._zero_action()
            self._last_action = self._zero_action()
            return self._last_action
        if self._action_iter is None:
            self._task = self._build_task()
            self._action_iter = self._task.iter_actions(self.context)
        try:
            self._last_position_action = self._normalize_action_dim(next(self._action_iter).detach().cpu())
            self._last_action = self._encode_action(self._last_position_action)
        except StopIteration:
            self._completed = True
            self._last_position_action = self._zero_action()
            self._last_action = self._zero_action()
        except ActionPrimitiveErrorGroup as exc:
            self.context.observer.report_primitive_failure(exc)
            self._failed = True
            self._last_position_action = self._zero_action()
            self._last_action = self._zero_action()
        except (ActionPrimitiveError, ValueError) as exc:
            self.context.observer.log(str(exc))
            self._failed = True
            self._last_position_action = self._zero_action()
            self._last_action = self._zero_action()
        return self._last_action

    def _encode_action(self, position_action: th.Tensor) -> th.Tensor:
        position_action = self._normalize_action_dim(position_action)
        if self._action_adapter is None:
            return position_action
        return self._normalize_action_dim(self._action_adapter.to_velocity_action(position_action))

    def to_env_action(self, action: th.Tensor) -> th.Tensor:
        """Translate the externally visible action back to the env command space."""
        if self._action_adapter is None:
            return self._normalize_action_dim(action)
        if self._last_position_action is None:
            return self._normalize_action_dim(self._action_adapter.to_position_action(action))
        # Reuse the exact action produced by the primitive to avoid integration drift across repeated frames.
        return self._normalize_action_dim(self._last_position_action)

    def reset(self) -> None:
        self._task = None
        self._action_iter = None
        self._last_action = None
        self._last_position_action = None
        self._completed = False
        self._failed = False
