"""Shared runtime objects passed through task primitive execution."""

from dataclasses import dataclass, field
from typing import Any

from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.learning.task_primitives.bddl_action_primitives import BDDLSemanticActionPrimitives

from omnigibson.learning.task_primitives.models import DEFAULT_MAX_SAMPLES
from omnigibson.learning.task_primitives.observers import ConsoleObserver, NullObserver


@dataclass
class GraspExecutionConfig:
    """User-facing knobs that affect primitive execution and debugging."""
    max_samples: int = DEFAULT_MAX_SAMPLES
    primitive_attempts: int = 5
    verbose: bool = True
    visualize: bool = False


@dataclass
class ActionContext:
    """Bundle the env, robot, controller and observer into one object."""
    env: Any
    scene: Any
    robot: Any
    controller: StarterSemanticActionPrimitives
    config: GraspExecutionConfig = field(default_factory=GraspExecutionConfig)
    observer: Any = field(default_factory=NullObserver)

    def step(self, action) -> None:
        self.env.step(action)


def create_action_context(
    env,
    config: GraspExecutionConfig | None = None,
    enable_head_tracking: bool = False,
    curobo_batch_size: int = 1,
) -> ActionContext:
    """Construct the primitive controller once and share it across all task steps."""
    robot = env.robots[0]
    scene = env.scene
    controller = BDDLSemanticActionPrimitives(
        env,
        robot,
        enable_head_tracking=enable_head_tracking,
        curobo_batch_size=curobo_batch_size,
    )
    # Silence debug output completely unless the caller explicitly asks for it.
    observer = ConsoleObserver() if (config or GraspExecutionConfig()).verbose else NullObserver()
    return ActionContext(
        env=env,
        scene=scene,
        robot=robot,
        controller=controller,
        config=config or GraspExecutionConfig(),
        observer=observer,
    )