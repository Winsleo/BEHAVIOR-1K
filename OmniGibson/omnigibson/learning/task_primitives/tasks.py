"""Task objects that compose backends into eval-friendly execution units."""

from dataclasses import dataclass
from typing import Generator, Protocol

from omnigibson.learning.task_primitives.context import ActionContext
from omnigibson.learning.task_primitives.models import PrimitiveExecutionResult
from omnigibson.learning.task_primitives.runtime import resolve_scene_object


class BaseTask(Protocol):
    """Small protocol so policies can treat different primitive tasks uniformly."""
    def iter_actions(self, context: ActionContext) -> Generator:
        pass

    def run(self, context: ActionContext) -> PrimitiveExecutionResult:
        pass


@dataclass
class NavigateAndGraspTask(BaseTask):
    """Resolve one scene object and hand execution off to the selected backend."""
    object_name: str
    backend: object

    def iter_actions(self, context: ActionContext) -> Generator:
        obj = resolve_scene_object(context.scene, self.object_name)
        yield from self.backend.iter_actions(context, obj)

    def run(self, context: ActionContext) -> PrimitiveExecutionResult:
        obj = resolve_scene_object(context.scene, self.object_name)
        return self.backend.grasp(context, obj)