"""Task objects that compose backends into eval-friendly execution units."""

from dataclasses import dataclass
from typing import Generator, Protocol

from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, ActionPrimitiveErrorGroup
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitiveSet
from omnigibson.learning.task_primitives.context import ActionContext
from omnigibson.learning.task_primitives.grasp_primitives import execute_controller
from omnigibson.learning.task_primitives.models import GraspResult, PrimitiveExecutionResult
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
        obj = resolve_scene_object(context.scene, self.object_name, task=context.env.task)
        yield from self.backend.iter_actions(context, obj)

    def run(self, context: ActionContext) -> PrimitiveExecutionResult:
        obj = resolve_scene_object(context.scene, self.object_name, task=context.env.task)
        return self.backend.grasp(context, obj)


@dataclass
class GraspAndPlaceInsideTask(BaseTask):
    """Grasp one task object, then place it inside a target receptacle."""
    object_name: str
    target_name: str
    backend: object

    def iter_actions(self, context: ActionContext) -> Generator:
        obj = resolve_scene_object(context.scene, self.object_name, task=context.env.task)
        target = resolve_scene_object(context.scene, self.target_name, task=context.env.task)
        yield from self.backend.iter_actions(context, obj)
        yield from context.controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_INSIDE, target)

    def run(self, context: ActionContext) -> PrimitiveExecutionResult:
        obj = resolve_scene_object(context.scene, self.object_name, task=context.env.task)
        target = resolve_scene_object(context.scene, self.target_name, task=context.env.task)
        grasp_result = self.backend.grasp(context, obj)
        if not grasp_result.success:
            return grasp_result
        try:
            execute_controller(context.controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_INSIDE, target), context.env)
            grasp_result.metadata["target_name"] = target.name
            return grasp_result
        except ActionPrimitiveError as exc:
            return PrimitiveExecutionResult(
                success=False,
                backend=grasp_result.backend,
                result=GraspResult.PLACE_FAILED,
                error_message=str(exc),
                metadata={"target_name": target.name, **getattr(exc, "metadata", {})},
            )
        except ActionPrimitiveErrorGroup as exc:
            context.observer.report_primitive_failure(exc)
            return PrimitiveExecutionResult(
                success=False,
                backend=grasp_result.backend,
                result=GraspResult.PLACE_FAILED,
                error_message=str(exc),
                metadata={"target_name": target.name, "attempt_count": len(exc.exceptions)},
            )
