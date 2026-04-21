"""Task objects that compose backends into eval-friendly execution units."""

import logging
from dataclasses import dataclass
from typing import Generator, List, Protocol

from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, ActionPrimitiveErrorGroup
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitiveSet
from omnigibson.learning.task_primitives.bddl_task_planner import PrimitiveStep
from omnigibson.learning.task_primitives.context import ActionContext
from omnigibson.learning.task_primitives.grasp_primitives import execute_controller
from omnigibson.learning.task_primitives.models import GraspResult, PrimitiveExecutionResult
from omnigibson.learning.task_primitives.runtime import resolve_scene_object

log = logging.getLogger(__name__)


class BaseTask(Protocol):
    """Small protocol so policies can treat different primitive tasks uniformly."""
    def iter_actions(self, context: ActionContext) -> Generator:
        pass

    def run(self, context: ActionContext) -> PrimitiveExecutionResult:
        pass


# ── Action classification ──────────────────────────────────────────────────

_GRASP_ACTIONS = {StarterSemanticActionPrimitiveSet.GRASP}

# Primitives handled by the controller's apply_ref (everything except GRASP)
_CONTROLLER_ACTIONS = {
    StarterSemanticActionPrimitiveSet.PLACE_ON_TOP,
    StarterSemanticActionPrimitiveSet.PLACE_INSIDE,
    StarterSemanticActionPrimitiveSet.OPEN,
    StarterSemanticActionPrimitiveSet.CLOSE,
    StarterSemanticActionPrimitiveSet.TOGGLE_ON,
    StarterSemanticActionPrimitiveSet.TOGGLE_OFF,
    StarterSemanticActionPrimitiveSet.NAVIGATE_TO,
    StarterSemanticActionPrimitiveSet.RELEASE,
}


@dataclass
class BDDLSequenceTask(BaseTask):
    """Execute an ordered sequence of primitive steps derived from BDDL goal conditions.

    The steps are produced by ``bddl_task_planner.plan_from_goal`` and may
    contain GRASP, PLACE_INSIDE, PLACE_ON_TOP, OPEN, CLOSE, TOGGLE_ON,
    TOGGLE_OFF, etc.  GRASP steps are delegated to the grasp backend;
    all other steps go through the controller's ``apply_ref``.
    """
    steps: List[PrimitiveStep]
    backend: object

    def iter_actions(self, context: ActionContext) -> Generator:
        for i, step in enumerate(self.steps):
            obj = resolve_scene_object(context.scene, step.object_name, task=context.env.task)
            log.info(f"Step {i+1}/{len(self.steps)}: {step.action.name}({step.object_name})")

            try:
                if step.action in _GRASP_ACTIONS:
                    yield from self.backend.iter_actions(context, obj)
                elif step.action == StarterSemanticActionPrimitiveSet.RELEASE:
                    yield from context.controller.apply_ref(step.action)
                elif step.action in _CONTROLLER_ACTIONS:
                    yield from context.controller.apply_ref(step.action, obj)
                else:
                    log.warning(f"Unknown action {step.action}, skipping")
            except NotImplementedError as exc:
                log.warning(
                    f"Step {i+1}/{len(self.steps)} {step.action.name}({step.object_name}) "
                    f"is not implemented — stopping early. Completed {i}/{len(self.steps)} steps. ({exc})"
                )
                return

    def run(self, context: ActionContext) -> PrimitiveExecutionResult:
        for i, step in enumerate(self.steps):
            obj = resolve_scene_object(context.scene, step.object_name, task=context.env.task)
            log.info(f"Step {i+1}/{len(self.steps)}: {step.action.name}({step.object_name})")

            try:
                if step.action in _GRASP_ACTIONS:
                    result = self.backend.grasp(context, obj)
                    if not result.success:
                        return result
                elif step.action == StarterSemanticActionPrimitiveSet.RELEASE:
                    execute_controller(context.controller.apply_ref(step.action), context.env)
                elif step.action in _CONTROLLER_ACTIONS:
                    execute_controller(context.controller.apply_ref(step.action, obj), context.env)
                else:
                    log.warning(f"Unknown action {step.action}, skipping")
            except NotImplementedError as exc:
                log.warning(
                    f"Step {i+1}/{len(self.steps)} {step.action.name}({step.object_name}) "
                    f"is not implemented — stopping early. Completed {i}/{len(self.steps)} steps. ({exc})"
                )
                return PrimitiveExecutionResult(
                    success=False,
                    backend="bddl_sequence",
                    result=GraspResult.GRASP_FAILED,
                    error_message=f"Step {i+1} ({step.action.name}) not implemented: {exc}",
                    metadata={"completed_steps": i, "total_steps": len(self.steps), "action": step.action.name, "object": step.object_name},
                )
            except (ActionPrimitiveError, ActionPrimitiveErrorGroup) as exc:
                return PrimitiveExecutionResult(
                    success=False,
                    backend="bddl_sequence",
                    result=GraspResult.GRASP_FAILED,
                    error_message=f"Step {i+1} ({step.action.name}) failed: {exc}",
                    metadata={"failed_step": i, "action": step.action.name, "object": step.object_name},
                )

        return PrimitiveExecutionResult(
            success=True,
            backend="bddl_sequence",
            result=GraspResult.SUCCESS,
            metadata={"total_steps": len(self.steps)},
        )
