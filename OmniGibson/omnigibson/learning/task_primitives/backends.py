"""Execution backends that turn a high-level grasp task into low-level actions."""

from abc import ABC, abstractmethod
from typing import Generator

from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, ActionPrimitiveErrorGroup
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitiveSet
from omnigibson.learning.task_primitives.grasp_primitives import (
    execute_controller,
    find_optimal_grasp_base_pose,
    iterate_grasp_actions,
    iterate_navigation_trajectory,
    iterate_restore_trunk_actions,
    plan_navigation,
)
from omnigibson.learning.task_primitives.models import GraspResult, PrimitiveExecutionResult


class GraspBackend(ABC):
    name = "base"

    @abstractmethod
    def iter_actions(self, context, obj) -> Generator:
        pass

    def grasp(self, context, obj) -> PrimitiveExecutionResult:
        # The eager API is still useful for dagger.py and tests; iter_actions stays the shared core.
        try:
            execute_controller(self.iter_actions(context, obj), context.env)
            return PrimitiveExecutionResult(success=True, backend=self.name, result=GraspResult.SUCCESS)
        except ActionPrimitiveError as exc:
            return PrimitiveExecutionResult(
                success=False,
                backend=self.name,
                result=GraspResult.GRASP_FAILED,
                error_message=str(exc),
                metadata=getattr(exc, "metadata", {}),
            )
        except ActionPrimitiveErrorGroup as exc:
            context.observer.report_primitive_failure(exc)
            return PrimitiveExecutionResult(
                success=False,
                backend=self.name,
                result=GraspResult.GRASP_FAILED,
                error_message=str(exc),
                metadata={"attempt_count": len(exc.exceptions)},
            )


class CustomGraspBackend(GraspBackend):
    name = "custom"

    def iter_actions(self, context, obj) -> Generator:
        # This backend keeps the sampling and navigation logic explicit so it is easier to debug.
        result = find_optimal_grasp_base_pose(
            controller=context.controller,
            obj=obj,
            max_samples=context.config.max_samples,
            observer=context.observer,
            verbose=context.config.verbose,
            visualize=context.config.visualize,
        )
        if not result.success:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Could not find optimal base pose for grasping",
                result.stats,
            )

        q_traj = plan_navigation(
            controller=context.controller,
            robot=context.robot,
            target_pose_2d=result.base_pose_2d,
            observer=context.observer,
            verbose=context.config.verbose,
            visualize=context.config.visualize,
        )
        if q_traj is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Navigation planning failed",
                result.stats,
            )

        if context.config.visualize:
            context.observer.show_sampled_pose(True, context.robot, result.base_pose_2d, context.config.verbose)
            context.observer.show_trajectory(True, q_traj, context.robot, context.config.verbose)

        if context.config.verbose:
            context.observer.section(f"[Step 3/5] Executing navigation ({len(q_traj)} waypoints)")
        yield from iterate_navigation_trajectory(context, q_traj)
        if context.config.verbose:
            context.observer.log("[OK] Navigation complete")
            context.observer.section("[Step 4/5] Moving hand to pre-grasp position")
        yield from context.controller._move_hand(result.pregrasp_pose)
        if context.config.verbose:
            context.observer.log("[OK] Hand at pre-grasp position")
            context.observer.section("[Step 5/5] Executing grasp")
        yield from context.controller._execute_release()
        yield from context.controller._move_hand(result.grasp_pose, ignore_objects=[obj], low_precision=True)
        yield from context.controller._execute_grasp()
        if context.robot.is_articulated_trunk:
            if context.config.verbose:
                context.observer.section("[Step 6/6] Restoring trunk to initial pose")
            yield from iterate_restore_trunk_actions(context)
            if context.config.verbose:
                context.observer.log("[OK] Stand up complete")

    def grasp(self, context, obj) -> PrimitiveExecutionResult:
        result = super().grasp(context, obj)
        if result.success:
            result.metadata["object_name"] = obj.name
        return result


class PrimitiveGraspBackend(GraspBackend):
    name = "primitive"

    def iter_actions(self, context, obj) -> Generator:
        # Delegate to OmniGibson's built-in primitive for comparison and fallback.
        yield from context.controller.apply_ref(
            StarterSemanticActionPrimitiveSet.GRASP,
            obj,
            attempts=context.config.primitive_attempts,
        )


class FallbackGraspBackend(GraspBackend):
    name = "fallback"

    def __init__(self, primary: GraspBackend, fallback: GraspBackend):
        self.primary = primary
        self.fallback = fallback

    def iter_actions(self, context, obj) -> Generator:
        # Preserve the streaming interface by only switching backends after a real execution failure.
        try:
            yield from self.primary.iter_actions(context, obj)
        except (ActionPrimitiveError, ActionPrimitiveErrorGroup) as exc:
            if isinstance(exc, ActionPrimitiveErrorGroup):
                context.observer.report_primitive_failure(exc)
            else:
                context.observer.log(str(exc))
            yield from self.fallback.iter_actions(context, obj)

    def grasp(self, context, obj) -> PrimitiveExecutionResult:
        primary_result = self.primary.grasp(context, obj)
        if primary_result.success:
            return primary_result
        fallback_result = self.fallback.grasp(context, obj)
        fallback_result.metadata["primary_backend"] = primary_result.backend
        fallback_result.metadata["primary_result"] = primary_result.result.value
        fallback_result.metadata["primary_error"] = primary_result.error_message
        return fallback_result