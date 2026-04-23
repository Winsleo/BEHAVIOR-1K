"""Unified grasp backend — one class, all strengths.

Two complementary layers run per grasp invocation:

    1. Explicit pipeline — sample base pose, execute the cached validated
       trajectory, move hand to pre-grasp, release, move to grasp, close
       fingers, restore trunk.  Retried ``explicit_attempts`` times.
    2. Fallback — OmniGibson's built-in GRASP primitive via ``apply_ref``,
       retried ``fallback_attempts`` times if the explicit pipeline is
       exhausted.

Extension points (``sample_base_pose``, ``execute_navigation``,
``execute_grasp_motion``, ``execute_restore``) let subclasses specialise
behaviour without rewriting the control flow in ``_run_attempt``.
"""

from abc import ABC, abstractmethod
from typing import Generator, List, Optional

from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, ActionPrimitiveErrorGroup
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitiveSet
from omnigibson.learning.task_primitives.grasp_primitives import (
    execute_controller,
    find_optimal_grasp_base_pose,
    iterate_navigation_trajectory,
    iterate_restore_trunk_actions,
    plan_navigation,
)
from omnigibson.learning.task_primitives.models import (
    GraspBasePoseResult,
    GraspResult,
    PrimitiveExecutionResult,
)


class GraspBackend(ABC):
    """Abstract backend — a grasp strategy callable from ``BDDLSequenceTask``."""

    name = "base"

    @abstractmethod
    def iter_actions(self, context, obj) -> Generator:
        pass

    def grasp(self, context, obj) -> PrimitiveExecutionResult:
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


class UnifiedGraspBackend(GraspBackend):
    """Single backend that merges the explicit pipeline with a primitive fallback.

    Per grasp invocation:

        for attempt in range(explicit_attempts):
            1. Sample base pose + cache its validated navigation trajectory
            2. Execute the cached trajectory (no re-plan)
            3. Move hand to pre-grasp, release, move to grasp, close fingers
            4. Restore trunk (if articulated)
            on ActionPrimitiveError(Group): capture, continue to next attempt

        If every explicit attempt failed, delegate to:
            controller.apply_ref(GRASP, obj, attempts=fallback_attempts)

    Both counts default to ``context.config.primitive_attempts`` (CLI knob),
    split evenly between the two layers.  Callers can override via the
    constructor.
    """

    name = "unified"

    def __init__(
        self,
        explicit_attempts: Optional[int] = None,
        fallback_attempts: Optional[int] = None,
        enable_fallback: bool = True,
    ):
        # Explicit retry count for the custom pipeline.  None means "pull from
        # context.config.primitive_attempts at runtime" so one backend instance
        # can serve multiple configurations.
        self._explicit_attempts = explicit_attempts
        self._fallback_attempts = fallback_attempts
        self._enable_fallback = enable_fallback

    # ------------------------------------------------------------------
    # Extension hooks — override in subclasses, not the control flow.
    # ------------------------------------------------------------------

    def sample_base_pose(self, context, obj) -> GraspBasePoseResult:
        """Run the sampling pipeline.  Returns pose + validated q_traj."""
        return find_optimal_grasp_base_pose(
            controller=context.controller,
            obj=obj,
            max_samples=context.config.max_samples,
            observer=context.observer,
            verbose=context.config.verbose,
            visualize=context.config.visualize,
        )

    def ensure_navigation_trajectory(self, context, result: GraspBasePoseResult):
        """Return a q_traj for executing navigation.

        Prefers the one cached inside ``result`` (planned during sampling) but
        falls back to planning from scratch if the sampler didn't keep one.
        """
        if result.q_traj is not None:
            return result.q_traj
        return plan_navigation(
            controller=context.controller,
            robot=context.robot,
            target_pose_2d=result.base_pose_2d,
            observer=context.observer,
            verbose=context.config.verbose,
            visualize=context.config.visualize,
        )

    def execute_navigation(self, context, q_traj) -> Generator:
        if context.config.verbose:
            context.observer.section(f"[Step 3/6] Executing navigation ({len(q_traj)} waypoints)")
        yield from iterate_navigation_trajectory(context, q_traj)
        if context.config.verbose:
            context.observer.log("[OK] Navigation complete")

    def execute_grasp_motion(self, context, obj, pregrasp_pose, grasp_pose) -> Generator:
        if context.config.verbose:
            context.observer.section("[Step 4/6] Moving hand to pre-grasp position")
        yield from context.controller._move_hand(pregrasp_pose)
        if context.config.verbose:
            context.observer.log("[OK] Hand at pre-grasp position")
            context.observer.section("[Step 5/6] Executing grasp")
        yield from context.controller._execute_release()
        yield from context.controller._move_hand(grasp_pose, ignore_objects=[obj], low_precision=True)
        yield from context.controller._execute_grasp()

    def execute_restore(self, context) -> Generator:
        if not context.robot.is_articulated_trunk:
            return
        if context.config.verbose:
            context.observer.section("[Step 6/6] Restoring trunk to initial pose")
        yield from iterate_restore_trunk_actions(context)
        if context.config.verbose:
            context.observer.log("[OK] Stand up complete")

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def _resolve_attempt_counts(self, context) -> tuple[int, int]:
        budget = max(1, int(context.config.primitive_attempts))
        explicit = self._explicit_attempts if self._explicit_attempts is not None else budget
        if self._enable_fallback:
            fallback = self._fallback_attempts if self._fallback_attempts is not None else budget
        else:
            fallback = 0
        return explicit, fallback

    def _run_attempt(self, context, obj) -> Generator:
        """One full sample→navigate→grasp cycle.  Raises on failure."""
        result = self.sample_base_pose(context, obj)
        if not result.success:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Could not find optimal base pose for grasping",
                result.stats,
            )

        q_traj = self.ensure_navigation_trajectory(context, result)
        if q_traj is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Navigation planning failed",
                result.stats,
            )

        if context.config.visualize:
            context.observer.show_sampled_pose(True, context.robot, result.base_pose_2d, context.config.verbose)
            context.observer.show_trajectory(True, q_traj, context.robot, context.config.verbose)

        yield from self.execute_navigation(context, q_traj)
        yield from self.execute_grasp_motion(context, obj, result.pregrasp_pose, result.grasp_pose)
        yield from self.execute_restore(context)

    def iter_actions(self, context, obj) -> Generator:
        explicit_attempts, fallback_attempts = self._resolve_attempt_counts(context)
        errors: List[str] = []

        # ---- Layer 1: explicit pipeline with retries ----
        for attempt in range(explicit_attempts):
            if context.config.verbose and explicit_attempts > 1:
                context.observer.section(f"[Explicit] Attempt {attempt + 1}/{explicit_attempts}")
            try:
                yield from self._run_attempt(context, obj)
                return
            except (ActionPrimitiveError, ActionPrimitiveErrorGroup) as exc:
                errors.append(str(exc))
                if isinstance(exc, ActionPrimitiveErrorGroup):
                    context.observer.report_primitive_failure(exc)
                else:
                    context.observer.log(f"[Explicit {attempt + 1}/{explicit_attempts}] {exc}")

        # ---- Layer 2: OmniGibson built-in primitive as safety net ----
        if fallback_attempts > 0:
            if context.config.verbose:
                context.observer.section(
                    f"[Fallback] Explicit pipeline exhausted, delegating to apply_ref "
                    f"(attempts={fallback_attempts})"
                )
            try:
                yield from context.controller.apply_ref(
                    StarterSemanticActionPrimitiveSet.GRASP,
                    obj,
                    attempts=fallback_attempts,
                )
                return
            except (ActionPrimitiveError, ActionPrimitiveErrorGroup) as exc:
                errors.append(f"fallback: {exc}")

        # ---- All layers failed ----
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.EXECUTION_ERROR,
            f"Grasp failed after {explicit_attempts} explicit + {fallback_attempts} fallback attempts",
            {"object": obj.name, "errors": errors},
        )

    def grasp(self, context, obj) -> PrimitiveExecutionResult:
        result = super().grasp(context, obj)
        if result.success:
            result.metadata["object_name"] = obj.name
        return result
