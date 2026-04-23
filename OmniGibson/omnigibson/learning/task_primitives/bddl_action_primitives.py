"""BDDLSemanticActionPrimitives — a StarterSemanticActionPrimitives subclass
tuned for running BDDL task plans end-to-end.

This class layers the task_primitives helpers on top of the upstream primitive
set without modifying it.  Concretely it adds:

    * Arm override — so primitives that need the non-default arm (e.g. the
      pressing hand for TOGGLE_ON while the other hand holds the appliance)
      can drive ``self.arm``-keyed code paths without re-plumbing every helper.
    * ``_navigate_to_obj`` — reuses the navigation trajectory that
      ``find_optimal_base_pose`` already validated, so curobo's stochastic
      planner is invoked exactly once per navigation.  Eliminates a common
      source of flaky ``PLANNING_ERROR`` retries.
    * ``_toggle`` — physical bimanual actuation.  On a bimanual robot the arm
      holding the target is kept still while the other arm presses the
      ``ToggledOn`` visual marker for ``CAN_TOGGLE_STEPS`` frames.  Falls back
      to a symbolic ``set_value`` if motion planning or alignment fails so the
      oracle still produces a usable trajectory for DAgger.
"""

import contextlib

from omnigibson import object_states
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    indented_print,
    m,
)
from omnigibson.learning.task_primitives.grasp_primitives import find_optimal_base_pose
from omnigibson.object_states import toggle as _toggle_module


class BDDLSemanticActionPrimitives(StarterSemanticActionPrimitives):
    """Primitive set specialised for BDDL task execution on bimanual robots."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # When set via _with_arm_override, self.arm returns this value instead
        # of self.robot.default_arm.  Lets _toggle (and future bimanual primitives)
        # drive the non-default arm while the default arm is busy.
        self._arm_override = None

    # ------------------------------------------------------------------
    # Arm selection
    # ------------------------------------------------------------------

    @property
    def arm(self):
        assert self.robot.is_manipulation, "Cannot use arm for non-manipulation robot"
        if self._arm_override is not None:
            return self._arm_override
        return self.robot.default_arm

    @contextlib.contextmanager
    def _with_arm_override(self, arm):
        previous = self._arm_override
        self._arm_override = arm
        try:
            yield
        finally:
            self._arm_override = previous

    # ------------------------------------------------------------------
    # Navigation — single-plan execution
    # ------------------------------------------------------------------

    def _navigate_to_obj(self, obj, eef_pose=None, skip_obstacle_update=False):
        """Navigate to ``obj`` by executing the trajectory validated during sampling.

        ``find_optimal_base_pose`` pre-validates each candidate by planning a
        navigation trajectory.  The upstream implementation throws that
        trajectory away and re-plans from scratch, which occasionally fails
        because curobo's ``compute_trajectories`` is stochastic.  We ask for
        the cached trajectory and execute it directly.
        """
        if eef_pose is None:
            eef_pose, _ = self._sample_grasp_pose(obj)

        pose, q_traj = find_optimal_base_pose(
            controller=self,
            target_eef_pose=eef_pose,
            obj=obj,
            max_samples=60,
            sampling_radius=m.BASE_POSE_SAMPLING_UPPER_BOUND,
            skip_obstacle_update=skip_obstacle_update,
            max_rounds=3,
            return_traj=True,
        )
        if pose is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Could not find a valid pose near the object",
                {"object": obj.name},
            )
        indented_print("Found valid position near object.")

        if q_traj is not None:
            pose_3d = self._get_robot_pose_from_2d_pose(pose)
            if self.debug_visual_marker is not None:
                self.debug_visual_marker.set_position_orientation(*pose_3d)
            yield from self._execute_motion_plan(q_traj)
        else:
            # Extremely unlikely — find_optimal_base_pose only returns a pose
            # after it has a valid q_traj.  Keep the classic path as a guard.
            yield from self._navigate_to_pose(pose, skip_obstacle_update=skip_obstacle_update)

    # ------------------------------------------------------------------
    # Toggle — bimanual physical actuation with symbolic fallback
    # ------------------------------------------------------------------

    def _toggle(self, obj, value):
        if object_states.ToggledOn not in obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not toggleable.",
                {"target object": obj.name},
            )

        toggle_state = obj.states[object_states.ToggledOn]
        if toggle_state.get_value() == value:
            return

        # Pick which arm (if any) currently holds obj, and which should press.
        # The typical BDDL plan for `toggled_on` is [GRASP(obj), TOGGLE_ON(obj)],
        # so the default arm usually already holds obj.
        holding_arm = None
        for arm in self.robot.arm_names:
            if self.robot._ag_obj_in_hand.get(arm) is obj:
                holding_arm = arm
                break
        if holding_arm is not None and len(self.robot.arm_names) > 1:
            pressing_arm = next(a for a in self.robot.arm_names if a != holding_arm)
        else:
            pressing_arm = self.arm

        # Press the toggle marker with the pressing EEF.
        button_pos = toggle_state.visual_marker.get_position_orientation()[0]
        _, current_orientation = self.robot.eef_links[pressing_arm].get_position_orientation()
        desired_pose = (button_pos, current_orientation)

        planning_ok = True
        with self._with_arm_override(pressing_arm):
            try:
                yield from self._move_hand(
                    desired_pose,
                    lock_auxiliary_arm=(holding_arm is not None),
                    ignore_objects=[obj],
                    low_precision=True,
                )
            except ActionPrimitiveError:
                planning_ok = False

        if planning_ok:
            # The toggle fires only after CAN_TOGGLE_STEPS consecutive overlap
            # frames — hold the EEF near the marker long enough for that.
            hold_steps = _toggle_module.m.CAN_TOGGLE_STEPS + 5
            for _ in range(hold_steps):
                yield self._postprocess_action(self._empty_action())
                if toggle_state.get_value() == value:
                    break

        # Physical actuation may miss (alignment, finger shape, planning fail).
        # Fall back to a direct state write so the oracle produces a valid trajectory.
        if toggle_state.get_value() != value:
            toggle_state.set_value(value)
            yield from self._settle_robot()

        if toggle_state.get_value() != value:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "The object did not toggle as expected - maybe try again",
                {
                    "target object": obj.name,
                    "is it currently toggled on": toggle_state.get_value(),
                },
            )
