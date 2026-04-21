"""Automatic task planning from BDDL goal conditions.

Given a compiled BDDL task (from BehaviorTask), this module:
1. Picks the shortest ground goal state option (fewest atomic conditions).
2. Filters out conditions already satisfied at runtime.
3. Maps each unsatisfied atomic predicate to a sequence of action primitives.
4. Orders the steps so that prerequisites (e.g. OPEN) come before placements.

Supported BDDL predicates (StarterSemanticActionPrimitiveSet physical primitives):
  ontop, inside, nextto, under — placement predicates
  toggled_on, open — state predicates

Unsupported predicates (no physical primitive available) generate a warning and are skipped:
  folded, unfolded, broken, on_fire, frozen, hot, filled, draped, overlaid,
  attached, covered, saturated, cooked, touching, insource, inroom, grasped
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitiveSet

log = logging.getLogger(__name__)

# Predicates that cannot be satisfied by any available physical primitive.
# They are logged as warnings and skipped rather than raising errors.
_UNSUPPORTED_PREDICATES = frozenset({
    "folded", "unfolded", "broken", "on_fire", "frozen", "hot",
    "filled", "draped", "overlaid", "attached",
    "covered", "saturated", "cooked",
    "touching", "insource", "inroom", "grasped",
    "under",
})


@dataclass
class PrimitiveStep:
    """A single action primitive to execute.

    Attributes:
        action: The primitive action enum value.
        object_name: BDDL instance name of the primary object (e.g. "bowl.n.01_1").
        target_name: BDDL instance name of the target for binary predicates
            (e.g. the receptacle for PLACE_INSIDE / PLACE_ON_TOP).
    """
    action: StarterSemanticActionPrimitiveSet
    object_name: str
    target_name: Optional[str] = None


# ── predicate info extraction ──────────────────────────────────────────────

def _extract_predicate_info(head) -> Optional[Tuple[str, bool, List[str]]]:
    """Extract (predicate_name, is_negated, args) from a compiled HEAD node.

    Returns None for complex expressions (Conjunction, Disjunction, etc.) that
    cannot be decomposed into a single atomic predicate.
    """
    from bddl.condition_evaluation import Negation
    from bddl.predicates import Predicate

    if not head.children:
        return None
    child = head.children[0]
    if isinstance(child, Negation):
        if child.children and isinstance(child.children[0], Predicate):
            pred = child.children[0]
            return (pred.STATE_NAME, True, list(pred.inputs))
    elif isinstance(child, Predicate):
        return (child.STATE_NAME, False, list(child.inputs))
    return None


# ── predicate → primitive mapping ──────────────────────────────────────────

def _predicate_to_steps(
    predicate_name: str,
    negated: bool,
    args: List[str],
) -> List[PrimitiveStep]:
    """Map a single BDDL predicate to zero or more primitive steps.

    Binary placement predicates use ``args[0]`` as the object to move and
    ``args[1]`` as the target surface/container.

    State predicates (unary) toggle or open/close the single ``args[0]`` object.
    """
    A = StarterSemanticActionPrimitiveSet

    # ── Binary placement predicates (positive only) ────────────────────────
    if not negated and len(args) == 2:
        if predicate_name == "ontop":
            return [
                PrimitiveStep(A.GRASP, args[0], target_name=args[1]),
                PrimitiveStep(A.PLACE_ON_TOP, args[1], target_name=args[0]),
            ]
        if predicate_name == "inside":
            return [
                PrimitiveStep(A.GRASP, args[0], target_name=args[1]),
                PrimitiveStep(A.PLACE_INSIDE, args[1], target_name=args[0]),
            ]
        if predicate_name == "nextto":
            # Navigate grasped object near the target, then release.
            return [
                PrimitiveStep(A.GRASP, args[0], target_name=args[1]),
                PrimitiveStep(A.NAVIGATE_TO, args[1]),
                PrimitiveStep(A.RELEASE, args[0]),
            ]

    # ── Unary state predicates ─────────────────────────────────────────────
    # Prepend NAVIGATE_TO so the robot reaches the object first.
    # For toggle predicates, also add GRASP (the robot must hold the object).
    if len(args) == 1:
        if predicate_name == "toggled_on":
            action = A.TOGGLE_ON if not negated else A.TOGGLE_OFF
            return [
                PrimitiveStep(A.GRASP, args[0]),
                PrimitiveStep(action, args[0]),
            ]
        if predicate_name == "open":
            action = A.OPEN if not negated else A.CLOSE
            return [
                PrimitiveStep(action, args[0]),
            ]

    # ── Unsupported predicates ─────────────────────────────────────────────
    if predicate_name in _UNSUPPORTED_PREDICATES:
        neg_str = "not " if negated else ""
        log.warning(
            f"Predicate '{neg_str}{predicate_name}({', '.join(args)})' has no physical "
            "primitive mapping — skipping. Task may be incomplete."
        )
        return []

    # Unknown predicate — also skip with a warning
    neg_str = "not " if negated else ""
    log.warning(
        f"Unknown predicate '{neg_str}{predicate_name}({', '.join(args)})' — skipping."
    )
    return []


# ── step ordering ──────────────────────────────────────────────────────────

# Each action is assigned a phase number so that prerequisites execute first:
#   Phase 0: OPEN (containers/drawers must be open before placing inside)
#   Phase 1: NAVIGATE_TO (position robot)
#   Phase 2: GRASP + PLACE pairs (main manipulation)
#   Phase 3: RELEASE (after manipulation)
#   Phase 4: CLOSE (close containers after placing)
#   Phase 5: TOGGLE_ON / TOGGLE_OFF (state changes last — toggle may not be implemented)
_PHASE: Dict[StarterSemanticActionPrimitiveSet, int] = {
    StarterSemanticActionPrimitiveSet.OPEN: 0,
    StarterSemanticActionPrimitiveSet.NAVIGATE_TO: 1,
    StarterSemanticActionPrimitiveSet.GRASP: 2,
    StarterSemanticActionPrimitiveSet.PLACE_ON_TOP: 2,
    StarterSemanticActionPrimitiveSet.PLACE_INSIDE: 2,
    StarterSemanticActionPrimitiveSet.RELEASE: 3,
    StarterSemanticActionPrimitiveSet.CLOSE: 4,
    StarterSemanticActionPrimitiveSet.TOGGLE_ON: 5,
    StarterSemanticActionPrimitiveSet.TOGGLE_OFF: 5,
}

_PLACE_ACTIONS = frozenset({
    StarterSemanticActionPrimitiveSet.PLACE_ON_TOP,
    StarterSemanticActionPrimitiveSet.PLACE_INSIDE,
    StarterSemanticActionPrimitiveSet.RELEASE,
})


def _order_steps(steps: List[PrimitiveStep]) -> List[PrimitiveStep]:
    """Order steps so prerequisites (OPEN) precede manipulation,
    GRASP-PLACE pairs remain contiguous, and partially-implemented
    primitives (TOGGLE) are deferred to the end.

    Strategy:
    1. Collect phase-0 prerequisite steps (OPEN).
    2. Group each GRASP with its immediately following PLACE/RELEASE.
    3. Append CLOSE post-actions.
    4. Append TOGGLE steps last (may raise NotImplementedError).
    """
    pre_steps: List[PrimitiveStep] = []     # OPEN
    manip_groups: List[List[PrimitiveStep]] = []  # [GRASP, PLACE, ...] groups
    post_steps: List[PrimitiveStep] = []    # CLOSE
    deferred_steps: List[PrimitiveStep] = []  # TOGGLE_ON / TOGGLE_OFF

    i = 0
    while i < len(steps):
        step = steps[i]
        phase = _PHASE.get(step.action, 2)

        if phase == 0:
            pre_steps.append(step)
            i += 1
        elif step.action == StarterSemanticActionPrimitiveSet.GRASP:
            group = [step]
            i += 1
            # Consume consecutive PLACE / RELEASE steps as part of this group
            while i < len(steps) and steps[i].action in _PLACE_ACTIONS:
                group.append(steps[i])
                i += 1
            manip_groups.append(group)
        elif phase == 4:
            post_steps.append(step)
            i += 1
        elif phase >= 5:
            deferred_steps.append(step)
            i += 1
        else:
            # Standalone placement / navigate without a preceding GRASP
            manip_groups.append([step])
            i += 1

    result: List[PrimitiveStep] = []
    result.extend(pre_steps)
    for group in manip_groups:
        result.extend(group)
    result.extend(post_steps)
    result.extend(deferred_steps)
    return result


# ── main planning API ──────────────────────────────────────────────────────

def plan_from_goal(behavior_task, evaluate_fn=None) -> List[PrimitiveStep]:
    """Decompose a BehaviorTask's goal conditions into ordered primitive steps.

    The function selects the goal option (from ``compiled_task.ground_goal_state_options``)
    with the fewest unsatisfied atomic conditions, then maps each unsatisfied condition
    to a sequence of ``PrimitiveStep`` objects using the physical StarterSemanticActionPrimitiveSet.

    Args:
        behavior_task: The ``BehaviorTask`` instance (``env.task``) with a valid
            ``compiled_task`` attribute populated by ``initialize_activity()``.
        evaluate_fn: Optional predicate evaluation callback with signature
            ``(predicate_cls, *entities) -> bool``.  Defaults to
            ``behavior_task._evaluate_predicate``.

    Returns:
        Ordered list of ``PrimitiveStep`` ready for execution by ``BDDLSequenceTask``.
        Returns an empty list if all conditions are already satisfied or no mapping
        is available.

    Raises:
        RuntimeError: If ``compiled_task`` is None (``initialize_activity`` was not called).
    """
    compiled_task = behavior_task.compiled_task
    if compiled_task is None:
        raise RuntimeError(
            "BehaviorTask.compiled_task is None — was initialize_activity() called?"
        )

    ground_options = compiled_task.ground_goal_state_options
    if not ground_options:
        log.warning("No ground goal state options available; returning empty plan.")
        return []

    if evaluate_fn is None:
        evaluate_fn = behavior_task._evaluate_predicate

    # Evaluate top-5 shortest options; choose the one with fewest unsatisfied atoms.
    ground_options_sorted = sorted(ground_options, key=len)
    best_steps: Optional[List[PrimitiveStep]] = None
    best_count = float("inf")

    for option in ground_options_sorted[:5]:
        unsatisfied: List[Tuple[str, bool, List[str]]] = []
        for head in option:
            try:
                if head.evaluate(evaluate_fn):
                    continue  # already satisfied — skip
            except Exception:
                pass  # treat evaluation errors as unsatisfied
            info = _extract_predicate_info(head)
            if info is not None:
                unsatisfied.append(info)

        steps: List[PrimitiveStep] = []
        for pred_name, negated, args in unsatisfied:
            steps.extend(_predicate_to_steps(pred_name, negated, args))

        if len(unsatisfied) < best_count:
            best_count = len(unsatisfied)
            best_steps = steps

    if best_steps is None or len(best_steps) == 0:
        log.info("No actionable primitive steps derived from BDDL goal (all satisfied or unsupported).")
        return []

    ordered = _order_steps(best_steps)

    log.info(f"BDDL task plan ({len(ordered)} steps):")
    for i, step in enumerate(ordered):
        target_str = f" → {step.target_name}" if step.target_name else ""
        log.info(f"  {i+1}. {step.action.name}({step.object_name}{target_str})")

    return ordered
