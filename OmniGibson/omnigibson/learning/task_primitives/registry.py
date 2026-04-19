"""Minimal registry from eval task names to task_primitive specifications."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class TaskPrimitiveSpec:
    task_type: str
    object_name: str
    target_name: Optional[str] = None


DEFAULT_TASK_SPECS: Dict[str, TaskPrimitiveSpec] = {
    "turning_on_radio": TaskPrimitiveSpec(task_type="navigate_and_grasp", object_name="radio_89"),
    "picking_up_trash": TaskPrimitiveSpec(
        task_type="grasp_and_place_inside",
        object_name="can__of__soda.n.01_2",
        target_name="ashcan.n.01_1",
    ),
}


def resolve_task_spec(
    task_name: str,
    object_name: Optional[str] = None,
    target_name: Optional[str] = None,
) -> TaskPrimitiveSpec:
    """Resolve a task name to the object that the expert should manipulate."""
    default_spec = DEFAULT_TASK_SPECS.get(task_name)
    if default_spec is not None:
        return TaskPrimitiveSpec(
            task_type=default_spec.task_type,
            object_name=object_name or default_spec.object_name,
            target_name=target_name or default_spec.target_name,
        )
    if object_name is not None:
        inferred_task_type = "grasp_and_place_inside" if target_name is not None else "navigate_and_grasp"
        return TaskPrimitiveSpec(task_type=inferred_task_type, object_name=object_name, target_name=target_name)
    raise ValueError(
        f"Task '{task_name}' is not registered for task_primitives expert policy. "
        "Add a TaskPrimitiveSpec in learning/task_primitives/registry.py or provide explicit object/target names."
    )
