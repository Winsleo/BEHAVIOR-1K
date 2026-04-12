"""Minimal registry from eval task names to task_primitive specifications."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class TaskPrimitiveSpec:
    task_type: str
    object_name: str


DEFAULT_TASK_SPECS: Dict[str, TaskPrimitiveSpec] = {
    "turning_on_radio": TaskPrimitiveSpec(task_type="navigate_and_grasp", object_name="radio_89"),
}


def resolve_task_spec(task_name: str, object_name: Optional[str] = None) -> TaskPrimitiveSpec:
    """Resolve a task name to the object that the expert should manipulate."""
    if object_name is not None:
        return TaskPrimitiveSpec(task_type="navigate_and_grasp", object_name=object_name)
    if task_name not in DEFAULT_TASK_SPECS:
        raise ValueError(
            f"Task '{task_name}' is not registered for task_primitives expert policy. "
            "Add a TaskPrimitiveSpec in learning/task_primitives/registry.py or provide an explicit object_name."
        )
    return DEFAULT_TASK_SPECS[task_name]