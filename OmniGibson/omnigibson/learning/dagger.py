"""
DAgger entrypoint built on reusable task primitives.
"""

import argparse
import traceback

from omnigibson.learning.task_primitives import (
    CustomGraspBackend,
    FallbackGraspBackend,
    GraspExecutionConfig,
    NavigateAndGraspTask,
    PrimitiveGraspBackend,
    build_grasp_context,
    report_grasp_debug_context,
    shutdown_simulation,
)


BACKEND_FACTORIES = {
    "custom": lambda: CustomGraspBackend(),
    "primitive": lambda: PrimitiveGraspBackend(),
    "both": lambda: FallbackGraspBackend(CustomGraspBackend(), PrimitiveGraspBackend()),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Navigate to object and grasp it")
    parser.add_argument("--task_name", type=str, default="turning_on_radio", help="Name of the task to load")
    parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode (no GUI)")
    parser.add_argument(
        "--eval_instance_ids",
        type=str,
        default=None,
        help="Comma-separated list of instance IDs to evaluate",
    )
    parser.add_argument("--object_name", type=str, default="radio_89", help="Name of object to grasp")
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable visualization of obstacles, poses, and trajectories",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum sampling attempts for pose near object",
    )
    parser.add_argument(
        "--grasp_mode",
        type=str,
        choices=["custom", "primitive", "both"],
        default="custom",
        help="Grasp execution mode backend",
    )
    parser.add_argument(
        "--primitive_attempts",
        type=int,
        default=5,
        help="Number of retries for the built-in GRASP primitive backend",
    )
    return parser.parse_args()


def parse_eval_instance_ids(raw_value: str | None):
    if not raw_value:
        return None
    return [int(value) for value in raw_value.split(",") if value.strip()]


def main():
    try:
        args = parse_args()
        config = GraspExecutionConfig(
            max_samples=args.max_samples,
            primitive_attempts=args.primitive_attempts,
            verbose=True,
            visualize=args.visualize,
        )
        context = build_grasp_context(
            task_name=args.task_name,
            headless=args.headless,
            eval_instance_ids=parse_eval_instance_ids(args.eval_instance_ids),
            config=config,
        )
        obj = context.scene.object_registry("name", args.object_name)
        if obj is None:
            print(f"[ERROR] Object '{args.object_name}' not found in scene")
            return

        report_grasp_debug_context(context, obj, max_samples=args.max_samples)
        backend = BACKEND_FACTORIES[args.grasp_mode]()
        task = NavigateAndGraspTask(object_name=args.object_name, backend=backend)
        result = task.run(context)

        if not result.success:
            print(f"[FAILED] Grasp task failed via backend={result.backend}, result={result.result.value}")
            if result.error_message:
                print(result.error_message)
        else:
            print(f"Finished executing grasp with backend={result.backend}")
    except Exception:
        traceback.print_exc()
    finally:
        shutdown_simulation()


if __name__ == "__main__":
    main()
