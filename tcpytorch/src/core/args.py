from typing import Any


def prepare_args_for_training() -> dict[str, Any]:
    raise NotImplementedError()


def prepare_args_for_evaluation() -> dict[str, Any]:
    raise NotImplementedError()
