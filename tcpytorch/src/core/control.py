from typing import Any

import core.training
import core.preprocessing
import core.visualization
import core.model
import core.utils


class ModelFacade:
    """
    Facade class for the model
    """

    def __init__(self, setting: dict[str, Any]) -> None:
        """
        Constructor for the model facade

        Args:
        -----
            setting (dict[str, Any]):
                Setting for the model

        Returns:
        --------
            None
        """

        self._setting = setting

    def start(self) -> None:
        ...

    def run_training(self) -> None:
        ...

    def run_evaluation(self) -> None:
        ...
