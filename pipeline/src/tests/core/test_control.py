import os
from glob import glob

import core.control
import core.utils


def test_ModelFacade():
    """
    Test ModelFacade.

    This is an end-to-end test but not a functional test.
    The correctness of the result is not checked.

    """

    directory_label = len(glob("results/*")) + 1
    setting = core.utils.load_yml("src/tests/test_setting.yml")
    facade = core.control.ModelFacade(setting=setting)
    facade.start()

    expected_files = [
        f"results/{directory_label}_in_test/accuracy_history.csv",
        f"results/{directory_label}_in_test/accuracy_history.jpg",
        f"results/{directory_label}_in_test/loss_history.csv",
        f"results/{directory_label}_in_test/loss_history.jpg",
        f"results/{directory_label}_in_test/best_model.onnx",
        f"results/{directory_label}_in_test/best_model.pt",
        f"results/{directory_label}_in_test/last_model.onnx",
        f"results/{directory_label}_in_test/last_model.pt",
        f"results/{directory_label}_in_test/class_mapping.yml",
        f"results/{directory_label}_in_test/confusion_matrix_in_test.png",
        f"results/{directory_label}_in_test/preview_train.png",
        f"results/{directory_label}_in_test/preview_val.png",
        f"results/{directory_label}_in_test/preview_eval.png",
        f"results/{directory_label}_in_test/roc_curve_cats.png",
        f"results/{directory_label}_in_test/roc_curve_dogs.png",
    ]

    for expected_file in expected_files:
        os.remove(expected_file)

    os.rmdir(f"results/{directory_label}_in_test/")
