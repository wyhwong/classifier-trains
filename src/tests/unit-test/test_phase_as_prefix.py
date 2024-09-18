from classifier_trains.schemas.constants import Phase


def test_phase_as_a_prefix():
    """Test Phase as a prefix"""

    assert Phase.TRAINING("loss") == "train_loss"
    assert Phase.VALIDATION("accuracy") == "val_accuracy"
    assert Phase.TESTING("f1") == "test_f1"
