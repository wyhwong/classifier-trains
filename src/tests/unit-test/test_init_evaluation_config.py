from classifier_trains.schemas import config


def test_init_evaluation_config():
    """Test EvaluationConfig initialization"""

    c = config.EvaluationConfig(
        name="test",
        device="cpu",
        random_seed=42,
        precision=64,
        evalset_dir="test",
        models=[],
    )

    assert c
