from classifier_trains.schemas.config import DataloaderConfig


def test_init_dataloader_config() -> None:
    """Test DataloaderConfig initialization"""

    c = DataloaderConfig(batch_size=32, num_workers=4)

    assert c
