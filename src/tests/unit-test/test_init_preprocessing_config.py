import pytest

from classifier_trains.schemas import config, constants


def test_init_resize_config():
    """Test ResizeConfig initialization"""

    for interpolation in constants.InterpolationType:
        for padding in constants.PaddingType:
            for maintain_aspect_ratio in [True, False]:
                c = config.ResizeConfig(
                    width=224,
                    height=224,
                    interpolation=interpolation,
                    padding=padding,
                    maintain_aspect_ratio=maintain_aspect_ratio,
                )
                assert c


def test_init_resize_config_failed_with_invalid_interpolation():
    """Test ResizeConfig initialization failed with invalid interpolation"""

    with pytest.raises(ValueError):
        config.ResizeConfig(
            width=224,
            height=224,
            interpolation="INVALID",
            padding=constants.PaddingType.CENTER,
            maintain_aspect_ratio=True,
        )


def test_init_resize_config_failed_with_invalid_padding():
    """Test ResizeConfig initialization failed with invalid padding"""

    with pytest.raises(ValueError):
        config.ResizeConfig(
            width=224,
            height=224,
            interpolation=constants.InterpolationType.BILINEAR,
            padding="INVALID",
            maintain_aspect_ratio=True,
        )


def test_init_resize_config_failed_with_invalid_width():
    """Test ResizeConfig initialization failed with invalid width"""

    with pytest.raises(ValueError):
        config.ResizeConfig(
            width=-1,
            height=224,
            interpolation=constants.InterpolationType.BILINEAR,
            padding=constants.PaddingType.CENTER,
            maintain_aspect_ratio=True,
        )


def test_init_resize_config_failed_with_invalid_height():
    """Test ResizeConfig initialization failed with invalid height"""

    with pytest.raises(ValueError):
        config.ResizeConfig(
            width=224,
            height=-1,
            interpolation=constants.InterpolationType.BILINEAR,
            padding=constants.PaddingType.CENTER,
            maintain_aspect_ratio=True,
        )


def test_init_spatial_transform_config():
    """Test SpatialTransformConfig initialization"""

    c = config.SpatialTransformConfig(
        hflip_prob=0.5,
        vflip_prob=0.5,
        max_rotate_in_degree=45,
        allow_center_crop=True,
        allow_random_crop=True,
    )

    assert c


def test_init_spatial_transform_config_failed_with_hflip_prob_under_low_bound():
    """Test SpatialTransformConfig initialization failed with invalid hflip_prob"""

    with pytest.raises(ValueError):
        config.SpatialTransformConfig(
            hflip_prob=-0.1,
            vflip_prob=0.5,
            max_rotate_in_degree=45,
            allow_center_crop=True,
            allow_random_crop=True,
        )


def test_init_spatial_transform_config_failed_with_hflip_prob_over_high_bound():
    """Test SpatialTransformConfig initialization failed with invalid hflip_prob"""

    with pytest.raises(ValueError):
        config.SpatialTransformConfig(
            hflip_prob=1.1,
            vflip_prob=0.5,
            max_rotate_in_degree=45,
            allow_center_crop=True,
            allow_random_crop=True,
        )


def test_init_spatial_transform_config_failed_with_vflip_prob_under_low_bound():
    """Test SpatialTransformConfig initialization failed with invalid vflip_prob"""

    with pytest.raises(ValueError):
        config.SpatialTransformConfig(
            hflip_prob=0.5,
            vflip_prob=-0.1,
            max_rotate_in_degree=45,
            allow_center_crop=True,
            allow_random_crop=True,
        )


def test_init_spatial_transform_config_failed_with_vflip_prob_over_high_bound():
    """Test SpatialTransformConfig initialization failed with invalid vflip_prob"""

    with pytest.raises(ValueError):
        config.SpatialTransformConfig(
            hflip_prob=0.5,
            vflip_prob=1.1,
            max_rotate_in_degree=45,
            allow_center_crop=True,
            allow_random_crop=True,
        )


def test_init_spatial_transform_config_failed_with_max_rotate_in_degree_under_low_bound():
    """Test SpatialTransformConfig initialization failed with invalid max_rotate_in_degree"""

    with pytest.raises(ValueError):
        config.SpatialTransformConfig(
            hflip_prob=0.5,
            vflip_prob=0.5,
            max_rotate_in_degree=-0.1,
            allow_center_crop=True,
            allow_random_crop=True,
        )


def test_init_spatial_transform_config_failed_with_max_rotate_in_degree_over_high_bound():
    """Test SpatialTransformConfig initialization failed with invalid max_rotate_in_degree"""

    with pytest.raises(ValueError):
        config.SpatialTransformConfig(
            hflip_prob=0.5,
            vflip_prob=0.5,
            max_rotate_in_degree=180.1,
            allow_center_crop=True,
            allow_random_crop=True,
        )


def test_init_color_transform_config():
    """Test ColorTransformConfig initialization"""

    c = config.ColorTransformConfig(
        allow_gray_scale=True,
        allow_random_color=True,
    )

    assert c


def test_init_preprocessing_config():
    """Test PreprocessingConfig initialization"""

    c = config.PreprocessingConfig(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        resize_config=config.ResizeConfig(
            width=224,
            height=224,
            interpolation=constants.InterpolationType.BILINEAR,
            padding=constants.PaddingType.CENTER,
            maintain_aspect_ratio=True,
        ),
        spatial_config=config.SpatialTransformConfig(
            hflip_prob=0.5,
            vflip_prob=0.5,
            max_rotate_in_degree=45,
            allow_center_crop=True,
            allow_random_crop=True,
        ),
        color_config=config.ColorTransformConfig(
            allow_gray_scale=True,
            allow_random_color=True,
        ),
    )

    assert c
