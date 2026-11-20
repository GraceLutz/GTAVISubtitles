import numpy as np
import pytest

from gta_felirat.config import ConfigError, config_from_dict
from gta_felirat.facecam import FacecamStyle, facecam_rect, place_facecam


def test_facecam_is_placed_at_saved_position_with_camera_aspect():
    assert facecam_rect(1920, 1080, 1280, 720, FacecamStyle(0.5, 0.25, 0.2, False)) == (960, 270, 384, 216)


def test_facecam_is_kept_inside_frame():
    x, y, width, height = facecam_rect(1920, 1080, 1280, 720, FacecamStyle(1.0, 1.0, 0.2, False))
    assert x + width == 1920 and y + height == 1080


def test_place_facecam_returns_new_frame_with_camera_pixels():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    cam = np.full((720, 1280, 3), 120, np.uint8)
    style = FacecamStyle(0.1, 0.1, 0.2, False)
    result = place_facecam(frame, cam, style)
    assert frame.max() == 0
    x, y, width, height = facecam_rect(1920, 1080, 1280, 720, style)
    assert result[y + height // 2, x + width // 2].tolist() == [120, 120, 120]


@pytest.mark.parametrize("raw", [
    {"webcam_x": 1.5},
    {"webcam_width_ratio": 0.9},
    {"webcam_device_index": 0, "device_index": 0},
])
def test_invalid_webcam_config_is_rejected(raw):
    with pytest.raises(ConfigError):
        config_from_dict(raw)
