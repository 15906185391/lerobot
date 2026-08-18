#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig, camera_orbbec
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError


class FakeDeviceList:
    def __init__(self, devices):
        self.devices = devices

    def get_count(self):
        return len(self.devices)

    def get_device_by_index(self, index):
        return self.devices[index]


class FakeDeviceInfo:
    def get_serial_number(self):
        return "SN123"

    def get_name(self):
        return "Orbbec Gemini 335"

    def get_uid(self):
        return "UID123"

    def get_vid(self):
        return 11205

    def get_pid(self):
        return 1815

    def get_firmware_version(self):
        return "1.2.3"

    def get_hardware_version(self):
        return "4.5.6"

    def get_connection_type(self):
        return "USB"


class FakeDevice:
    def get_device_info(self):
        return FakeDeviceInfo()


def fake_ob_sdk(*, pipeline=None, context=None):
    return SimpleNamespace(
        OBFormat=SimpleNamespace(RGB="RGB", BGR="BGR", MJPG="MJPG", YUYV="YUYV", UYVY="UYVY"),
        OBSensorType=SimpleNamespace(COLOR_SENSOR="color"),
        Config=MagicMock(return_value=MagicMock()),
        Context=MagicMock(return_value=context or MagicMock()),
        Pipeline=MagicMock(return_value=pipeline or MagicMock()),
    )


@pytest.fixture(autouse=True)
def patch_orbbec_dependency(monkeypatch):
    monkeypatch.setattr(camera_orbbec, "_require_orbbec_sdk", lambda: None)


def make_rgb_frames(width=4, height=3):
    raw = np.arange(width * height * 3, dtype=np.uint8)
    color_frame = MagicMock()
    color_frame.get_width.return_value = width
    color_frame.get_height.return_value = height
    color_frame.get_format.return_value = "RGB"
    color_frame.get_data.return_value = raw

    frames = MagicMock()
    frames.get_color_frame.return_value = color_frame
    return frames, raw.reshape((height, width, 3))


def test_abc_implementation():
    _ = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name="SN123"))


def test_partial_resolution_requires_all_values():
    with pytest.raises(ValueError, match="fps.*width.*height"):
        OrbbecCameraConfig(serial_number_or_name="SN123", width=640, height=480)


def test_make_cameras_from_configs_returns_orbbec_camera():
    config = OrbbecCameraConfig(serial_number_or_name="SN123", width=4, height=3, fps=30)

    cameras = make_cameras_from_configs({"front": config})

    assert isinstance(cameras["front"], OrbbecCamera)


def test_find_cameras(monkeypatch):
    profile = MagicMock()
    profile.get_type.return_value = "COLOR"
    profile.get_format.return_value = "RGB"
    profile.get_width.return_value = 1280
    profile.get_height.return_value = 720
    profile.get_fps.return_value = 30

    profile_list = MagicMock()
    profile_list.get_default_video_stream_profile.return_value = profile

    pipeline = MagicMock()
    pipeline.get_stream_profile_list.return_value = profile_list

    context = MagicMock()
    context.query_devices.return_value = FakeDeviceList([FakeDevice()])
    monkeypatch.setattr(camera_orbbec, "ob", fake_ob_sdk(pipeline=pipeline, context=context))

    found = OrbbecCamera.find_cameras()

    assert found == [
        {
            "name": "Orbbec Gemini 335",
            "type": "Orbbec",
            "id": "SN123",
            "uid": "UID123",
            "vid": 11205,
            "pid": 1815,
            "firmware_version": "1.2.3",
            "hardware_version": "4.5.6",
            "connection_type": "USB",
            "default_stream_profile": {
                "stream_type": "COLOR",
                "format": "RGB",
                "width": 1280,
                "height": 720,
                "fps": 30,
            },
        }
    ]


def test_connect_read_and_disconnect(monkeypatch):
    frames, expected = make_rgb_frames()
    pipeline = MagicMock()
    pipeline.wait_for_frames.return_value = frames
    fake_ob = fake_ob_sdk(pipeline=pipeline)
    monkeypatch.setattr(camera_orbbec, "ob", fake_ob)

    config = OrbbecCameraConfig(serial_number_or_name="SN123", width=4, height=3, fps=30)
    camera = OrbbecCamera(config)
    profile = MagicMock()

    with (
        patch.object(camera, "_find_device", return_value=FakeDevice()),
        patch.object(camera, "_select_color_profile", return_value=profile),
    ):
        camera.connect(warmup=False)
        image = camera.read()
        camera.disconnect()

    np.testing.assert_array_equal(image, expected)
    assert not camera.is_connected
    pipeline.start.assert_called_once()
    pipeline.stop.assert_called_once()


def test_connect_start_failure_clears_partial_state(monkeypatch):
    pipeline = MagicMock()
    pipeline.start.side_effect = RuntimeError("start failed")
    monkeypatch.setattr(camera_orbbec, "ob", fake_ob_sdk(pipeline=pipeline))

    config = OrbbecCameraConfig(serial_number_or_name="SN123", width=4, height=3, fps=30)
    camera = OrbbecCamera(config)

    with (
        patch.object(camera, "_find_device", return_value=FakeDevice()),
        patch.object(camera, "_select_color_profile", return_value=MagicMock()),
        pytest.raises(ConnectionError, match="Failed to open"),
    ):
        camera.connect(warmup=False)

    assert camera.pipeline is None
    assert camera.profile is None
    assert camera.device is None
    assert not camera.is_connected


def test_postprocess_color_mode_conversion():
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape((3, 4, 3))
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name="SN123", color_mode=ColorMode.BGR))
    camera.capture_width = 4
    camera.capture_height = 3

    np.testing.assert_array_equal(camera._postprocess_image(image), image[..., ::-1])


def test_read_before_connect():
    camera = OrbbecCamera(OrbbecCameraConfig(serial_number_or_name="SN123"))

    with pytest.raises(DeviceNotConnectedError):
        camera.read()
