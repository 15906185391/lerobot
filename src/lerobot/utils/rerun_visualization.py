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

"""Rerun visualization backend.

Live control-loop streaming to the Rerun viewer (:func:`log_rerun_data`). Callers usually select a
backend at runtime through the dispatch in :mod:`lerobot.utils.visualization_utils` rather than
importing from here directly. Requires the ``viz`` extra (``pip install 'lerobot[viz]'``).
"""

import numbers
import os
import random
import signal
import socket
import time

import numpy as np

from lerobot.configs import DEPTH_MILLIMETER_UNIT, infer_depth_unit
from lerobot.lerobot_types import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR
from .import_utils import require_package

_RERUN_VIEWER_PID: int | None = None


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def init_rerun(
    session_name: str = "lerobot_control_loop", ip: str | None = None, port: int | None = None
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    global _RERUN_VIEWER_PID
    _RERUN_VIEWER_PID = None
    log_rerun_data.blueprint = None  # Reset blueprint cache for new session

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rerun_port = port or int(os.getenv("LEROBOT_RERUN_PORT", "9876"))
    if ip:
        rr.connect_grpc(url=f"rerun+http://{ip}:{rerun_port}/proxy")
    else:
        if _env_bool("LEROBOT_RERUN_AUTO_PORT", False) and port is None:
            rerun_port = _find_free_tcp_port()
        # Keep the viewer in the caller's process group by default, so Ctrl-C/GUI stop also
        # cleans up the spawned Rerun process and frees the port for the next recording.
        detach_process = _env_bool("LEROBOT_RERUN_DETACH_PROCESS", False)
        spawn_viewer = _get_rerun_spawn_viewer()
        if spawn_viewer is None:
            rr.spawn(port=rerun_port, memory_limit=memory_limit)
            return

        _RERUN_VIEWER_PID = spawn_viewer(
            port=rerun_port,
            memory_limit=memory_limit,
            detach_process=detach_process,
        )
        rr.connect_grpc(f"rerun+http://127.0.0.1:{rerun_port}/proxy")


def shutdown_rerun() -> None:
    """Shuts down the Rerun SDK gracefully."""

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    disconnect = getattr(rr, "disconnect", None)
    if callable(disconnect):
        disconnect()
    rr.rerun_shutdown()
    _terminate_spawned_rerun_viewer()


def _get_rerun_spawn_viewer():
    try:
        from rerun import _spawn
    except Exception:
        return None
    return getattr(_spawn, "_spawn_viewer", None)


def _terminate_spawned_rerun_viewer(timeout_s: float = 2.0) -> None:
    global _RERUN_VIEWER_PID
    pid = _RERUN_VIEWER_PID
    _RERUN_VIEWER_PID = None
    if pid is None:
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        return

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            waited_pid, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if waited_pid == pid:
            return
        time.sleep(0.05)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        return


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _find_free_tcp_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except OSError:
        return random.randint(20000, 60999)


def _build_blueprint(
    observation_paths: set[str],
    action_paths: set[str],
    image_paths: set[str],
    metadata_paths: set[str],
):
    """Build a Rerun blueprint laying out camera images, observation and action scalars in separate views.

    Camera images, observation and action scalars are arranged in a grid.
    """

    # Safe + zero-overhead: `log_rerun_data` already ran the `require_package` guard and imported rerun.
    import rerun.blueprint as rrb

    views = [rrb.Spatial3DView(origin="robot", name="robot")]
    views.extend(rrb.Spatial2DView(origin=path, name=path) for path in sorted(image_paths))

    if observation_paths:
        views.append(rrb.TimeSeriesView(name="observation", contents=sorted(observation_paths)))
    if action_paths:
        views.append(rrb.TimeSeriesView(name="action", contents=sorted(action_paths)))
    if metadata_paths:
        views.append(rrb.TimeSeriesView(name="recording", contents=sorted(metadata_paths)))

    return rrb.Blueprint(rrb.Grid(*views))


def _ensure_blueprint(
    observation_paths: set[str],
    action_paths: set[str],
    image_paths: set[str],
    metadata_paths: set[str],
) -> None:
    """Build and send the blueprint once, from the first observation and action data."""
    if getattr(log_rerun_data, "blueprint", None) is not None:
        return

    if not (observation_paths or action_paths or image_paths or metadata_paths):
        return

    # Safe + zero-overhead: `log_rerun_data` already ran the `require_package` guard and imported rerun.
    import rerun as rr

    blueprint = _build_blueprint(observation_paths, action_paths, image_paths, metadata_paths)
    log_rerun_data.blueprint = blueprint
    rr.send_blueprint(blueprint)


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
    metadata: dict[str, float | int | bool] | None = None,
    frame_index: int | None = None,
    timestamp: float | None = None,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 1D NumPy arrays are logged as a single `rr.Scalars` batch under one entity path, so that every
      dimension shares the same view instead of being split across one view per element.
    - Multi-dimensional **action** arrays are flattened and logged as a single `rr.Scalars` batch.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    On the first call, a blueprint is built and sent so observation and action scalars get separate
    time-series views and each image gets its own spatial view.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
        metadata: Optional scalar metadata to log under ``recording.*``.
        frame_index: Optional frame index for Rerun's ``frame_index`` timeline.
        timestamp: Optional timestamp in seconds for Rerun's ``timestamp`` timeline.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    if frame_index is not None:
        rr.set_time("frame_index", sequence=frame_index)
    if timestamp is not None:
        rr.set_time("timestamp", timestamp=timestamp)

    observation_paths: set[str] = set()
    action_paths: set[str] = set()
    image_paths: set[str] = set()
    metadata_paths: set[str] = set()

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
                observation_paths.add(key)
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    rr.log(key, rr.Scalars(arr.astype(float)))
                    observation_paths.add(key)
                else:
                    if arr.shape[-1] == 1:
                        # At record time, the depth unit is inferred from the frame type.
                        depth_unit = infer_depth_unit(arr.dtype)
                        img_entity = rr.DepthImage(
                            arr,
                            meter=1000.0 if depth_unit == DEPTH_MILLIMETER_UNIT else 1.0,
                            colormap=rr.components.Colormap.Viridis,
                        )
                    else:
                        img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity, static=True)
                    image_paths.add(key)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
                action_paths.add(key)
            elif isinstance(v, np.ndarray):
                # Flatten any (incl. higher-dimensional) array into a single batched Scalars
                rr.log(key, rr.Scalars(v.reshape(-1).astype(float)))
                action_paths.add(key)

    if metadata:
        for k, v in metadata.items():
            if v is None:
                continue
            key = str(k) if str(k).startswith("recording.") else f"recording.{k}"
            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
                metadata_paths.add(key)

    _ensure_blueprint(observation_paths, action_paths, image_paths, metadata_paths)
