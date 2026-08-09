#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Standalone PICO visualizer for validating wheeled_arm teleoperation before hardware runs."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.teleoperators.wheeled_arm_pico import WheeledArmPico, WheeledArmPicoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8082, type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--urdf-path", type=Path, default=None)
    parser.add_argument("--scale", default=1.0, type=float)
    parser.add_argument("--activation-threshold", default=0.9, type=float)
    parser.add_argument("--gripper-open-pos", default=0.0, type=float)
    parser.add_argument("--gripper-closed-pos", default=1.0, type=float)
    parser.add_argument("--position-only", action="store_true")
    parser.add_argument("--mock-xr", action="store_true")
    parser.add_argument("--duration-s", default=None, type=float)
    parser.add_argument("--solve-frequency-hz", default=30.0, type=float)
    parser.add_argument("--visualization-update-hz", default=10.0, type=float)
    parser.add_argument("--disable-self-collision", action="store_true")
    parser.add_argument("--d-min", default=0.03, type=float)
    parser.add_argument("--solver", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = WheeledArmPicoConfig(
        urdf_path=args.urdf_path,
        scale=args.scale,
        activation_threshold=args.activation_threshold,
        gripper_open_pos=args.gripper_open_pos,
        gripper_closed_pos=args.gripper_closed_pos,
        position_only=args.position_only,
        mock_xr=args.mock_xr,
        solve_frequency_hz=args.solve_frequency_hz,
        solver=args.solver,
        use_self_collision=not args.disable_self_collision,
        d_min=args.d_min,
        visualize=True,
        visualization_host=args.host,
        visualization_port=args.port,
        visualization_open_browser=not args.no_browser,
        visualization_update_hz=args.visualization_update_hz,
    )
    teleop = WheeledArmPico(cfg)
    teleop.connect()
    print(f"Open http://localhost:{args.port}")
    print("PICO control: hold left/right grip to move each arm; press Y to reset baseline.")
    print("Use left/right trigger to control the left/right gripper.")
    if args.mock_xr:
        print("Mock XR is enabled, so the simulated controllers will move without a PICO device.")

    period_s = 1.0 / args.solve_frequency_hz
    start_t = time.perf_counter()
    last_print_t = 0.0
    try:
        while args.duration_s is None or time.perf_counter() - start_t < args.duration_s:
            loop_t = time.perf_counter()
            action = teleop.get_action()
            now = time.perf_counter()
            if now - last_print_t >= 1.0:
                left = [action[f"left_arm_{idx}.pos"] for idx in range(7)]
                right = [action[f"right_arm_{idx}.pos"] for idx in range(7)]
                grippers = [action["left_gripper.pos"], action["right_gripper.pos"]]
                print(f"left={left!r}")
                print(f"right={right!r}")
                print(f"grippers={grippers!r}")
                last_print_t = now
            time.sleep(max(0.0, period_s - (time.perf_counter() - loop_t)))
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()


if __name__ == "__main__":
    main()
