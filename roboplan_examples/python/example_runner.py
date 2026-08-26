#!/usr/bin/env python3
"""Run one RoboPlan Python example from a JSON parameter payload."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import ctypes
import runpy
import sys
import sysconfig
import types
from pathlib import Path
from typing import Any


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))


def _add_cmeel_site_packages() -> None:
    purelib = Path(sysconfig.get_paths()["purelib"])
    cmeel_site = (
        purelib
        / "cmeel.prefix"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if cmeel_site.exists() and str(cmeel_site) not in sys.path:
        sys.path.insert(0, str(cmeel_site))


_add_cmeel_site_packages()


def _shared_library_roots() -> list[Path]:
    purelib = Path(sysconfig.get_paths()["purelib"])
    roots = [
        Path(sys.prefix) / "lib",
        purelib / "lib",
        purelib / "cmeel.prefix" / "lib",
    ]
    unique_roots: list[Path] = []
    for root in roots:
        if root.exists() and root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def _preload_shared_library(name: str) -> bool:
    for root in _shared_library_roots():
        candidate = root / name
        if not candidate.exists():
            continue
        ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        return True
    return False


def _preload_runtime_libraries() -> None:
    for lib_name in ("libtinyxml2.so.11", "libtinyxml2.so.10", "libtinyxml2.so"):
        try:
            if _preload_shared_library(lib_name):
                break
        except OSError:
            continue


_preload_runtime_libraries()


def _install_tyro_import_stub() -> None:
    """Allow GUI-driven imports when Tyro is not installed.

    The examples only need Tyro inside their ``if __name__ == "__main__"`` block.
    The GUI runner calls ``main(**params)`` directly, so a tiny placeholder is
    enough to import those modules without adding an unnecessary runtime dependency.
    """
    if importlib.util.find_spec("tyro") is not None:
        return
    stub = types.ModuleType("tyro")

    def cli(_fn):
        raise RuntimeError(
            "tyro is not installed. Install tyro to run this example as a "
            "standalone command-line script."
        )

    stub.cli = cli
    sys.modules["tyro"] = stub


_install_tyro_import_stub()


def _patch_viser_visualizer_port_argument() -> None:
    """Keep Pinocchio's ViserVisualizer compatible with newer Viser releases."""
    try:
        import time
        import viser
        from pinocchio.visualize import ViserVisualizer
    except (ModuleNotFoundError, ImportError):
        return

    def initViewer(
        self,
        viewer=None,
        open=False,
        loadModel=False,
        host="localhost",
        port="8000",
    ):
        if (viewer is not None) and not isinstance(viewer, viser.ViserServer):
            raise RuntimeError(
                "'viewer' argument must be None or a valid ViserServer instance."
            )

        self.viewer = viewer or viser.ViserServer(host=host, port=int(port))
        self.frames = {}

        if open:
            import webbrowser

            webbrowser.open(f"http://{self.viewer.get_host()}:{self.viewer.get_port()}")
            while len(self.viewer.get_clients()) == 0:
                time.sleep(0.1)

        if loadModel:
            self.loadViewerModel()

    ViserVisualizer.initViewer = initViewer


_patch_viser_visualizer_port_argument()


ENUM_TYPES = {
    ("example_cartesian_planning", "speed_mode"): (
        "roboplan.cartesian_planning",
        "CartesianSpeedMode",
    ),
    ("example_rrt", "toppra_mode"): ("roboplan.toppra", "SplineFittingMode"),
    (
        "example_toppra_joint_planning",
        "toppra_mode",
    ): ("roboplan.toppra", "SplineFittingMode"),
}


def _coerce_value(module_name: str, parameter_name: str, value: Any) -> Any:
    enum_ref = ENUM_TYPES.get((module_name, parameter_name))
    if enum_ref is None or not isinstance(value, str):
        return value

    enum_module_name, enum_name = enum_ref
    enum_module = importlib.import_module(enum_module_name)
    enum_type = getattr(enum_module, enum_name)
    return getattr(enum_type, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", help="Example module name, such as example_rrt.")
    parser.add_argument(
        "--params-json",
        default="{}",
        help="JSON object passed as keyword arguments to the example main().",
    )
    args = parser.parse_args()

    try:
        params = json.loads(args.params_json)
    except json.JSONDecodeError as exc:
        print(f"Invalid parameter JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(params, dict):
        print("Parameter JSON must decode to an object.", file=sys.stderr)
        return 2

    coerced_params = {
        name: _coerce_value(args.module, name, value) for name, value in params.items()
    }
    try:
        module = importlib.import_module(args.module)
    except (ModuleNotFoundError, ImportError) as exc:
        print(
            f"Could not import '{args.module}': {exc}. "
            "Make sure the roboplan example dependencies and shared libraries "
            f"are installed in the Python environment that launched this GUI: {sys.executable}",
            file=sys.stderr,
        )
        return 1
    if hasattr(module, "main"):
        module.main(**coerced_params)
    elif coerced_params:
        print(
            f"{args.module} does not define main(), so parameters cannot be passed.",
            file=sys.stderr,
        )
        return 2
    else:
        runpy.run_module(args.module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
