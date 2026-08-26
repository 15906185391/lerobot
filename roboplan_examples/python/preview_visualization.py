from __future__ import annotations

import pinocchio as pin
from pinocchio.visualize import ViserVisualizer


def make_static_and_preview_visualizers(
    pin_model: pin.Model,
    collision_model: pin.GeometryModel,
    visual_model: pin.GeometryModel,
    host: str,
    port: str,
    preview_collision_model: pin.GeometryModel | None = None,
    fixed_root: str = "initial_robot",
    preview_root: str = "preview_robot",
    preview_color: tuple[float, float, float, float] = (0.0, 0.48, 1.0, 0.28),
) -> tuple[ViserVisualizer, ViserVisualizer]:
    fixed_viz = ViserVisualizer(pin_model, collision_model, visual_model, copy_models=True)
    fixed_viz.initViewer(open=False, loadModel=False, host=host, port=port)
    fixed_viz.loadViewerModel(rootNodeName=fixed_root)
    fixed_viz.displayCollisions(False)

    # Use collision geometry for the moving ghost so mesh opacity is reliable even
    # when the visual URDF meshes are DAE files with embedded materials.
    if preview_collision_model is not None:
        preview_geometry_model = preview_collision_model
    elif collision_model is not None:
        preview_geometry_model = collision_model
    else:
        preview_geometry_model = visual_model
    preview_viz = ViserVisualizer(
        pin_model,
        collision_model=pin.GeometryModel(),
        visual_model=preview_geometry_model,
        copy_models=False,
    )
    preview_viz.initViewer(
        viewer=fixed_viz.viewer,
        open=False,
        loadModel=False,
        host=host,
        port=port,
    )
    preview_viz.loadViewerModel(
        rootNodeName=preview_root,
        visual_color=list(preview_color),
    )
    preview_viz.displayCollisions(False)
    return fixed_viz, preview_viz
