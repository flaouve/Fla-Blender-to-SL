"""Blender 4.2+ Extension entrypoint for Fla-Blender-to-SL."""

from __future__ import annotations

try:
    import bpy
    from bpy.props import BoolProperty, EnumProperty
except ImportError:  # pragma: no cover - allows importing scene_extract outside Blender
    bpy = None  # type: ignore[assignment]

bl_info = {
    "name": "Fla-Blender-to-SL Exporter",
    "author": "Flaouve",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > SCP:SL",
    "description": "Export Blender scenes to SCP:SL (Fla-SL-CustomObjects / FlaProjectMER) schematics",
    "category": "Import-Export",
}


def register() -> None:
    if bpy is None:
        return

    import sys
    from pathlib import Path

    addon_dir = str(Path(__file__).resolve().parent)
    if addon_dir not in sys.path:
        sys.path.insert(0, addon_dir)

    # Clear cached submodules on addon reload/reinstall so updated IR fields are picked up
    for mod_name in list(sys.modules):
        if mod_name == "fla_blender_to_sl" or mod_name.startswith("fla_blender_to_sl."):
            del sys.modules[mod_name]

    from . import operators, ui

    bpy.types.Object.sl_collidable = BoolProperty(
        name="Collidable",
        description="Enable SCP:SL primitive collider flag",
        default=True,
    )
    bpy.types.Object.sl_visible = BoolProperty(
        name="Visible",
        description="Enable SCP:SL primitive visible flag",
        default=True,
    )
    bpy.types.Object.sl_static = BoolProperty(
        name="Static",
        description="Mark exported block as static",
        default=False,
    )
    bpy.types.Object.sl_export_override = EnumProperty(
        name="Export Mode",
        description="Override automatic primitive classification for this object",
        items=[
            ("AUTO", "Auto", "Classify primitive automatically"),
            ("FORCE_CUBE", "Force Cube", "Export bounding box as Cube"),
            ("FORCE_CYLINDER", "Force Cylinder", "Export as Cylinder"),
            ("FORCE_SPHERE", "Force Sphere", "Export as Sphere"),
            ("FORCE_AI_HANDOFF", "AI Handoff", "Emit placeholder for AI reconstruction"),
        ],
        default="AUTO",
    )

    for cls in (*operators.CLASSES, *ui.CLASSES):
        bpy.utils.register_class(cls)


def unregister() -> None:
    if bpy is None:
        return

    from . import operators, ui

    for cls in reversed((*operators.CLASSES, *ui.CLASSES)):
        bpy.utils.unregister_class(cls)

    for prop_name in ("sl_export_override", "sl_static", "sl_visible", "sl_collidable"):
        if hasattr(bpy.types.Object, prop_name):
            delattr(bpy.types.Object, prop_name)
