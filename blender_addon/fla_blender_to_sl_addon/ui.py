"""Blender N-panel UI for SCP:SL object flags and schematic export."""

from __future__ import annotations

import bpy


class VIEW3D_PT_fla_sl_export(bpy.types.Panel):
    """N-panel tab for configuring SCP:SL per-object flags and exporting schematics."""

    bl_label = "SCP:SL Exporter"
    bl_idname = "VIEW3D_PT_fla_sl_export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SCP:SL"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        obj = context.active_object

        box = layout.box()
        box.label(text="Active Object Settings")
        if obj is not None:
            box.prop(obj, "sl_collidable")
            box.prop(obj, "sl_visible")
            box.prop(obj, "sl_static")
            box.prop(obj, "sl_export_override")
        else:
            box.label(text="No active object selected", icon="INFO")

        layout.separator()
        col = layout.column(align=True)
        col.operator("flasl.export_schematic", icon="EXPORT")
        col.operator("flasl.export_hybrid_handoff", icon="RENDER_STILL")


CLASSES = (VIEW3D_PT_fla_sl_export,)
