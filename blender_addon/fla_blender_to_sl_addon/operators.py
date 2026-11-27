"""Blender operators for exporting scenes to SCP:SL schematics and AI handoff packages."""

from __future__ import annotations

import json
from pathlib import Path

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper

from fla_blender_to_sl.exporter import ComplexMode, ExportConfig, export_scene
from fla_blender_to_sl.validator import validate_schematic

from .scene_extract import extract_blender_objects


def _collect_export_objects(context: bpy.types.Context, selected_only: bool) -> list[bpy.types.Object]:
    if selected_only:
        return list(context.selected_objects)
    return list(context.scene.objects)


class FLASL_OT_export_schematic(bpy.types.Operator, ExportHelper):
    """Export Blender scene directly to a Fla-SL-CustomObjects / ProjectMER .json schematic."""

    bl_idname = "flasl.export_schematic"
    bl_label = "Export to ProjectMER (.json)"
    bl_options = {"PRESET"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})
    selected_only: BoolProperty(
        name="Selected Only",
        description="Export only selected objects",
        default=False,
    )
    overwrite_existing: BoolProperty(
        name="Overwrite Existing",
        description="Allow overwriting an existing .json schematic at the target path",
        default=False,
    )
    max_primitives_per_complex: IntProperty(
        name="Max Primitives / Complex",
        description="Maximum fallback primitives per complex mesh",
        default=16,
        min=1,
        max=150,
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        objs = _collect_export_objects(context, self.selected_only)
        out_path = Path(self.filepath)
        if out_path.exists() and not self.overwrite_existing:
            self.report(
                {"ERROR"},
                f"File already exists: {out_path.name}. Enable 'Overwrite Existing' in the export sidebar to overwrite.",
            )
            return {"CANCELLED"}
        graph = extract_blender_objects(objs, scene_name=out_path.stem or context.scene.name)

        config = ExportConfig(
            complex_mode=ComplexMode.DECOMPOSE,
            max_primitives_per_complex=self.max_primitives_per_complex,
        )
        result = export_scene(graph, config=config)
        payload = result.schematic.to_dict()
        report = validate_schematic(payload)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.schematic.to_json(indent=2), encoding="utf-8")

        if not report.valid:
            self.report({"WARNING"}, f"Exported with validation issues: {'; '.join(report.errors)}")
        else:
            self.report(
                {"INFO"},
                f"Exported {report.block_count} blocks ({report.primitive_count} primitives) to {out_path}",
            )
        return {"FINISHED"}


class FLASL_OT_export_hybrid_handoff(bpy.types.Operator):
    """Export deterministic base schematic plus AI handoff manifest and multi-angle renders."""

    bl_idname = "flasl.export_hybrid_handoff"
    bl_label = "Export Hybrid Handoff"
    bl_options = {"PRESET"}

    directory: StringProperty(
        name="Output Directory",
        description="Directory where schematic JSON, handoff_manifest.json, and renders are written",
        subtype="DIR_PATH",
    )
    selected_only: BoolProperty(
        name="Selected Only",
        description="Export only selected objects",
        default=False,
    )
    overwrite_existing: BoolProperty(
        name="Overwrite Existing",
        description="Allow overwriting existing schematic and manifest files in the output directory",
        default=False,
    )
    max_primitives_per_complex: IntProperty(
        name="Suggested Budget / Complex",
        description="Suggested AI reconstruction primitive budget per complex mesh",
        default=16,
        min=1,
        max=150,
    )

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        out_dir = Path(self.directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        objs = _collect_export_objects(context, self.selected_only)
        graph = extract_blender_objects(objs, scene_name=context.scene.name)

        config = ExportConfig(
            complex_mode=ComplexMode.AI_HANDOFF,
            max_primitives_per_complex=self.max_primitives_per_complex,
        )
        result = export_scene(graph, config=config)

        json_path = out_dir / f"{result.schematic_name}.json"
        manifest_path = out_dir / "handoff_manifest.json"
        if not self.overwrite_existing and (json_path.exists() or manifest_path.exists()):
            self.report(
                {"ERROR"},
                f"Output already exists in {out_dir}. Enable 'Overwrite Existing' in the export sidebar to overwrite.",
            )
            return {"CANCELLED"}
        json_path.write_text(result.schematic.to_json(indent=2), encoding="utf-8")

        nodes_by_id = {n.id: n for n in graph.nodes}
        renders_dir = out_dir / "handoff_renders"
        manifest_entries = []
        from fla_blender_to_sl.mesh_loader import render_multiview_sheet

        for entry in result.handoff_entries:
            entry_dict = entry.to_dict()
            node = nodes_by_id.get(entry.node_id)
            if node is not None and node.mesh is not None:
                render_path = renders_dir / f"{entry.name}.png"
                tile_size = 512 if node.mesh.texture_srgb is not None else 256
                render_multiview_sheet(node.mesh, render_path, tile_size=tile_size)
                entry_dict["render_path"] = str(render_path)
            manifest_entries.append(entry_dict)

        manifest_path.write_text(
            json.dumps(
                {
                    "schematic_path": str(json_path),
                    "entries": manifest_entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        self.report(
            {"INFO"},
            f"Exported {json_path.name} with {len(manifest_entries)} AI handoff entries to {out_dir}",
        )
        return {"FINISHED"}


CLASSES = (
    FLASL_OT_export_schematic,
    FLASL_OT_export_hybrid_handoff,
)
