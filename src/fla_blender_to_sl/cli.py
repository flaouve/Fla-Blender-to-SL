"""Command-line interface for fla-blender-to-sl."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

from fla_blender_to_sl.classifier import classify_mesh
from fla_blender_to_sl.exporter import ComplexMode, ExportConfig, export_scene
from fla_blender_to_sl.mesh_loader import load_mesh_file, render_multiview_sheet
from fla_blender_to_sl.merger import merge_reconstructed_blocks
from fla_blender_to_sl.validator import validate_schematic


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"\s+", "_", name.strip())
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "", cleaned)
    return cleaned or "mesh"


def _cmd_inspect(args: argparse.Namespace) -> int:
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    graph = load_mesh_file(input_path)
    native_count = 0
    complex_count = 0
    for node in graph.nodes:
        if node.mesh is None:
            continue
        res = classify_mesh(node.mesh, override=node.override)
        if res.is_complex:
            complex_count += 1
        else:
            native_count += 1

    print(
        f"Scene: {graph.name}\n"
        f"Total nodes: {len(graph.nodes)}\n"
        f"Native primitives: {native_count}\n"
        f"Complex meshes: {complex_count}\n"
        f"Estimated base blocks: {len(graph.nodes)}"
    )
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = load_mesh_file(input_path)
    mode = ComplexMode(args.complex_mode)
    config = ExportConfig(
        complex_mode=mode,
        max_primitives_per_complex=args.max_primitives,
        max_total_primitives=args.max_total_primitives,
    )
    result = export_scene(graph, config)

    schematic_dict = result.schematic.to_dict()
    report = validate_schematic(schematic_dict, max_primitives=args.max_total_primitives)
    if not report.valid:
        for err in report.errors:
            print(f"Validation error: {err}", file=sys.stderr)
        return 1

    json_path = out_dir / f"{result.schematic_name}.json"
    json_path.write_text(result.schematic.to_json(indent=2), encoding="utf-8")

    if mode == ComplexMode.AI_HANDOFF and result.handoff_entries:
        renders_dir = out_dir / "handoff_renders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        nodes_by_id = {n.id: n for n in graph.nodes}
        manifest_entries = []
        for entry in result.handoff_entries:
            entry_dict = entry.to_dict()
            node = nodes_by_id.get(entry.node_id)
            if node is not None and node.mesh is not None:
                render_file = renders_dir / f"{_safe_filename(entry.name)}_{entry.placeholder_object_id}.png"
                render_multiview_sheet(node.mesh, render_file)
                entry_dict["render_path"] = str(render_file.relative_to(out_dir))
            manifest_entries.append(entry_dict)

        manifest_path = out_dir / "handoff_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schematic": str(json_path.name),
                    "handoff_entries": manifest_entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        f"Exported {json_path} "
        f"(blocks={report.block_count}, primitives={report.primitive_count}, "
        f"handoff={len(result.handoff_entries)})"
    )
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    base_path = Path(args.base_json)
    repl_path = Path(args.replacement)
    out_path = Path(args.output)

    if not base_path.exists():
        print(f"Error: base schematic not found: {base_path}", file=sys.stderr)
        return 1
    if not repl_path.exists():
        print(f"Error: replacement schematic not found: {repl_path}", file=sys.stderr)
        return 1

    base_data = json.loads(base_path.read_text(encoding="utf-8"))
    repl_data = json.loads(repl_path.read_text(encoding="utf-8"))

    try:
        merged = merge_reconstructed_blocks(
            base_data,
            placeholder_id=args.placeholder_id,
            replacement_schematic=repl_data,
        )
    except ValueError as exc:
        print(f"Merge error: {exc}", file=sys.stderr)
        return 1

    report = validate_schematic(merged, max_primitives=args.max_primitives)
    if not report.valid:
        for err in report.errors:
            print(f"Validation error: {err}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Merged schematic written to {out_path} (blocks={report.block_count}, primitives={report.primitive_count})")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    schematic_path = Path(args.schematic_json)
    if not schematic_path.exists():
        print(f"Error: schematic file not found: {schematic_path}", file=sys.stderr)
        return 1

    data = json.loads(schematic_path.read_text(encoding="utf-8"))
    report = validate_schematic(data, max_primitives=args.max_primitives)
    for warn in report.warnings:
        print(f"Warning: {warn}", file=sys.stderr)
    if not report.valid:
        for err in report.errors:
            print(f"Error: {err}", file=sys.stderr)
        return 1

    print(f"VALID: {schematic_path} (blocks={report.block_count}, primitives={report.primitive_count})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fla-blender-to-sl",
        description="Hybrid Blender / 3D mesh to SCP:SL (Fla-SL-CustomObjects) schematic converter.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect a 3D mesh file and classify native vs complex nodes.")
    p_inspect.add_argument("input_file", help="Path to .obj, .glb, or .gltf file.")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_export = sub.add_parser("export", help="Export a 3D mesh file to a Fla-SL-CustomObjects JSON schematic.")
    p_export.add_argument("input_file", help="Path to .obj, .glb, or .gltf file.")
    p_export.add_argument("-o", "--output-dir", required=True, help="Output directory for schematic and handoff assets.")
    p_export.add_argument(
        "--complex-mode",
        choices=[m.value for m in ComplexMode],
        default=ComplexMode.AI_HANDOFF.value,
        help="How to handle complex meshes (default: ai_handoff).",
    )
    p_export.add_argument(
        "--max-primitives",
        type=int,
        default=16,
        help="Max primitives per complex mesh in decompose/handoff mode (default: 16).",
    )
    p_export.add_argument(
        "--max-total-primitives",
        type=int,
        default=2000,
        help="Max total primitives allowed in the schematic (default: 2000).",
    )
    p_export.set_defaults(func=_cmd_export)

    p_merge = sub.add_parser("merge", help="Merge an AI-reconstructed schematic into a base schematic placeholder.")
    p_merge.add_argument("base_json", help="Path to the base schematic JSON.")
    p_merge.add_argument("--placeholder-id", type=int, required=True, help="Placeholder Empty ObjectId to parent under.")
    p_merge.add_argument("--replacement", required=True, help="Path to the reconstructed prop schematic JSON.")
    p_merge.add_argument("-o", "--output", required=True, help="Output path for the merged schematic JSON.")
    p_merge.add_argument(
        "--max-primitives",
        type=int,
        default=2000,
        help="Max total primitives allowed after merge (default: 2000).",
    )
    p_merge.set_defaults(func=_cmd_merge)

    p_validate = sub.add_parser("validate", help="Validate a Fla-SL-CustomObjects schematic JSON.")
    p_validate.add_argument("schematic_json", help="Path to schematic JSON file.")
    p_validate.add_argument(
        "--max-primitives",
        type=int,
        default=2000,
        help="Max total primitives allowed (default: 2000).",
    )
    p_validate.set_defaults(func=_cmd_validate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
