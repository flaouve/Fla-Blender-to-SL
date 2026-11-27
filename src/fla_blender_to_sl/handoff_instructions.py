"""Self-contained AI reconstruction instructions written into Hybrid Handoff export folders."""

from __future__ import annotations


def build_ai_instructions_md(schematic_filename: str, entry_count: int) -> str:
    """Return a concise, self-contained Markdown guide for any AI completing a Hybrid Handoff."""
    return f"""# AI Hybrid Handoff Instructions

This folder was exported by **Fla-Blender-to-SL** (`Export Hybrid Handoff`).
There are **{entry_count}** complex mesh placeholder(s) listed in `handoff_manifest.json`.

## Goal
Reconstruct each complex entry in `handoff_manifest.json` using clean, low-poly Unity primitives (`Cube`, `Cylinder`, `Sphere`, `Capsule`, `Quad`) and `Text` blocks — **never** per-triangle geometry — then merge them into `{schematic_filename}` and validate.

## Step-by-Step Workflow
1. **Read `handoff_manifest.json`**:
   - Each entry gives `placeholder_object_id`, `name`, `world_position_unity`, `world_rotation_unity`, `world_bounds_unity`, `dominant_color_hex`, `suggested_budget`, and `render_path`.
2. **Inspect Reference Renders**:
   - Open every `render_path` image (`handoff_renders/<name>.png`) to see the 6-angle orthographic/isometric views (including any texture-painted logos, signs, or surface markings).
3. **Reconstruct Each Entry Within `suggested_budget`**:
   - Write a small Python script to generate `<name>_reconstructed.json` in `Fla-SL-CustomObjects` / `FlaProjectMER` format:
     - `RootObjectId`: `1000`, child `ObjectId`s starting at `1001`, `ParentId`: `1000`.
     - Coordinate system: **Unity Left-Handed Y-Up** (`1.0` unit = `1 meter`), local to the placeholder origin.
     - `BlockType`: `0` (`Empty`), `1` (`Primitive`), `2` (`Light`), `8` (`Text`).
     - `PrimitiveType`: `0` (`Sphere`), `1` (`Capsule`, height `2.0`), `2` (`Cylinder`, height `2.0`), `3` (`Cube`), `4` (`Plane`), `5` (`Quad`).
     - `PrimitiveFlags`: `3` (`Collidable | Visible`) for main structure, `2` (`Visible` only) for surface trim/details/logos.
     - `Color`: `"RRGGBBAA"` hex (no `#`) or HDR `"R:G:B:A"` (255-scaled floats).
     - `Scale`: never `0` or negative on any axis (minimum `0.002`). For `Cylinder`/`Capsule`, `Scale.y = height / 2.0`.
     - `Text` (`BlockType: 8`): use `Scale` around `0.02`–`0.04` (never `1.0`), `DisplaySize: {{"x": 20.0, "y": 5.0}}` (no `z`), and `"<align=center><color=#RRGGBBAA><size=14>TEXT</size></color></align>"`. Note TextMeshPro front faces `-Z` at `Rotation.y = 0`.
4. **Merge Into Base Schematic**:
   ```bash
   fla-blender-to-sl merge {schematic_filename} --placeholder-id <ID> --replacement <name>_reconstructed.json -o {schematic_filename}
   ```
5. **Validate Final Output**:
   ```bash
   fla-blender-to-sl validate {schematic_filename}
   ```
"""
