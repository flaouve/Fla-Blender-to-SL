# Fla-Blender-to-SL

Hybrid Blender 4.2+ add-on, CLI, and AI-assisted reconstruction pipeline for exporting 3D scenes and meshes to **SCP:SL** (`Fla-SL-CustomObjects` / `FlaProjectMER`) `.json` schematics.

## Overview

`Fla-Blender-to-SL` bridges Blender / standard 3D mesh formats (`.obj`, `.glb`, `.gltf`) with SCP:SL schematic files:

- **Deterministic 1:1 Export**: Converts native Blender primitives (cubes, spheres, cylinders), lights, text objects, and empties directly into `Fla-SL-CustomObjects` / `FlaProjectMER` schematic blocks while preserving hierarchy and transforms.
- **Hybrid AI Handoff**: Complex meshes that cannot be represented by a single primitive can be either decomposed automatically into fallback primitives (`decompose`) or exported as placeholder empties alongside multi-angle reference renders and a `handoff_manifest.json` (`ai_handoff`) for AI-assisted primitive reconstruction (`blender-to-sl` skill), then merged back into the base schematic.
- **Blender 4.2+ Extension & Standalone CLI**: Use directly inside Blender's 3D Viewport N-panel or from the command line via `fla-blender-to-sl`.

## Installation

### 1. Blender Extension (Blender 4.2+)

1. Download `fla_blender_to_sl_addon.zip` from the [GitHub Releases](https://github.com/flaouve/Fla-Blender-to-SL/releases) page (or build it locally into `dist/fla_blender_to_sl_addon.zip`).
2. In Blender, open **Edit → Preferences → Get Extensions** (or **Add-ons**).
3. Click the drop-down menu in the top-right corner and choose **Install from Disk...**.
4. Select `fla_blender_to_sl_addon.zip` and enable **Fla-Blender-to-SL Exporter**.

### 2. CLI (`uv`)

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/):

```bash
git clone <repository-url>
cd Fla-Blender-to-SL
uv sync
```

Run the CLI with:

```bash
uv run fla-blender-to-sl --help
```

## Blender N-Panel (`SCP:SL` Tab)

In the 3D Viewport, press `N` to open the sidebar and select the **SCP:SL** tab (**SCP:SL Exporter**):

### Active Object Settings
- **Collidable**: Enable or disable the SCP:SL primitive collider flag (`sl_collidable`, default: enabled).
- **Visible**: Enable or disable the SCP:SL primitive visibility flag (`sl_visible`, default: enabled).
- **Static**: Mark the exported block as static (`sl_static`, default: disabled).
- **Export Mode**: Override automatic primitive classification for the active object:
  - `Auto` — Classify primitive automatically.
  - `Force Cube` — Export bounding box as a Cube.
  - `Force Cylinder` — Export as a Cylinder.
  - `Force Sphere` — Export as a Sphere.
  - `AI Handoff` — Emit a placeholder empty for AI reconstruction.

### Export Operators
- **Export to ProjectMER (.json)**: Exports the scene (or selected objects) directly to a `.json` schematic, decomposing complex meshes into fallback primitives. In the export file browser sidebar, enable **Overwrite Existing** to overwrite an existing `.json` file at the target path (disabled by default to prevent accidental overwrites).
- **Export Hybrid Handoff**: Exports the base `.json` schematic along with `handoff_manifest.json` and multi-angle reference renders (`handoff_renders/`) for complex meshes. Also supports **Overwrite Existing** in the export sidebar.

## CLI Usage

### `inspect`
Inspect a `.obj`, `.glb`, or `.gltf` file and classify native primitives vs. complex meshes:

```bash
uv run fla-blender-to-sl inspect path/to/model.glb
```

### `export`
Export a 3D mesh file to a `Fla-SL-CustomObjects` `.json` schematic (and optional AI handoff assets):

```bash
uv run fla-blender-to-sl export path/to/model.glb -o ./output \
  --complex-mode ai_handoff \
  --max-primitives 16 \
  --max-total-primitives 2000
```

`--complex-mode` options: `ai_handoff` (default), `decompose`, `bbox`.

### `merge`
Merge an AI-reconstructed prop schematic into a placeholder `ObjectId` in a base schematic:

```bash
uv run fla-blender-to-sl merge ./output/model.json \
  --placeholder-id 42 \
  --replacement ./reconstructed_prop.json \
  -o ./output/model_merged.json \
  --max-primitives 2000
```

### `validate`
Validate an existing `Fla-SL-CustomObjects` `.json` schematic:

```bash
uv run fla-blender-to-sl validate ./output/model.json --max-primitives 2000
```

## Running Tests

Run the test suite with `uv`:

```bash
uv run pytest -v
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
