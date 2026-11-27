# Fla-Blender-to-SL Hybrid Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid Blender-to-SCP:SL (`Fla-SL-CustomObjects` / `FlaProjectMER`) conversion pipeline consisting of a deterministic Python core + CLI + Blender 4.2+ addon for instant 1:1 map/primitive/light/text export, paired with an AI reconstruction skill for complex `.blend`/`.obj`/`.glb` meshes that avoids triangle-explosion performance issues.

**Architecture:** All inputs (Blender scenes via the addon, or standalone `.obj`/`.glb`/`.gltf` files via the CLI) normalize into a shared pure-Python `SceneGraph` IR. A geometric classifier separates native SCP:SL primitives (`Cube`, `Cylinder`, `Sphere`, `Capsule`, `Quad`), lights, texts, and empties (exported deterministically 1:1 with exact Blender-to-Unity coordinate/scale/color conversion) from complex arbitrary meshes; complex meshes either use a budget-capped OBB/multi-box fallback or emit a handoff package (multi-angle renders + exact bounds + dominant colors) for the `blender-to-sl` AI skill to reconstruct cleanly and merge back into the final validated `.json` schematic.

**Tech Stack:** Python 3.11+ (tested on 3.14 via `uv`), `numpy`, `trimesh`, `Pillow`, `pytest`, Blender 4.2+ Extension API (`bpy`, `blender_manifest.toml`), and Antigravity Skill (`SKILL.md`).

**Spec:** Interactive requirements gathered on 2026-11-27 and verified against `/home/flaouve/Projects/Fla-SL-CustomObjects/Assets/DONT TOUCH/Scripts/` (`Schematic.cs`, `SchematicBlock.cs`, `PrimitiveComponent.cs`, `LightComponent.cs`, `TextComponent.cs`, `BlockType.cs`, `PrimitiveFlags.cs`).

## Global Constraints

- **Target Schema Compatibility:** Output `.json` MUST deserialize cleanly in `Fla-SL-CustomObjects` (`SchematicObjectDataList` / `SchematicBlockData`) and `FlaProjectMER` with `RootObjectId = 1000` and unique integer `ObjectId` values starting at `1001`.
- **Supported BlockTypes:** `Empty = 0`, `Primitive = 1`, `Light = 2`, `Text = 8` (exact values from `BlockType.cs`).
- **Supported Unity PrimitiveTypes:** `Sphere = 0`, `Capsule = 1`, `Cylinder = 2` (height `2.0`), `Cube = 3`, `Plane = 4`, `Quad = 5`.
- **Coordinate System:** Blender Right-Handed Z-Up `(bx, by, bz)` maps to Unity Left-Handed Y-Up `(bx, bz, by)`; Euler angles follow Unity intrinsic `Z-X-Y` (`R = R_y @ R_x @ R_z`) in degrees `[0, 360)`.
- **Scale Floor:** No emitted block may have any scale axis `< 0.002` or negative; `Empty` parent blocks MUST keep scale `(1.0, 1.0, 1.0)`.
- **Color Encoding:** LDR colors (`r,g,b <= 1.0`) serialize as 8-char uppercase `"RRGGBBAA"` hex without `#`; HDR colors (`any(r,g,b > 1.0)`) serialize as `"R:G:B:A"` with 255-scaled floats and dot decimal separator (matching `PrimitiveComponent.cs:68-77`).
- **Light Scaling:** Multiply Unity `Intensity` and `Range` by `100.0` on export (matching `LightComponent.cs:8-11`).
- **Primitive Budget:** Never convert arbitrary meshes triangle-by-triangle; default max budget is `150` primitives per complex mesh and `2000` primitives per schematic (configurable).
- **Testability Without Blender Binary:** Core package (`src/fla_blender_to_sl/`) and all unit tests MUST run via `uv run pytest` without requiring `bpy` or a live Blender installation.

## Review Focus

1. **Negative / mirrored Blender scales (e.g. `(-1, 1, 1)` from Mirror modifier):** Must normalize into positive Unity scales without triggering `PrimitiveComponent.cs:58` negative-scale collider bugs or inverting transforms — pinned in Task 2 (`test_negative_scale_normalized`).
2. **Non-uniformly scaled or unapplied-transform Blender cylinders/capsules:** Must compute dimensions from actual mesh vertex bounds × scale and divide Unity Cylinder/Capsule `y` scale by `2.0` so cylinders do not export 2× too tall — pinned in Task 3 (`test_classify_scaled_cylinder_divides_height_by_two`).
3. **Nested parent hierarchies with non-uniform parent scale (Unity shear):** Must bake scale down into leaf primitives and emit `Empty` parents at scale `(1, 1, 1)` so child rotations never shear — pinned in Task 5 (`test_non_uniform_parent_scale_flattens_without_shear`).
4. **High-poly arbitrary meshes exceeding SCP:SL network limits:** Must enforce `max_primitives` budget in fallback decomposition and emit a `handoff_manifest.json` entry when `mode="ai_handoff"` — pinned in Task 4 (`test_decomposer_respects_primitive_budget`) and Task 5 (`test_exporter_emits_handoff_manifest_for_complex_meshes`).
5. **Root schematic names with spaces, duplicate child names, or zero-thickness planes (`scale == 0`):** Must sanitize schematic filenames (no spaces, per `Schematic.cs:107`), assign unique `ObjectId`s, and clamp scale axes to `>= 0.002` — pinned in Task 5 (`test_exporter_sanitizes_names_and_clamps_zero_scale`).

---

## File Structure

- `pyproject.toml` — Package metadata, dependencies (`numpy`, `trimesh`, `Pillow`), dev dependencies (`pytest`), and `fla-blender-to-sl` console script entry point.
- `src/fla_blender_to_sl/schema.py` — Exact `Fla-SL-CustomObjects` enums (`BlockType`, `PrimitiveType`, `PrimitiveFlags`, `LightType`, `ShadowType`) and JSON serialization dataclasses (`Vec3`, `Vec2`, `SchematicBlockData`, `SchematicObjectDataList`).
- `src/fla_blender_to_sl/transforms.py` — Pure math for Blender RH Z-Up → Unity LH Y-Up positions, quaternions/matrices → Unity `Z-X-Y` Euler degrees, scale normalization, and linear-to-sRGB / HDR color formatting.
- `src/fla_blender_to_sl/ir.py` — Shared intermediate representation (`MaterialInfo`, `MeshData`, `LightData`, `TextData`, `SceneNode`, `SceneGraph`) consumed by both the standalone loader and the Blender addon.
- `src/fla_blender_to_sl/classifier.py` — Analyzes `MeshData` vertices/faces to classify whether a mesh matches a native Unity primitive (`Cube`, `Cylinder`, `Sphere`, `Capsule`, `Quad`) within tolerance, or is `COMPLEX`.
- `src/fla_blender_to_sl/decomposer.py` — Deterministic, budget-capped fallback that approximates `COMPLEX` meshes with 1..N oriented bounding boxes (OBB / recursive PCA split) without ever emitting per-triangle geometry.
- `src/fla_blender_to_sl/exporter.py` — Converts `SceneGraph` into `SchematicObjectDataList` JSON, flattening non-uniform parent scales, clamping minimum scales, and generating `handoff_manifest.json` for complex meshes in `ai_handoff` mode.
- `src/fla_blender_to_sl/validator.py` — Audits any `SchematicObjectDataList` JSON for unique `ObjectId`s, valid `ParentId`s, scale floors, color syntax, primitive budget, and world-space grounding.
- `src/fla_blender_to_sl/mesh_loader.py` — Loads `.obj`/`.glb`/`.gltf` via `trimesh` into `SceneGraph` and renders multi-angle orthographic/isometric `.png` previews (Front, Back, Left, Right, Top, Iso) using `Pillow` software rasterization (zero GPU/OpenGL dependency) for AI handoff inspection.
- `src/fla_blender_to_sl/merger.py` — Replaces `COMPLEX_PLACEHOLDER` `Empty` nodes in a base schematic JSON with AI-reconstructed primitive blocks (re-indexing `ObjectId`s and parenting cleanly).
- `src/fla_blender_to_sl/cli.py` — CLI subcommands: `inspect`, `export`, `merge`, `validate`.
- `blender_addon/fla_blender_to_sl_addon/blender_manifest.toml` — Blender 4.2+ Extension manifest.
- `blender_addon/fla_blender_to_sl_addon/scene_extract.py` — Adapter extracting Blender `bpy` objects (meshes, lights, texts, empties, Principled BSDF colors, custom `sl_*` properties) into `SceneGraph` (tested with lightweight mock objects).
- `blender_addon/fla_blender_to_sl_addon/operators.py` & `ui.py` & `__init__.py` — Blender N-panel UI (`Collidable`, `Visible`, `Static`, `Export Mode: Auto / Force Primitive / AI Handoff`) and Export operators.
- `/home/flaouve/.gemini/config/skills/blender-to-sl/SKILL.md` — Hybrid AI skill instructing the agent how to run `fla-blender-to-sl export --complex-mode ai_handoff`, visually inspect extracted renders, reconstruct complex meshes within budget, merge, and validate.

---

### Task 1: Project Scaffolding & `Fla-SL-CustomObjects` JSON Schema

**Files:**
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/pyproject.toml`
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/src/fla_blender_to_sl/__init__.py`
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/src/fla_blender_to_sl/schema.py`
- Test: `/home/flaouve/Projects/Fla-Blender-to-SL/tests/test_schema.py`

**Interfaces:**
- Consumes: None
- Produces:
  - `BlockType(IntEnum)`: `EMPTY=0, PRIMITIVE=1, LIGHT=2, TEXT=8`
  - `PrimitiveType(IntEnum)`: `SPHERE=0, CAPSULE=1, CYLINDER=2, CUBE=3, PLANE=4, QUAD=5`
  - `PrimitiveFlags(IntFlag)`: `NONE=0, COLLIDABLE=1, VISIBLE=2`
  - `LightType(IntEnum)`: `SPOT=0, POINT=2`
  - `ShadowType(IntEnum)`: `NONE=0, HARD=1, SOFT=2`
  - `Vec3(x: float, y: float, z: float)` with `.to_dict() -> dict[str, float]`
  - `Vec2(x: float, y: float)` with `.to_dict() -> dict[str, float]` (no `z` key, matching `SerializableVector.cs:13`)
  - `SchematicBlockData` dataclass with `.to_dict() -> dict[str, Any]` (omitting `None` keys, matching `NullValueHandling.Ignore`)
  - `SchematicObjectDataList(root_object_id: int = 1000, blocks: list[SchematicBlockData])` with `.to_dict() -> dict[str, Any]` and `.to_json(indent: int = 2) -> str`
  - Helper builders: `make_empty_block(...)`, `make_primitive_block(...)`, `make_light_block(...)`, `make_text_block(...)`

- [ ] **Step 1: Initialize project structure and write failing schema tests**

Create `pyproject.toml` with `python >= 3.11`, dependencies `numpy`, `trimesh`, `Pillow`, dev dependency `pytest`, and console script `fla-blender-to-sl = "fla_blender_to_sl.cli:main"`.

Write `tests/test_schema.py`:
```python
import json
from fla_blender_to_sl.schema import (
    BlockType,
    LightType,
    PrimitiveFlags,
    PrimitiveType,
    SchematicObjectDataList,
    ShadowType,
    Vec2,
    Vec3,
    make_light_block,
    make_primitive_block,
    make_text_block,
)


def test_primitive_block_matches_fla_sl_schema():
    block = make_primitive_block(
        name="Wall_01",
        object_id=1001,
        parent_id=1000,
        position=Vec3(1.0, 2.0, 3.0),
        rotation=Vec3(0.0, 90.0, 0.0),
        scale=Vec3(4.0, 3.0, 0.2),
        primitive_type=PrimitiveType.CUBE,
        color="434B56FF",
        flags=PrimitiveFlags.COLLIDABLE | PrimitiveFlags.VISIBLE,
        static=True,
    )
    doc = SchematicObjectDataList(root_object_id=1000, blocks=[block])
    parsed = json.loads(doc.to_json())

    assert parsed["RootObjectId"] == 1000
    b = parsed["Blocks"][0]
    assert b["Name"] == "Wall_01"
    assert b["ObjectId"] == 1001
    assert b["ParentId"] == 1000
    assert b["BlockType"] == 1
    assert "AnimatorName" not in b
    assert b["Properties"] == {
        "PrimitiveType": 3,
        "Color": "434B56FF",
        "PrimitiveFlags": 3,
        "Static": True,
    }


def test_light_block_scales_intensity_and_range_by_100():
    block = make_light_block(
        name="CeilingLight",
        object_id=1002,
        parent_id=1000,
        position=Vec3(0.0, 3.0, 0.0),
        rotation=Vec3(90.0, 0.0, 0.0),
        light_type=LightType.POINT,
        color="FFFFFFFF",
        unity_intensity=1.5,
        unity_range=10.0,
        shadow_type=ShadowType.SOFT,
    )
    props = block.to_dict()["Properties"]
    assert props["LightType"] == 2
    assert props["Intensity"] == 150.0
    assert props["Range"] == 1000.0
    assert props["ShadowType"] == 2


def test_text_block_formats_tmp_tags_and_omits_z_in_display_size():
    block = make_text_block(
        name="Sign",
        object_id=1003,
        parent_id=1000,
        position=Vec3(0.0, 2.0, 0.0),
        rotation=Vec3(0.0, 180.0, 0.0),
        scale=Vec3(0.03, 0.03, 0.03),
        text="SITE-83",
        color="FFFFFFFF",
        font_size=14.0,
        align="center",
        display_size=Vec2(20.0, 5.0),
    )
    props = block.to_dict()["Properties"]
    assert props["Text"] == "<align=center><color=#FFFFFFFF><size=14.0>SITE-83</size></color></align>"
    assert props["DisplaySize"] == {"x": 20.0, "y": 5.0}
    assert "z" not in props["DisplaySize"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fla_blender_to_sl'`

- [ ] **Step 3: Implement `schema.py` and `__init__.py`**

Implement the enums, dataclasses, and helper builders in `src/fla_blender_to_sl/schema.py` matching the exact PascalCase JSON keys (`RootObjectId`, `Blocks`, `Name`, `ObjectId`, `ParentId`, `Position`, `Rotation`, `Scale`, `BlockType`, `Properties`) and rounding floats to 5 decimal places to avoid `-0.0` and floating-point noise.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_schema.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/flaouve/Projects/Fla-Blender-to-SL && git init && git add pyproject.toml src/ tests/ && git commit -m "feat: initialize project and Fla-SL-CustomObjects JSON schema"
```

---

### Task 2: Blender-to-Unity Coordinate, Rotation, Scale & Color Math

**Files:**
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/src/fla_blender_to_sl/transforms.py`
- Test: `/home/flaouve/Projects/Fla-Blender-to-SL/tests/test_transforms.py`

**Interfaces:**
- Consumes: `Vec3` from `fla_blender_to_sl.schema`
- Produces:
  - `MIN_SCALE: float = 0.002`
  - `blender_pos_to_unity(pos: tuple[float, float, float]) -> Vec3`
  - `blender_scale_to_unity(scale: tuple[float, float, float], min_scale: float = MIN_SCALE) -> Vec3`
  - `blender_matrix_to_unity_trs(matrix_4x4: np.ndarray) -> tuple[Vec3, Vec3, Vec3]` — returns `(position, euler_deg_zxy, positive_scale)`
  - `unity_euler_to_matrix(euler_deg: Vec3) -> np.ndarray` — `R_y @ R_x @ R_z` for round-trip verification
  - `linear_rgba_to_sl_color(rgba: tuple[float, float, float, float], emission_strength: float = 0.0) -> str`

- [ ] **Step 1: Write failing transform and color tests**

Write `tests/test_transforms.py`:
```python
import math
import numpy as np
from fla_blender_to_sl.transforms import (
    blender_matrix_to_unity_trs,
    blender_pos_to_unity,
    blender_scale_to_unity,
    linear_rgba_to_sl_color,
    unity_euler_to_matrix,
)


def test_position_maps_z_up_to_y_up():
    u = blender_pos_to_unity((1.5, 2.5, 3.5))
    assert (u.x, u.y, u.z) == (1.5, 3.5, 2.5)


def test_rotation_90_deg_around_blender_z_maps_to_270_around_unity_y():
    # +90 deg around Blender +Z moves +X (Right) to +Y (Forward)
    # In Unity LH, moving +X (Right) to +Z (Forward) is -90 deg (270 deg) around +Y
    m = np.eye(4)
    m[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pos, rot, scale = blender_matrix_to_unity_trs(m)
    assert math.isclose(rot.x, 0.0, abs_tol=1e-3)
    assert math.isclose(rot.y, 270.0, abs_tol=1e-3)
    assert math.isclose(rot.z, 0.0, abs_tol=1e-3)
    # Verify round-trip matrix equality against P @ R_b @ P
    P = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    assert np.allclose(unity_euler_to_matrix(rot), P @ m[:3, :3] @ P, atol=1e-4)


def test_negative_scale_normalized():
    # Review Focus #1: negative/mirrored Blender scale must produce positive Unity scale
    m = np.diag([-2.0, 3.0, 4.0, 1.0])
    _, _, scale = blender_matrix_to_unity_trs(m)
    assert scale.x == 2.0 and scale.y == 4.0 and scale.z == 3.0


def test_zero_scale_clamped_to_min_scale():
    s = blender_scale_to_unity((0.0, 1.0, -0.0001))
    assert s.x == 0.002 and s.y == 0.002 and s.z == 1.0


def test_linear_rgba_to_ldr_hex_and_hdr_colon():
    assert linear_rgba_to_sl_color((0.0, 0.0, 0.0, 1.0)) == "000000FF"
    assert linear_rgba_to_sl_color((1.0, 1.0, 1.0, 0.5)) == "FFFFFF80"
    hdr = linear_rgba_to_sl_color((1.0, 0.5, 0.0, 1.0), emission_strength=2.0)
    assert ":" in hdr
    r, g, b, a = [float(v) for v in hdr.split(":")]
    assert r > 255.0 and g > 120.0 and b == 0.0 and a == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_transforms.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `transforms.py`**

Implement `src/fla_blender_to_sl/transforms.py`:
- Change-of-basis matrix `P = [[1,0,0],[0,0,1],[0,1,0]]`.
- Decompose `M_u = P @ M_b @ P`: extract translation `M_u[:3, 3]`, column norms as positive scales (handling negative determinant by keeping proper rotation `det(R) = +1` via polar decomposition / sign flip on the reflected axis), and extract intrinsic `Z-X-Y` (`R_y @ R_x @ R_z`) Euler angles in degrees normalized to `[0, 360)`.
- Implement IEC 61966-2-1 linear→sRGB transfer for LDR hex `"RRGGBBAA"`, and `"R:G:B:A"` when `emission_strength > 1.0` or any channel `> 1.0`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_transforms.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/flaouve/Projects/Fla-Blender-to-SL && git add src/fla_blender_to_sl/transforms.py tests/test_transforms.py && git commit -m "feat: add Blender-to-Unity transform and color conversion math"
```

---

### Task 3: Scene Intermediate Representation & Primitive Classifier

**Files:**
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/src/fla_blender_to_sl/ir.py`
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/src/fla_blender_to_sl/classifier.py`
- Test: `/home/flaouve/Projects/Fla-Blender-to-SL/tests/test_classifier.py`

**Interfaces:**
- Consumes: `PrimitiveType`, `PrimitiveFlags`, `LightType`, `ShadowType`, `Vec2` from `schema.py`
- Produces:
  - `MaterialInfo(name: str, base_color_linear: tuple[float, float, float, float], emission_color_linear: tuple[float, float, float] = (0,0,0), emission_strength: float = 0.0)`
  - `MeshData(vertices: np.ndarray, faces: np.ndarray, material: MaterialInfo)` (vertices in local object space, Blender coords)
  - `LightData(light_type: LightType, color_linear: tuple[float, float, float, float], intensity: float, range_m: float, spot_angle: float = 30.0, shadow_type: ShadowType = ShadowType.NONE)`
  - `TextData(text: str, color_linear: tuple[float, float, float, float], font_size: float = 14.0, align: str = "center", display_size: Vec2 = Vec2(20.0, 5.0))`
  - `NodeKind(Enum)`: `EMPTY`, `MESH`, `LIGHT`, `TEXT`
  - `ExportOverride(Enum)`: `AUTO`, `FORCE_CUBE`, `FORCE_CYLINDER`, `FORCE_SPHERE`, `FORCE_AI_HANDOFF`
  - `SceneNode(id: str, name: str, kind: NodeKind, parent_id: str | None, local_matrix_blender: np.ndarray, mesh: MeshData | None = None, light: LightData | None = None, text: TextData | None = None, collidable: bool = True, visible: bool = True, static: bool = False, override: ExportOverride = ExportOverride.AUTO)`
  - `SceneGraph(name: str, nodes: list[SceneNode])`
  - `ClassificationResult(primitive_type: PrimitiveType | None, local_center_blender: np.ndarray, unity_scale: tuple[float, float, float], confidence: float, is_complex: bool)`
  - `classify_mesh(mesh: MeshData, world_scale_blender: tuple[float, float, float] = (1.0, 1.0, 1.0), override: ExportOverride = ExportOverride.AUTO) -> ClassificationResult`

- [ ] **Step 1: Write failing classifier tests**

Write `tests/test_classifier.py`:
```python
import numpy as np
import trimesh
from fla_blender_to_sl.classifier import classify_mesh
from fla_blender_to_sl.ir import ExportOverride, MaterialInfo, MeshData
from fla_blender_to_sl.schema import PrimitiveType


def _mat() -> MaterialInfo:
    return MaterialInfo(name="Steel", base_color_linear=(0.2, 0.25, 0.3, 1.0))


def test_classify_box_as_cube():
    box = trimesh.creation.box(extents=(2.0, 4.0, 3.0))
    res = classify_mesh(MeshData(box.vertices, box.faces, _mat()))
    assert not res.is_complex
    assert res.primitive_type == PrimitiveType.CUBE
    # Blender (2, 4, 3) -> Unity (2, 3, 4)
    assert np.allclose(res.unity_scale, (2.0, 3.0, 4.0), atol=1e-3)


def test_classify_scaled_cylinder_divides_height_by_two():
    # Review Focus #2: Unity Cylinder height is 2.0, so 6.0m tall cylinder needs unity_scale.y == 3.0
    cyl = trimesh.creation.cylinder(radius=0.5, height=2.0, sections=32)
    res = classify_mesh(
        MeshData(cyl.vertices, cyl.faces, _mat()),
        world_scale_blender=(1.0, 1.0, 3.0),
    )
    assert not res.is_complex
    assert res.primitive_type == PrimitiveType.CYLINDER
    assert np.allclose(res.unity_scale, (1.0, 3.0, 1.0), atol=1e-2)


def test_classify_uv_sphere():
    sph = trimesh.creation.uv_sphere(radius=1.0, count=[16, 16])
    res = classify_mesh(MeshData(sph.vertices, sph.faces, _mat()))
    assert not res.is_complex
    assert res.primitive_type == PrimitiveType.SPHERE
    assert np.allclose(res.unity_scale, (2.0, 2.0, 2.0), atol=1e-2)


def test_classify_torus_as_complex():
    torus = trimesh.creation.torus(major_radius=1.0, minor_radius=0.25)
    res = classify_mesh(MeshData(torus.vertices, torus.faces, _mat()))
    assert res.is_complex
    assert res.primitive_type is None


def test_override_forces_ai_handoff_even_for_cube():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    res = classify_mesh(
        MeshData(box.vertices, box.faces, _mat()),
        override=ExportOverride.FORCE_AI_HANDOFF,
    )
    assert res.is_complex
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ir.py` and `classifier.py`**

Implement `src/fla_blender_to_sl/ir.py` with the dataclasses above and `src/fla_blender_to_sl/classifier.py`:
- Apply `world_scale_blender` to local vertices when computing extents (`dx, dy, dz`), and compute centroid offset `local_center_blender`.
- Compare volume ratio and surface-to-primitive SDF/radial residual against `Cube` (all vertices lie on AABB boundary planes, `vol_ratio > 0.96`), `Sphere` (normalized ellipsoid radius `||p / r|| ≈ 1.0`, `std < 0.03`), `Cylinder` (circular cross-section in XY with flat Z caps, `radial_std < 0.03`, `vol_ratio > 0.90`, `unity_scale.y = dz / 2.0`), `Quad` (one axis extent `< 1e-4` and 4–8 planar vertices).
- Otherwise return `is_complex=True, primitive_type=None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_classifier.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/flaouve/Projects/Fla-Blender-to-SL && git add src/fla_blender_to_sl/ir.py src/fla_blender_to_sl/classifier.py tests/test_classifier.py && git commit -m "feat: add SceneGraph IR and geometric primitive classifier"
```

---

### Task 4: Budget-Capped Deterministic Fallback Decomposer

**Files:**
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/src/fla_blender_to_sl/decomposer.py`
- Test: `/home/flaouve/Projects/Fla-Blender-to-SL/tests/test_decomposer.py`

**Interfaces:**
- Consumes: `MeshData` from `ir.py`, `PrimitiveType` from `schema.py`
- Produces:
  - `DecomposedPart(primitive_type: PrimitiveType, center_blender: np.ndarray, rotation_blender_3x3: np.ndarray, extents_blender: np.ndarray)`
  - `decompose_complex_mesh(mesh: MeshData, max_primitives: int = 16, min_improvement: float = 0.08) -> list[DecomposedPart]`

- [ ] **Step 1: Write failing decomposer tests**

Write `tests/test_decomposer.py`:
```python
import trimesh
from fla_blender_to_sl.decomposer import decompose_complex_mesh
from fla_blender_to_sl.ir import MaterialInfo, MeshData
from fla_blender_to_sl.schema import PrimitiveType


def _mat() -> MaterialInfo:
    return MaterialInfo(name="Default", base_color_linear=(0.5, 0.5, 0.5, 1.0))


def test_decomposer_splits_l_shape_into_multiple_boxes():
    a = trimesh.creation.box(extents=(2.0, 0.5, 1.0))
    b = trimesh.creation.box(extents=(0.5, 2.0, 1.0))
    b.apply_translation((0.75, 0.75, 0.0))
    l_mesh = trimesh.util.concatenate([a, b])

    parts = decompose_complex_mesh(MeshData(l_mesh.vertices, l_mesh.faces, _mat()), max_primitives=8)
    assert 2 <= len(parts) <= 8
    assert all(p.primitive_type == PrimitiveType.CUBE for p in parts)


def test_decomposer_respects_primitive_budget():
    # Review Focus #4: high-poly complex mesh must never exceed max_primitives
    torus = trimesh.creation.torus(major_radius=1.0, minor_radius=0.25, major_sections=64, minor_sections=32)
    parts = decompose_complex_mesh(MeshData(torus.vertices, torus.faces, _mat()), max_primitives=12)
    assert 1 <= len(parts) <= 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_decomposer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `decomposer.py`**

Implement `src/fla_blender_to_sl/decomposer.py` using iterative greedy OBB/AABB-volume-reduction splitting:
- Start with 1 cluster containing all mesh face centroids (or connected components first if multiple disconnected islands exist, up to `max_primitives`).
- While `len(clusters) < max_primitives`: pick the cluster with the largest bounding-box volume, split along its principal PCA axis at the median, and keep the split only if `child_vol_sum < parent_vol * (1.0 - min_improvement)`.
- Fit an oriented bounding box (via PCA eigenvectors with right-handed determinant `+1`) to each final cluster and emit `DecomposedPart(PrimitiveType.CUBE, center, rot_3x3, extents)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_decomposer.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/flaouve/Projects/Fla-Blender-to-SL && git add src/fla_blender_to_sl/decomposer.py tests/test_decomposer.py && git commit -m "feat: add budget-capped PCA-OBB complex mesh decomposer"
```

---

### Task 5: Deterministic Schematic Exporter & Validator

**Files:**
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/src/fla_blender_to_sl/exporter.py`
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/src/fla_blender_to_sl/validator.py`
- Test: `/home/flaouve/Projects/Fla-Blender-to-SL/tests/test_exporter_validator.py`

**Interfaces:**
- Consumes: `schema.py`, `transforms.py`, `ir.py`, `classifier.py`, `decomposer.py`
- Produces:
  - `ComplexMode(Enum)`: `AI_HANDOFF`, `DECOMPOSE`, `BBOX`
  - `ExportConfig(complex_mode: ComplexMode = ComplexMode.AI_HANDOFF, max_primitives_per_complex: int = 16, max_total_primitives: int = 2000)`
  - `HandoffEntry(placeholder_object_id: int, node_id: str, name: str, world_position_unity: Vec3, world_rotation_unity: Vec3, world_bounds_unity: Vec3, dominant_color_hex: str, triangle_count: int, suggested_budget: int)`
  - `ExportResult(schematic_name: str, schematic: SchematicObjectDataList, handoff_entries: list[HandoffEntry])`
  - `export_scene(graph: SceneGraph, config: ExportConfig = ExportConfig()) -> ExportResult`
  - `ValidationReport(valid: bool, errors: list[str], warnings: list[str], block_count: int, primitive_count: int)`
  - `validate_schematic(data: dict[str, Any], max_primitives: int = 2000) -> ValidationReport`

- [ ] **Step 1: Write failing exporter and validator tests**

Write `tests/test_exporter_validator.py`:
```python
import numpy as np
import trimesh
from fla_blender_to_sl.exporter import ComplexMode, ExportConfig, export_scene
from fla_blender_to_sl.ir import MaterialInfo, MeshData, NodeKind, SceneGraph, SceneNode
from fla_blender_to_sl.validator import validate_schematic


def _mat() -> MaterialInfo:
    return MaterialInfo(name="Concrete", base_color_linear=(0.4, 0.4, 0.4, 1.0))


def test_non_uniform_parent_scale_flattens_without_shear():
    # Review Focus #3: Parent Empty with non-uniform scale must export with scale (1,1,1)
    parent_mat = np.diag([2.0, 1.0, 3.0, 1.0])
    child_mat = np.eye(4)
    child_mat[:3, 3] = [1.0, 0.0, 0.0]
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    graph = SceneGraph(
        name="Test Room",
        nodes=[
            SceneNode("p1", "RoomRoot", NodeKind.EMPTY, None, parent_mat),
            SceneNode("c1", "Wall", NodeKind.MESH, "p1", child_mat, mesh=MeshData(box.vertices, box.faces, _mat())),
        ],
    )
    result = export_scene(graph)
    blocks = result.schematic.to_dict()["Blocks"]
    empty_b = next(b for b in blocks if b["Name"] == "RoomRoot")
    wall_b = next(b for b in blocks if b["Name"] == "Wall")

    assert empty_b["Scale"] == {"x": 1.0, "y": 1.0, "z": 1.0}
    assert wall_b["Scale"]["x"] == 2.0
    assert wall_b["Scale"]["y"] == 3.0
    assert wall_b["Position"]["x"] == 2.0


def test_exporter_emits_handoff_manifest_for_complex_meshes():
    # Review Focus #4: Complex mesh in AI_HANDOFF mode creates Empty placeholder + HandoffEntry
    torus = trimesh.creation.torus(major_radius=1.0, minor_radius=0.25)
    graph = SceneGraph(
        name="LogoMap",
        nodes=[
            SceneNode("t1", "MTF_Ring", NodeKind.MESH, None, np.eye(4), mesh=MeshData(torus.vertices, torus.faces, _mat())),
        ],
    )
    res = export_scene(graph, ExportConfig(complex_mode=ComplexMode.AI_HANDOFF))
    assert len(res.handoff_entries) == 1
    assert res.handoff_entries[0].name == "MTF_Ring"
    blocks = res.schematic.to_dict()["Blocks"]
    assert blocks[0]["BlockType"] == 0  # Empty placeholder


def test_exporter_sanitizes_names_and_clamps_zero_scale():
    # Review Focus #5: Spaces removed from schematic name; zero thickness clamped >= 0.002
    quad = trimesh.creation.box(extents=(2.0, 2.0, 0.0))
    graph = SceneGraph(
        name="My SCP Map 01",
        nodes=[
            SceneNode("q1", "Floor", NodeKind.MESH, None, np.eye(4), mesh=MeshData(quad.vertices, quad.faces, _mat())),
            SceneNode("q2", "Floor", NodeKind.MESH, None, np.eye(4), mesh=MeshData(quad.vertices, quad.faces, _mat())),
        ],
    )
    res = export_scene(graph)
    assert " " not in res.schematic_name
    report = validate_schematic(res.schematic.to_dict())
    assert report.valid
    assert report.block_count == 2
    ids = [b["ObjectId"] for b in res.schematic.to_dict()["Blocks"]]
    assert len(set(ids)) == 2
    assert all(b["Scale"]["y"] >= 0.002 for b in res.schematic.to_dict()["Blocks"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_exporter_validator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `exporter.py` and `validator.py`**

Implement `src/fla_blender_to_sl/exporter.py` (accumulating world matrices to eliminate parent non-uniform scale shear while preserving logical `Empty` grouping at `(1, 1, 1)` scale; sanitizing `schematic_name` by stripping spaces; supporting `AI_HANDOFF`, `DECOMPOSE`, and `BBOX` modes) and `src/fla_blender_to_sl/validator.py` (checking unique `ObjectId`s, valid `ParentId`s pointing to `RootObjectId` or an existing block, non-zero positive `Scale >= 0.002`, valid `Color` regex, and `primitive_count <= max_primitives`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_exporter_validator.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/flaouve/Projects/Fla-Blender-to-SL && git add src/fla_blender_to_sl/exporter.py src/fla_blender_to_sl/validator.py tests/test_exporter_validator.py && git commit -m "feat: add deterministic schematic exporter and JSON validator"
```

---

### Task 6: Standalone `.obj`/`.glb` Loader & Multi-Angle Reference Renderer

**Files:**
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/src/fla_blender_to_sl/mesh_loader.py`
- Test: `/home/flaouve/Projects/Fla-Blender-to-SL/tests/test_mesh_loader.py`

**Interfaces:**
- Consumes: `ir.py` (`SceneGraph`, `SceneNode`, `MeshData`, `MaterialInfo`)
- Produces:
  - `load_mesh_file(path: Path) -> SceneGraph` — loads `.obj`, `.glb`, `.gltf` via `trimesh`, normalizing glTF Y-Up to Blender Z-Up so the rest of the pipeline has one canonical input frame.
  - `render_multiview_sheet(mesh: MeshData, output_path: Path, tile_size: int = 256) -> Path` — renders a 3×2 labeled contact sheet (`Front`, `Back`, `Left`, `Right`, `Top`, `Iso`) as a `.png` using pure `numpy` + `Pillow` flat-shaded depth-sorted rasterizer (no OpenGL/display required).

- [ ] **Step 1: Write failing loader and renderer tests**

Write `tests/test_mesh_loader.py`:
```python
from pathlib import Path
from PIL import Image
import trimesh
from fla_blender_to_sl.mesh_loader import load_mesh_file, render_multiview_sheet


def test_load_glb_scene_preserves_multiple_objects(tmp_path: Path):
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=(1, 1, 1)), node_name="CubeA")
    cyl = trimesh.creation.cylinder(radius=0.5, height=2.0)
    cyl.apply_translation((3.0, 0.0, 0.0))
    scene.add_geometry(cyl, node_name="PillarB")

    glb_path = tmp_path / "test_map.glb"
    scene.export(glb_path)

    graph = load_mesh_file(glb_path)
    assert graph.name == "test_map"
    mesh_nodes = [n for n in graph.nodes if n.mesh is not None]
    assert len(mesh_nodes) == 2


def test_render_multiview_sheet_creates_6_view_png(tmp_path: Path):
    torus = trimesh.creation.torus(major_radius=1.0, minor_radius=0.25)
    obj_path = tmp_path / "ring.obj"
    torus.export(obj_path)

    graph = load_mesh_file(obj_path)
    out_png = tmp_path / "ring_views.png"
    render_multiview_sheet(graph.nodes[0].mesh, out_png, tile_size=128)

    assert out_png.exists()
    with Image.open(out_png) as img:
        assert img.size == (128 * 3, 128 * 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_mesh_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `mesh_loader.py`**

Implement `src/fla_blender_to_sl/mesh_loader.py`:
- `load_mesh_file`: load via `trimesh.load(path, force="scene")`, extract each geometry node's transform and PBR/diffuse base color, convert Y-Up `.glb`/`.gltf`/`.obj` into canonical Blender Z-Up `(x, -z, y)` (configurable `up_axis="Y"` default for `.glb`/`.obj`).
- `render_multiview_sheet`: painter's-algorithm orthographic/isometric triangle rasterizer in `numpy` + `Pillow` (`ImageDraw.polygon`) with simple directional diffuse shading, composited into a 3×2 grid with view labels (`Front`, `Back`, `Left`, `Right`, `Top`, `Iso`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_mesh_loader.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/flaouve/Projects/Fla-Blender-to-SL && git add src/fla_blender_to_sl/mesh_loader.py tests/test_mesh_loader.py && git commit -m "feat: add standalone obj/glb loader and software multiview renderer"
```

---

### Task 7: Complex Mesh Merger & CLI

**Files:**
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/src/fla_blender_to_sl/merger.py`
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/src/fla_blender_to_sl/cli.py`
- Test: `/home/flaouve/Projects/Fla-Blender-to-SL/tests/test_merger_cli.py`

**Interfaces:**
- Consumes: `exporter.py`, `validator.py`, `mesh_loader.py`
- Produces:
  - `merge_reconstructed_blocks(base_schematic: dict[str, Any], placeholder_id: int, replacement_schematic: dict[str, Any]) -> dict[str, Any]` — re-indexes all `ObjectId`/`ParentId` in `replacement_schematic` above `max(existing_ids)` and attaches root children to `placeholder_id`.
  - CLI `fla-blender-to-sl`:
    - `inspect <input_file>` — prints node count, native primitives vs complex meshes, estimated block count.
    - `export <input_file> -o <output_dir> [--complex-mode ai_handoff|decompose|bbox] [--max-primitives 16]` — writes `<Name>.json`, plus `handoff_manifest.json` and `handoff_renders/<node>.png` when complex meshes exist in `ai_handoff` mode.
    - `merge <base_json> --placeholder-id <id> --replacement <prop_json> -o <out_json>` — merges AI-reconstructed prop into base schematic and runs validation.
    - `validate <schematic_json> [--max-primitives 2000]` — exits `0` if valid, `1` with error list if invalid.

- [ ] **Step 1: Write failing merger and CLI tests**

Write `tests/test_merger_cli.py`:
```python
import json
from pathlib import Path
import trimesh
from fla_blender_to_sl.cli import main
from fla_blender_to_sl.merger import merge_reconstructed_blocks
from fla_blender_to_sl.validator import validate_schematic


def test_merge_reindexes_ids_and_parents_under_placeholder():
    base = {
        "RootObjectId": 1000,
        "Blocks": [
            {"Name": "Wall", "ObjectId": 1001, "ParentId": 1000, "Position": {"x": 0, "y": 0, "z": 0}, "Rotation": {"x": 0, "y": 0, "z": 0}, "Scale": {"x": 1, "y": 1, "z": 1}, "BlockType": 1, "Properties": {"PrimitiveType": 3, "Color": "FFFFFFFF", "PrimitiveFlags": 3, "Static": False}},
            {"Name": "LogoPlaceholder", "ObjectId": 1002, "ParentId": 1000, "Position": {"x": 2, "y": 1, "z": 0}, "Rotation": {"x": 0, "y": 0, "z": 0}, "Scale": {"x": 1, "y": 1, "z": 1}, "BlockType": 0, "Properties": {"Static": False}},
        ],
    }
    replacement = {
        "RootObjectId": 1000,
        "Blocks": [
            {"Name": "RingPart1", "ObjectId": 1001, "ParentId": 1000, "Position": {"x": 0, "y": 0.5, "z": 0}, "Rotation": {"x": 0, "y": 0, "z": 0}, "Scale": {"x": 0.5, "y": 0.5, "z": 0.1}, "BlockType": 1, "Properties": {"PrimitiveType": 3, "Color": "343A44FF", "PrimitiveFlags": 2, "Static": False}},
        ],
    }
    merged = merge_reconstructed_blocks(base, placeholder_id=1002, replacement_schematic=replacement)
    report = validate_schematic(merged)
    assert report.valid
    added = next(b for b in merged["Blocks"] if b["Name"] == "RingPart1")
    assert added["ObjectId"] == 1003
    assert added["ParentId"] == 1002


def test_cli_export_and_validate_end_to_end(tmp_path: Path):
    glb = tmp_path / "room.glb"
    trimesh.creation.box(extents=(4.0, 3.0, 0.2)).export(glb)
    out_dir = tmp_path / "out"

    assert main(["export", str(glb), "-o", str(out_dir), "--complex-mode", "decompose"]) == 0
    json_path = out_dir / "room.json"
    assert json_path.exists()
    assert main(["validate", str(json_path)]) == 0
    data = json.loads(json_path.read_text())
    assert data["RootObjectId"] == 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_merger_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `merger.py` and `cli.py`**

Implement `src/fla_blender_to_sl/merger.py` and `src/fla_blender_to_sl/cli.py` using `argparse`, returning integer exit codes (`0` success, `1` validation/input error).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_merger_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/flaouve/Projects/Fla-Blender-to-SL && git add src/fla_blender_to_sl/merger.py src/fla_blender_to_sl/cli.py tests/test_merger_cli.py && git commit -m "feat: add complex mesh merger and fla-blender-to-sl CLI"
```

---

### Task 8: Blender 4.2+ Extension Addon

**Files:**
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/blender_addon/fla_blender_to_sl_addon/blender_manifest.toml`
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/blender_addon/fla_blender_to_sl_addon/scene_extract.py`
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/blender_addon/fla_blender_to_sl_addon/operators.py`
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/blender_addon/fla_blender_to_sl_addon/ui.py`
- Create: `/home/flaouve/Projects/Fla-Blender-to-SL/blender_addon/fla_blender_to_sl_addon/__init__.py`
- Test: `/home/flaouve/Projects/Fla-Blender-to-SL/tests/test_addon_extract.py`

**Interfaces:**
- Consumes: `ir.py`, `exporter.py`, `validator.py`, `mesh_loader.py`
- Produces:
  - `extract_blender_objects(objects: Iterable[Any], scene_name: str) -> SceneGraph` — pure adapter reading `.name`, `.type` (`MESH`, `LIGHT`, `FONT`, `EMPTY`), `.parent`, `.matrix_local`, evaluated mesh vertices/polygons, Principled BSDF `Base Color` / `Emission Strength`, and custom properties (`sl_collidable`, `sl_visible`, `sl_static`, `sl_export_override`).
  - Blender UI panel `VIEW3D_PT_fla_sl_export` (N-panel → `SCP:SL` tab) with per-object flags and `Export to ProjectMER (.json)` + `Export Hybrid Handoff` operators.

- [ ] **Step 1: Write failing Blender extraction adapter test**

Write `tests/test_addon_extract.py` using lightweight stub objects mimicking `bpy.types.Object` (`MESH`, `LIGHT`, `FONT`, `EMPTY`) so extraction runs in `pytest` without requiring `bpy`:
```python
from types import SimpleNamespace
import numpy as np
from blender_addon.fla_blender_to_sl_addon.scene_extract import extract_blender_objects
from fla_blender_to_sl.exporter import export_scene
from fla_blender_to_sl.validator import validate_schematic


def _mock_cube_obj(name: str):
    verts = [SimpleNamespace(co=c) for c in [
        (-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
        (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1),
    ]]
    polys = [SimpleNamespace(vertices=f) for f in [
        (0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(0,3,7,4),
    ]]
    return SimpleNamespace(
        name=name,
        type="MESH",
        parent=None,
        matrix_local=np.eye(4),
        data=SimpleNamespace(vertices=verts, polygons=polys, materials=[]),
        sl_collidable=True,
        sl_visible=True,
        sl_static=True,
        sl_export_override="AUTO",
    )


def test_extract_blender_objects_and_export_valid_schematic():
    cube = _mock_cube_obj("CorridorWall")
    light = SimpleNamespace(
        name="HallLight",
        type="LIGHT",
        parent=None,
        matrix_local=np.eye(4),
        data=SimpleNamespace(type="POINT", color=(1.0, 0.9, 0.8), energy=50.0, shadow_soft_size=0.2, spot_size=0.52),
        sl_collidable=False,
        sl_visible=True,
        sl_static=False,
        sl_export_override="AUTO",
    )
    graph = extract_blender_objects([cube, light], scene_name="LightContainment")
    res = export_scene(graph)
    report = validate_schematic(res.schematic.to_dict())
    assert report.valid
    assert report.block_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_addon_extract.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `blender_manifest.toml`, `scene_extract.py`, `operators.py`, `ui.py`, `__init__.py`**

Keep `scene_extract.py` free of top-level `import bpy` so it is 100% unit-testable; put `bpy` UI/operator registration in `operators.py`, `ui.py`, and `__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest tests/test_addon_extract.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/flaouve/Projects/Fla-Blender-to-SL && git add blender_addon/ tests/test_addon_extract.py && git commit -m "feat: add Blender 4.2+ extension addon and scene extractor"
```

---

### Task 9: Hybrid AI Reconstruction Skill (`blender-to-sl`)

**Files:**
- Create: `/home/flaouve/.gemini/config/skills/blender-to-sl/SKILL.md`

**Interfaces:**
- Consumes: `fla-blender-to-sl` CLI (`inspect`, `export --complex-mode ai_handoff`, `merge`, `validate`) and the existing `scp-sl-prop-generator` visual-preview & cold SCP palette rules.
- Produces: End-to-end AI workflow for converting any `.blend` / `.obj` / `.glb` model or map into a verified `Fla-SL-CustomObjects` / `FlaProjectMER` `.json` schematic.

- [ ] **Step 1: Write `/home/flaouve/.gemini/config/skills/blender-to-sl/SKILL.md`**

Include valid YAML frontmatter (`name: blender-to-sl`, `description: ...`) and 5 deterministic phases:
1. **Inspect & Classify:** Run `fla-blender-to-sl inspect <file>` to see native vs complex counts.
2. **Deterministic Base Export:** Run `fla-blender-to-sl export <file> -o <out_dir> --complex-mode ai_handoff`. If `handoff_manifest.json` is empty, stop — the `.json` is already 100% complete and validated!
3. **Visual Reference Inspection:** For each entry in `handoff_manifest.json`, inspect `handoff_renders/<node>.png` using `read_file` alongside exact `world_bounds_unity`, `dominant_color_hex`, and `suggested_budget`.
4. **Silhouette-First Primitive Reconstruction:** Write a Python generator script that builds the complex mesh within `suggested_budget` clean Unity primitives (no triangle soup), renders a matching 6-view comparison `.png`, visually verifies likeness/grounding/symmetry, and exports `<node>_reconstructed.json`.
5. **Merge & Final Audit:** Run `fla-blender-to-sl merge` for each placeholder ID and `fla-blender-to-sl validate` on the final schematic `.json`.

- [ ] **Step 2: Run full test suite and verify skill frontmatter**

Run: `cd /home/flaouve/Projects/Fla-Blender-to-SL && uv run pytest -v && head -n 6 /home/flaouve/.gemini/config/skills/blender-to-sl/SKILL.md`
Expected: All pytest tests PASS; `SKILL.md` starts with valid `---` frontmatter containing `name: blender-to-sl`.

- [ ] **Step 3: Commit final plan verification notes**

```bash
cd /home/flaouve/Projects/Fla-Blender-to-SL && git status
```
