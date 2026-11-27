import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from blender_addon.fla_blender_to_sl_addon.scene_extract import extract_blender_objects
from fla_blender_to_sl.exporter import ComplexMode, ExportConfig, export_scene
from fla_blender_to_sl.ir import ExportOverride, NodeKind
from fla_blender_to_sl.schema import LightType, ShadowType
from fla_blender_to_sl.validator import validate_schematic


def _mock_cube_obj(name: str, parent=None, materials=None, override: str = "AUTO"):
    verts = [
        SimpleNamespace(co=c)
        for c in [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ]
    ]
    polys = [
        SimpleNamespace(vertices=f)
        for f in [
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (1, 2, 6, 5),
            (3, 0, 4, 7),
        ]
    ]
    return SimpleNamespace(
        name=name,
        type="MESH",
        parent=parent,
        matrix_local=np.eye(4),
        data=SimpleNamespace(vertices=verts, polygons=polys, materials=materials or []),
        sl_collidable=True,
        sl_visible=True,
        sl_static=True,
        sl_export_override=override,
    )


def test_extract_blender_objects_and_export_valid_schematic():
    cube = _mock_cube_obj("CorridorWall")
    light = SimpleNamespace(
        name="HallLight",
        type="LIGHT",
        parent=None,
        matrix_local=np.eye(4),
        data=SimpleNamespace(
            type="POINT",
            color=(1.0, 0.9, 0.8),
            energy=50.0,
            shadow_soft_size=0.2,
            spot_size=0.52,
        ),
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


def test_extract_hierarchy_text_spot_light_and_principled_bsdf():
    root_empty = SimpleNamespace(
        name="RoomRoot",
        type="EMPTY",
        parent=None,
        matrix_local=np.eye(4),
        data=None,
        sl_collidable=False,
        sl_visible=True,
        sl_static=True,
        sl_export_override="AUTO",
    )

    bsdf_node = SimpleNamespace(
        type="BSDF_PRINCIPLED",
        inputs={
            "Base Color": SimpleNamespace(default_value=(0.2, 0.3, 0.4, 1.0)),
            "Emission Color": SimpleNamespace(default_value=(0.9, 0.1, 0.1, 1.0)),
            "Emission Strength": SimpleNamespace(default_value=2.5),
        },
    )
    mat = SimpleNamespace(
        name="AlertPanel",
        use_nodes=True,
        node_tree=SimpleNamespace(nodes=[bsdf_node]),
        diffuse_color=(0.5, 0.5, 0.5, 1.0),
    )

    panel = _mock_cube_obj("AlertSign", parent=root_empty, materials=[mat], override="FORCE_AI_HANDOFF")

    sign_text = SimpleNamespace(
        name="ZoneLabel",
        type="FONT",
        parent=root_empty,
        matrix_local=np.eye(4),
        data=SimpleNamespace(
            body="LCZ-01",
            size=1.0,
            align_x="CENTER",
            materials=[],
        ),
        sl_collidable=False,
        sl_visible=True,
        sl_static=True,
        sl_export_override="AUTO",
    )

    spot = SimpleNamespace(
        name="SpotLamp",
        type="LIGHT",
        parent=root_empty,
        matrix_local=np.eye(4),
        data=SimpleNamespace(
            type="SPOT",
            color=(1.0, 1.0, 1.0),
            energy=100.0,
            use_shadow=True,
            shadow_soft_size=0.1,
            spot_size=np.radians(45.0),
        ),
        sl_collidable=False,
        sl_visible=True,
        sl_static=False,
        sl_export_override="AUTO",
    )

    graph = extract_blender_objects([root_empty, panel, sign_text, spot], scene_name="GateA")
    assert graph.name == "GateA"
    assert len(graph.nodes) == 4

    by_name = {n.name: n for n in graph.nodes}
    assert by_name["RoomRoot"].kind == NodeKind.EMPTY
    assert by_name["AlertSign"].parent_id == by_name["RoomRoot"].id
    assert by_name["AlertSign"].override == ExportOverride.FORCE_AI_HANDOFF
    assert by_name["AlertSign"].mesh is not None
    assert by_name["AlertSign"].mesh.material.name == "AlertPanel"
    assert by_name["AlertSign"].mesh.material.emission_strength == 2.5
    assert by_name["ZoneLabel"].kind == NodeKind.TEXT
    assert by_name["ZoneLabel"].text is not None
    assert by_name["ZoneLabel"].text.text == "LCZ-01"
    assert by_name["SpotLamp"].kind == NodeKind.LIGHT
    assert by_name["SpotLamp"].light is not None
    assert by_name["SpotLamp"].light.light_type == LightType.SPOT
    assert by_name["SpotLamp"].light.shadow_type == ShadowType.SOFT
    assert np.isclose(by_name["SpotLamp"].light.spot_angle, 45.0, atol=1e-3)

    res = export_scene(graph, ExportConfig(complex_mode=ComplexMode.AI_HANDOFF))
    assert len(res.handoff_entries) == 1
    report = validate_schematic(res.schematic.to_dict())
    assert report.valid


def test_blender_manifest_exists_and_targets_4_2():
    manifest = Path(__file__).resolve().parents[1] / "blender_addon" / "fla_blender_to_sl_addon" / "blender_manifest.toml"
    assert manifest.exists()
    content = manifest.read_text(encoding="utf-8")
    assert 'schema_version = "1.0.0"' in content
    assert 'type = "add-on"' in content
    assert 'blender_version_min = "4.2.0"' in content


def test_extract_texture_painted_logo_renders_into_handoff_sheet(tmp_path: Path):
    from PIL import Image
    from fla_blender_to_sl.mesh_loader import render_multiview_sheet

    # 2x2 texture: left half red, right half blue
    pixels = [
        1.0, 0.0, 0.0, 1.0,  0.0, 0.0, 1.0, 1.0,
        1.0, 0.0, 0.0, 1.0,  0.0, 0.0, 1.0, 1.0,
    ]
    tex_node = SimpleNamespace(
        type="TEX_IMAGE",
        image=SimpleNamespace(size=(2, 2), pixels=pixels),
    )
    mat = SimpleNamespace(
        name="PaintedLogoMat",
        use_nodes=True,
        node_tree=SimpleNamespace(nodes=[tex_node]),
        diffuse_color=(1.0, 1.0, 1.0, 1.0),
    )
    verts = [SimpleNamespace(co=c) for c in [(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)]]
    polys = [SimpleNamespace(vertices=(0, 1, 2, 3), loop_indices=(0, 1, 2, 3))]
    uv_data = [
        SimpleNamespace(uv=(0.0, 0.0)),
        SimpleNamespace(uv=(1.0, 0.0)),
        SimpleNamespace(uv=(1.0, 1.0)),
        SimpleNamespace(uv=(0.0, 1.0)),
    ]
    logo_obj = SimpleNamespace(
        name="PaintedLogo",
        type="MESH",
        parent=None,
        matrix_local=np.eye(4),
        data=SimpleNamespace(
            vertices=verts,
            polygons=polys,
            materials=[mat],
            uv_layers=SimpleNamespace(active=SimpleNamespace(data=uv_data)),
        ),
        sl_collidable=False,
        sl_visible=True,
        sl_static=True,
        sl_export_override="FORCE_AI_HANDOFF",
    )

    graph = extract_blender_objects([logo_obj], scene_name="LogoTest")
    mesh = graph.nodes[0].mesh
    assert mesh is not None
    assert mesh.texture_srgb is not None
    assert mesh.face_uvs is not None and mesh.face_uvs.shape == (2, 3, 2)

    out_png = tmp_path / "painted_logo.png"
    render_multiview_sheet(mesh, out_png, tile_size=64)
    arr = np.asarray(Image.open(out_png))
    # Verify both red-dominant and blue-dominant pixels appear in the rendered sheet
    assert np.any((arr[:, :, 0] > 120) & (arr[:, :, 2] < 40))
    assert np.any((arr[:, :, 2] > 120) & (arr[:, :, 0] < 40))
