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
