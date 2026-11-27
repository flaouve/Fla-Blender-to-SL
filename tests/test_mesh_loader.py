from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from fla_blender_to_sl.ir import MaterialInfo, MeshData, NodeKind
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
    assert all(n.kind == NodeKind.MESH for n in mesh_nodes)


def test_load_glb_converts_y_up_to_blender_z_up(tmp_path: Path):
    # A tall box in Y-Up: width(X)=1, height(Y)=4, depth(Z)=2
    # In Blender Z-Up (x, -z, y): X=1, Y=2, Z=4
    box = trimesh.creation.box(extents=(1.0, 4.0, 2.0))
    box.apply_translation((0.0, 2.0, 0.0))  # sits on Y=0..4
    glb_path = tmp_path / "tall_box.glb"
    box.export(glb_path)

    graph = load_mesh_file(glb_path)
    assert len(graph.nodes) == 1
    node = graph.nodes[0]
    assert node.mesh is not None

    # Transform local vertices by node.local_matrix_blender to get Blender world coords
    verts = node.mesh.vertices
    ones = np.ones((len(verts), 1), dtype=float)
    world_verts = (node.local_matrix_blender @ np.hstack([verts, ones]).T).T[:, :3]
    extents = world_verts.max(axis=0) - world_verts.min(axis=0)
    assert np.allclose(extents, (1.0, 2.0, 4.0), atol=1e-4)
    assert np.isclose(world_verts[:, 2].min(), 0.0, atol=1e-4)
    assert np.isclose(world_verts[:, 2].max(), 4.0, atol=1e-4)


def test_load_obj_respects_z_up_axis_option(tmp_path: Path):
    box = trimesh.creation.box(extents=(1.0, 2.0, 4.0))
    obj_path = tmp_path / "zup_box.obj"
    box.export(obj_path)

    graph = load_mesh_file(obj_path, up_axis="Z")
    assert len(graph.nodes) == 1
    verts = graph.nodes[0].mesh.vertices
    extents = verts.max(axis=0) - verts.min(axis=0)
    assert np.allclose(extents, (1.0, 2.0, 4.0), atol=1e-4)


def test_render_multiview_sheet_creates_6_view_png(tmp_path: Path):
    torus = trimesh.creation.torus(major_radius=1.0, minor_radius=0.25)
    obj_path = tmp_path / "ring.obj"
    torus.export(obj_path)

    graph = load_mesh_file(obj_path)
    out_png = tmp_path / "ring_views.png"
    result_path = render_multiview_sheet(graph.nodes[0].mesh, out_png, tile_size=128)

    assert result_path == out_png
    assert out_png.exists()
    with Image.open(out_png) as img:
        assert img.size == (128 * 3, 128 * 2)
        arr = np.asarray(img)
        # Ensure the image is not a uniform solid color (geometry was actually rasterized)
        assert arr.std() > 5.0


def test_render_multiview_sheet_handles_empty_mesh(tmp_path: Path):
    empty_mesh = MeshData(
        vertices=np.zeros((0, 3), dtype=float),
        faces=np.zeros((0, 3), dtype=int),
        material=MaterialInfo(name="Empty", base_color_linear=(0.5, 0.5, 0.5, 1.0)),
    )
    out_png = tmp_path / "nested" / "empty.png"
    result_path = render_multiview_sheet(empty_mesh, out_png, tile_size=64)

    assert result_path == out_png
    assert out_png.exists()
    with Image.open(out_png) as img:
        assert img.size == (64 * 3, 64 * 2)


def test_render_multiview_numpy_fallback_without_pillow(tmp_path: Path):
    from fla_blender_to_sl.mesh_loader import _render_multiview_numpy

    box = trimesh.creation.box(extents=(1.0, 2.0, 1.5))
    mesh = MeshData(
        vertices=np.asarray(box.vertices, dtype=float),
        faces=np.asarray(box.faces, dtype=int),
        material=MaterialInfo(name="BoxMat", base_color_linear=(0.4, 0.45, 0.5, 1.0)),
    )
    out_png = tmp_path / "fallback.png"
    _render_multiview_numpy(mesh, out_png, tile_size=64)

    assert out_png.exists()
    with Image.open(out_png) as img:
        assert img.size == (64 * 3, 64 * 2)
        assert np.asarray(img).std() > 5.0
