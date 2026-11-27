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
