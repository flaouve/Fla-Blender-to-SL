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


def test_textured_cube_auto_classifies_as_complex():
    box = trimesh.creation.box(extents=(2.95, 0.025, 1.75))
    tex = np.zeros((4, 4, 3), dtype=np.uint8)
    tex[:, :2] = (230, 230, 230)
    tex[:, 2:] = (20, 25, 30)
    face_uvs = np.zeros((len(box.faces), 3, 2), dtype=float)
    res = classify_mesh(MeshData(box.vertices, box.faces, _mat(), face_uvs=face_uvs, texture_srgb=tex))
    assert res.is_complex
    assert res.primitive_type is None
