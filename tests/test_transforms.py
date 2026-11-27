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
