"""Pure math for Blender RH Z-Up -> Unity LH Y-Up transforms and SCP:SL color encoding."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from fla_blender_to_sl.schema import Vec3, _clean_float

MIN_SCALE: float = 0.002

# Change-of-basis matrix swapping Y and Z: (bx, by, bz) -> (bx, bz, by)
_BASIS_SWAP_3X3 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)

_BASIS_SWAP_4X4 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def _normalize_angle_deg(degrees: float) -> float:
    """Normalize an angle in degrees to [0.0, 360.0), avoiding -0.0 and 360.0."""
    angle = float(degrees) % 360.0
    if abs(angle) < 1e-9 or abs(angle - 360.0) < 1e-9:
        return 0.0
    return angle


def blender_pos_to_unity(pos: Sequence[float]) -> Vec3:
    """Convert a Blender RH Z-Up position (bx, by, bz) to Unity LH Y-Up (bx, bz, by)."""
    return Vec3(float(pos[0]), float(pos[2]), float(pos[1]))


def blender_scale_to_unity(
    scale: Sequence[float],
    min_scale: float = MIN_SCALE,
) -> Vec3:
    """Convert a Blender scale (sx, sy, sz) to positive Unity scale (sx, sz, sy) >= min_scale."""
    ux = max(abs(float(scale[0])), min_scale)
    uy = max(abs(float(scale[2])), min_scale)
    uz = max(abs(float(scale[1])), min_scale)
    return Vec3(ux, uy, uz)


def unity_euler_to_matrix(euler_deg: Vec3) -> np.ndarray:
    """Build Unity intrinsic Z-X-Y rotation matrix R = R_y @ R_x @ R_z from degrees."""
    rx = math.radians(euler_deg.x)
    ry = math.radians(euler_deg.y)
    rz = math.radians(euler_deg.z)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    r_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=float,
    )
    r_y = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=float,
    )
    r_z = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return r_y @ r_x @ r_z


def _matrix_to_unity_euler_deg(rot_3x3: np.ndarray) -> Vec3:
    """Extract Unity intrinsic Z-X-Y (R = R_y @ R_x @ R_z) Euler angles in degrees [0, 360)."""
    sin_x = float(np.clip(-rot_3x3[1, 2], -1.0, 1.0))
    x_rad = math.asin(sin_x)
    cos_x = math.cos(x_rad)

    if abs(cos_x) > 1e-6:
        y_rad = math.atan2(float(rot_3x3[0, 2]), float(rot_3x3[2, 2]))
        z_rad = math.atan2(float(rot_3x3[1, 0]), float(rot_3x3[1, 1]))
    else:
        y_rad = math.atan2(float(-rot_3x3[2, 0]), float(rot_3x3[0, 0]))
        z_rad = 0.0

    return Vec3(
        _normalize_angle_deg(math.degrees(x_rad)),
        _normalize_angle_deg(math.degrees(y_rad)),
        _normalize_angle_deg(math.degrees(z_rad)),
    )


def blender_matrix_to_unity_trs(
    matrix_4x4: np.ndarray,
    min_scale: float = MIN_SCALE,
) -> tuple[Vec3, Vec3, Vec3]:
    """Decompose a Blender 4x4 transform into Unity (position, euler_deg_zxy, positive_scale)."""
    m_b = np.asarray(matrix_4x4, dtype=float)
    if m_b.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform matrix, got shape {m_b.shape}")

    m_u = _BASIS_SWAP_4X4 @ m_b @ _BASIS_SWAP_4X4
    position = Vec3(float(m_u[0, 3]), float(m_u[1, 3]), float(m_u[2, 3]))

    linear = m_u[:3, :3]
    col_norms = np.linalg.norm(linear, axis=0)
    scale = Vec3(
        max(float(col_norms[0]), min_scale),
        max(float(col_norms[1]), min_scale),
        max(float(col_norms[2]), min_scale),
    )

    # Normalize non-zero columns to extract rotation
    rot_raw = np.eye(3, dtype=float)
    for col_idx in range(3):
        if col_norms[col_idx] > 1e-12:
            rot_raw[:, col_idx] = linear[:, col_idx] / col_norms[col_idx]

    # If reflected (negative/mirrored Blender scale), flip first axis to restore det > 0
    if np.linalg.det(rot_raw) < 0.0:
        rot_raw[:, 0] *= -1.0

    # Polar decomposition via SVD ensures exact orthonormal proper rotation (det == +1)
    u_mat, _, vt_mat = np.linalg.svd(rot_raw)
    rot_clean = u_mat @ vt_mat
    if np.linalg.det(rot_clean) < 0.0:
        u_mat[:, -1] *= -1.0
        rot_clean = u_mat @ vt_mat

    euler_deg = _matrix_to_unity_euler_deg(rot_clean)
    return position, euler_deg, scale


def _linear_to_srgb_channel(value: float) -> float:
    """IEC 61966-2-1 linear to sRGB transfer function for a non-negative channel."""
    if value <= 0.0:
        return 0.0
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def linear_rgba_to_sl_color(
    rgba: Sequence[float],
    emission_strength: float = 0.0,
) -> str:
    """Encode linear RGBA into Fla-SL-CustomObjects LDR 'RRGGBBAA' hex or HDR 'R:G:B:A'."""
    r_lin = max(float(rgba[0]), 0.0)
    g_lin = max(float(rgba[1]), 0.0)
    b_lin = max(float(rgba[2]), 0.0)
    a_lin = float(np.clip(rgba[3] if len(rgba) > 3 else 1.0, 0.0, 1.0))

    is_hdr = emission_strength > 1.0 or any(c > 1.0 for c in (r_lin, g_lin, b_lin))
    if is_hdr:
        multiplier = float(emission_strength) if emission_strength > 1.0 else 1.0
        r_hdr = _clean_float(_linear_to_srgb_channel(r_lin) * multiplier * 255.0, ndigits=4)
        g_hdr = _clean_float(_linear_to_srgb_channel(g_lin) * multiplier * 255.0, ndigits=4)
        b_hdr = _clean_float(_linear_to_srgb_channel(b_lin) * multiplier * 255.0, ndigits=4)
        a_hdr = _clean_float(a_lin, ndigits=4)
        return f"{r_hdr}:{g_hdr}:{b_hdr}:{a_hdr}"

    r_byte = int(min(max(round(_linear_to_srgb_channel(r_lin) * 255.0), 0), 255))
    g_byte = int(min(max(round(_linear_to_srgb_channel(g_lin) * 255.0), 0), 255))
    b_byte = int(min(max(round(_linear_to_srgb_channel(b_lin) * 255.0), 0), 255))
    a_byte = int(min(max(round(a_lin * 255.0), 0), 255))
    return f"{r_byte:02X}{g_byte:02X}{b_byte:02X}{a_byte:02X}"
