"""Geometric classifier mapping MeshData to native Unity/SCP:SL primitives or COMPLEX."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fla_blender_to_sl.ir import ExportOverride, MeshData
from fla_blender_to_sl.schema import PrimitiveType
from fla_blender_to_sl.transforms import MIN_SCALE


@dataclass(frozen=True)
class ClassificationResult:
    """Result of classifying a mesh against native Unity primitives."""

    primitive_type: PrimitiveType | None
    local_center_blender: np.ndarray
    unity_scale: tuple[float, float, float]
    confidence: float
    is_complex: bool


def _clamp_unity_scale(dx: float, dy: float, dz: float, *, height_divisor: float = 1.0) -> tuple[float, float, float]:
    """Map Blender extents (dx, dy, dz) to Unity scale (dx, dz / height_divisor, dy) >= MIN_SCALE."""
    return (
        max(float(dx), MIN_SCALE),
        max(float(dz) / height_divisor, MIN_SCALE),
        max(float(dy), MIN_SCALE),
    )


def classify_mesh(
    mesh: MeshData,
    world_scale_blender: tuple[float, float, float] = (1.0, 1.0, 1.0),
    override: ExportOverride = ExportOverride.AUTO,
) -> ClassificationResult:
    """Classify MeshData into a native Unity primitive or mark it as COMPLEX."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] != 3:
        return ClassificationResult(
            primitive_type=None,
            local_center_blender=np.zeros(3, dtype=float),
            unity_scale=(MIN_SCALE, MIN_SCALE, MIN_SCALE),
            confidence=0.0,
            is_complex=True,
        )

    v_min_local = vertices.min(axis=0)
    v_max_local = vertices.max(axis=0)
    local_center_blender = 0.5 * (v_min_local + v_max_local)

    scale_vec = np.abs(np.asarray(world_scale_blender, dtype=float))
    pts = (vertices - local_center_blender) * scale_vec
    extents = pts.max(axis=0) - pts.min(axis=0)
    dx, dy, dz = float(extents[0]), float(extents[1]), float(extents[2])

    default_unity_scale = _clamp_unity_scale(dx, dy, dz)

    if override == ExportOverride.FORCE_AI_HANDOFF:
        return ClassificationResult(
            primitive_type=None,
            local_center_blender=local_center_blender,
            unity_scale=default_unity_scale,
            confidence=0.0,
            is_complex=True,
        )
    if override == ExportOverride.FORCE_CUBE:
        return ClassificationResult(
            primitive_type=PrimitiveType.CUBE,
            local_center_blender=local_center_blender,
            unity_scale=default_unity_scale,
            confidence=1.0,
            is_complex=False,
        )
    if override == ExportOverride.FORCE_CYLINDER:
        return ClassificationResult(
            primitive_type=PrimitiveType.CYLINDER,
            local_center_blender=local_center_blender,
            unity_scale=_clamp_unity_scale(dx, dy, dz, height_divisor=2.0),
            confidence=1.0,
            is_complex=False,
        )
    if override == ExportOverride.FORCE_SPHERE:
        return ClassificationResult(
            primitive_type=PrimitiveType.SPHERE,
            local_center_blender=local_center_blender,
            unity_scale=default_unity_scale,
            confidence=1.0,
            is_complex=False,
        )

    # Non-uniform image-textured / texture-painted surfaces (e.g. painted signs/logos)
    # cannot be represented by a single solid-colored Unity primitive; route to AI handoff.
    if (
        mesh.texture_srgb is not None
        and mesh.face_uvs is not None
        and mesh.texture_srgb.size > 0
        and float(mesh.texture_srgb.std()) > 5.0
    ):
        return ClassificationResult(
            primitive_type=None,
            local_center_blender=local_center_blender,
            unity_scale=default_unity_scale,
            confidence=0.0,
            is_complex=True,
        )

    # Check planar Quad (one axis extent < 1e-4 and 4-8 unique planar vertices)
    thin_axes = int(np.sum(extents < 1e-4))
    if thin_axes == 1:
        unique_verts = np.unique(np.round(pts, decimals=5), axis=0)
        if 4 <= len(unique_verts) <= 8:
            return ClassificationResult(
                primitive_type=PrimitiveType.QUAD,
                local_center_blender=local_center_blender,
                unity_scale=default_unity_scale,
                confidence=0.95,
                is_complex=False,
            )

    if thin_axes >= 1 or dx < 1e-6 or dy < 1e-6 or dz < 1e-6 or len(faces) == 0:
        return ClassificationResult(
            primitive_type=None,
            local_center_blender=local_center_blender,
            unity_scale=default_unity_scale,
            confidence=0.0,
            is_complex=True,
        )

    tris = pts[faces]
    mesh_vol = abs(float(np.sum(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2]))) / 6.0))
    bbox_vol = dx * dy * dz
    half = extents * 0.5

    # 1. Cube check: all vertices lie on AABB boundary planes and volume ratio > 0.96
    norm_abs = np.abs(pts) / half
    dist_to_aabb = np.abs(1.0 - norm_abs.max(axis=1))
    vol_ratio_box = mesh_vol / bbox_vol if bbox_vol > 0.0 else 0.0
    if float(dist_to_aabb.max()) < 0.02 and vol_ratio_box > 0.96:
        return ClassificationResult(
            primitive_type=PrimitiveType.CUBE,
            local_center_blender=local_center_blender,
            unity_scale=default_unity_scale,
            confidence=min(1.0, vol_ratio_box),
            is_complex=False,
        )

    # 2. Sphere / Ellipsoid check: normalized radius ||p / half|| ~ 1.0 with low std
    if len(vertices) >= 12:
        radii = np.linalg.norm(pts / half, axis=1)
        r_std = float(np.std(radii))
        r_mean = float(np.mean(radii))
        expected_sphere_vol = (4.0 / 3.0) * math.pi * half[0] * half[1] * half[2]
        vol_ratio_sphere = mesh_vol / expected_sphere_vol if expected_sphere_vol > 0.0 else 0.0
        if r_std < 0.03 and abs(r_mean - 1.0) < 0.03 and 0.85 <= vol_ratio_sphere <= 1.05:
            # Recover exact diameter for isotropic spheres (avoids UV-sphere equator tessellation sag)
            iso_radii = np.linalg.norm(pts, axis=1)
            iso_mean = float(np.mean(iso_radii))
            if iso_mean > 0.0 and float(np.std(iso_radii)) / iso_mean < 0.02:
                diam = max(2.0 * iso_mean, MIN_SCALE)
                sphere_scale = (diam, diam, diam)
            else:
                sphere_scale = default_unity_scale
            return ClassificationResult(
                primitive_type=PrimitiveType.SPHERE,
                local_center_blender=local_center_blender,
                unity_scale=sphere_scale,
                confidence=max(0.0, min(1.0, 1.0 - r_std)),
                is_complex=False,
            )

    # 3. Cylinder check (Z-aligned in Blender -> Y-aligned in Unity, height divided by 2.0)
    if len(vertices) >= 12:
        r_xy = np.sqrt((pts[:, 0] / half[0]) ** 2 + (pts[:, 1] / half[1]) ** 2)
        z_norm = np.abs(pts[:, 2]) / half[2]
        cap_interior = r_xy <= 0.5
        cap_ok = bool(np.all(z_norm[cap_interior] > 0.98)) if np.any(cap_interior) else True
        rim_mask = ~cap_interior
        if cap_ok and np.any(rim_mask):
            rim_std = float(np.std(r_xy[rim_mask]))
            rim_mean = float(np.mean(r_xy[rim_mask]))
            expected_cyl_vol = math.pi * half[0] * half[1] * dz
            vol_ratio_cyl = mesh_vol / expected_cyl_vol if expected_cyl_vol > 0.0 else 0.0
            if rim_std < 0.03 and abs(rim_mean - 1.0) < 0.03 and 0.90 <= vol_ratio_cyl <= 1.05:
                return ClassificationResult(
                    primitive_type=PrimitiveType.CYLINDER,
                    local_center_blender=local_center_blender,
                    unity_scale=_clamp_unity_scale(dx, dy, dz, height_divisor=2.0),
                    confidence=min(1.0, vol_ratio_cyl),
                    is_complex=False,
                )

    return ClassificationResult(
        primitive_type=None,
        local_center_blender=local_center_blender,
        unity_scale=default_unity_scale,
        confidence=0.0,
        is_complex=True,
    )
