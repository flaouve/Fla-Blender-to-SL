"""Budget-capped deterministic PCA-OBB decomposer for complex meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fla_blender_to_sl.ir import MeshData
from fla_blender_to_sl.schema import PrimitiveType
from fla_blender_to_sl.transforms import MIN_SCALE


def _connected_face_components(faces: np.ndarray, num_vertices: int) -> list[np.ndarray]:
    """Find vertex-connected face components using pure-Python Union-Find (no scipy/networkx)."""
    parent = list(range(num_vertices))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for face in faces:
        v0 = int(face[0])
        for v in face[1:]:
            union(v0, int(v))

    groups: dict[int, list[int]] = {}
    for face_idx, face in enumerate(faces):
        root = find(int(face[0]))
        groups.setdefault(root, []).append(face_idx)

    return [faces[indices] for indices in groups.values()]


@dataclass(frozen=True)
class DecomposedPart:
    """Single oriented primitive box produced by complex mesh decomposition."""

    primitive_type: PrimitiveType
    center_blender: np.ndarray
    rotation_blender_3x3: np.ndarray
    extents_blender: np.ndarray


def _fit_obb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a right-handed PCA oriented bounding box to a 3D point set."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros(3, dtype=float), np.eye(3, dtype=float), np.full(3, MIN_SCALE, dtype=float), MIN_SCALE ** 3

    mean = pts.mean(axis=0)
    centered = pts - mean
    if len(pts) >= 3:
        cov = (centered.T @ centered) / float(len(pts))
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        axes = eigvecs[:, order].copy()
        if np.linalg.det(axes) < 0.0:
            axes[:, 2] *= -1.0
    else:
        axes = np.eye(3, dtype=float)

    proj = pts @ axes
    p_min = proj.min(axis=0)
    p_max = proj.max(axis=0)
    extents = np.maximum(p_max - p_min, MIN_SCALE)
    center_local = 0.5 * (p_min + p_max)
    center_world = axes @ center_local
    volume = float(extents[0] * extents[1] * extents[2])
    return center_world, axes, extents, volume


@dataclass
class _Cluster:
    vertices: np.ndarray
    faces: np.ndarray
    center: np.ndarray
    axes: np.ndarray
    extents: np.ndarray
    volume: float
    splittable: bool = True


def _make_cluster(vertices: np.ndarray, faces: np.ndarray) -> _Cluster:
    if len(faces) > 0:
        pts = vertices[np.unique(faces.reshape(-1))]
    else:
        pts = vertices
    center, axes, extents, volume = _fit_obb(pts)
    return _Cluster(
        vertices=vertices,
        faces=faces,
        center=center,
        axes=axes,
        extents=extents,
        volume=volume,
        splittable=len(faces) >= 2,
    )


def decompose_complex_mesh(
    mesh: MeshData,
    max_primitives: int = 16,
    min_improvement: float = 0.08,
) -> list[DecomposedPart]:
    """Approximate a complex mesh with 1..max_primitives oriented bounding boxes (CUBE)."""
    budget = max(1, int(max_primitives))
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] != 3:
        return []

    clusters: list[_Cluster] = []
    if len(faces) > 0:
        components = _connected_face_components(faces, len(vertices))
        if 1 <= len(components) <= budget:
            for comp_faces in components:
                clusters.append(_make_cluster(vertices, comp_faces))
        else:
            clusters.append(_make_cluster(vertices, faces))
    else:
        clusters.append(_make_cluster(vertices, faces))

    while len(clusters) < budget:
        candidates = [idx for idx, c in enumerate(clusters) if c.splittable and len(c.faces) >= 2]
        if not candidates:
            break

        best_idx = max(candidates, key=lambda idx: clusters[idx].volume)
        parent = clusters[best_idx]

        face_centroids = parent.vertices[parent.faces].mean(axis=1)
        principal_axis = parent.axes[:, 0]
        projections = face_centroids @ principal_axis
        median = float(np.median(projections))

        left_mask = projections <= median
        right_mask = ~left_mask
        if not np.any(left_mask) or not np.any(right_mask):
            order = np.argsort(projections)
            mid = len(order) // 2
            left_faces = parent.faces[order[:mid]]
            right_faces = parent.faces[order[mid:]]
        else:
            left_faces = parent.faces[left_mask]
            right_faces = parent.faces[right_mask]

        if len(left_faces) == 0 or len(right_faces) == 0:
            parent.splittable = False
            continue

        left_cluster = _make_cluster(parent.vertices, left_faces)
        right_cluster = _make_cluster(parent.vertices, right_faces)

        if (left_cluster.volume + right_cluster.volume) < parent.volume * (1.0 - min_improvement):
            clusters.pop(best_idx)
            clusters.append(left_cluster)
            clusters.append(right_cluster)
        else:
            parent.splittable = False

    return [
        DecomposedPart(
            primitive_type=PrimitiveType.CUBE,
            center_blender=c.center,
            rotation_blender_3x3=c.axes,
            extents_blender=c.extents,
        )
        for c in clusters
    ]
