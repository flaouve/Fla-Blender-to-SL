"""Deterministic SceneGraph -> Fla-SL-CustomObjects schematic exporter."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from fla_blender_to_sl.classifier import classify_mesh
from fla_blender_to_sl.decomposer import decompose_complex_mesh
from fla_blender_to_sl.ir import MaterialInfo, MeshData, NodeKind, SceneGraph, SceneNode
from fla_blender_to_sl.schema import (
    LightType,
    PrimitiveFlags,
    PrimitiveType,
    SchematicBlockData,
    SchematicObjectDataList,
    Vec3,
    make_empty_block,
    make_light_block,
    make_primitive_block,
    make_text_block,
)
from fla_blender_to_sl.transforms import (
    MIN_SCALE,
    blender_matrix_to_unity_trs,
    linear_rgba_to_sl_color,
)


class ComplexMode(Enum):
    """Strategy for handling complex (non-primitive) meshes."""

    AI_HANDOFF = "ai_handoff"
    DECOMPOSE = "decompose"
    BBOX = "bbox"


@dataclass
class ExportConfig:
    """Configuration for deterministic schematic export."""

    complex_mode: ComplexMode = ComplexMode.AI_HANDOFF
    max_primitives_per_complex: int = 16
    max_total_primitives: int = 2000


@dataclass
class HandoffEntry:
    """Metadata describing a complex mesh placeholder awaiting AI reconstruction."""

    placeholder_object_id: int
    node_id: str
    name: str
    world_position_unity: Vec3
    world_rotation_unity: Vec3
    world_bounds_unity: Vec3
    dominant_color_hex: str
    triangle_count: int
    suggested_budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "placeholder_object_id": int(self.placeholder_object_id),
            "node_id": self.node_id,
            "name": self.name,
            "world_position_unity": self.world_position_unity.to_dict(),
            "world_rotation_unity": self.world_rotation_unity.to_dict(),
            "world_bounds_unity": self.world_bounds_unity.to_dict(),
            "dominant_color_hex": self.dominant_color_hex,
            "triangle_count": int(self.triangle_count),
            "suggested_budget": int(self.suggested_budget),
        }


@dataclass
class ExportResult:
    """Output of exporting a SceneGraph."""

    schematic_name: str
    schematic: SchematicObjectDataList
    handoff_entries: list[HandoffEntry] = field(default_factory=list)


def _sanitize_schematic_name(name: str) -> str:
    """Remove spaces and unsafe characters from schematic filenames (Schematic.cs:107)."""
    cleaned = re.sub(r"\s+", "_", name.strip())
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "", cleaned)
    return cleaned or "Schematic"


def _decompose_blender_world(matrix_4x4: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Extract (position_b, proper_rot_3x3_b, positive_scale_b) from a Blender 4x4 matrix."""
    m = np.asarray(matrix_4x4, dtype=float)
    pos = m[:3, 3].copy()
    linear = m[:3, :3]
    col_norms = np.linalg.norm(linear, axis=0)

    rot_raw = np.eye(3, dtype=float)
    for i in range(3):
        if col_norms[i] > 1e-12:
            rot_raw[:, i] = linear[:, i] / col_norms[i]
    if np.linalg.det(rot_raw) < 0.0:
        rot_raw[:, 0] *= -1.0

    u_mat, _, vt_mat = np.linalg.svd(rot_raw)
    rot_clean = u_mat @ vt_mat
    if np.linalg.det(rot_clean) < 0.0:
        u_mat[:, -1] *= -1.0
        rot_clean = u_mat @ vt_mat

    scale = (float(col_norms[0]), float(col_norms[1]), float(col_norms[2]))
    return pos, rot_clean, scale


def _make_rigid_matrix(pos: np.ndarray, rot_3x3: np.ndarray) -> np.ndarray:
    """Build a 4x4 rigid transform matrix (scale = 1)."""
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rot_3x3
    mat[:3, 3] = pos
    return mat


def _invert_rigid_matrix(mat: np.ndarray) -> np.ndarray:
    """Exact inverse of a 4x4 rigid transform matrix."""
    inv = np.eye(4, dtype=float)
    rt = mat[:3, :3].T
    inv[:3, :3] = rt
    inv[:3, 3] = -(rt @ mat[:3, 3])
    return inv


def _material_to_sl_color(material: MaterialInfo) -> str:
    """Encode a MaterialInfo's base/emission color into SCP:SL LDR hex or HDR string."""
    if material.emission_strength > 1.0:
        er, eg, eb = material.emission_color_linear
        if er > 0.0 or eg > 0.0 or eb > 0.0:
            return linear_rgba_to_sl_color((er, eg, eb, material.base_color_linear[3]), material.emission_strength)
        return linear_rgba_to_sl_color(material.base_color_linear, material.emission_strength)
    return linear_rgba_to_sl_color(material.base_color_linear, material.emission_strength)


def _node_flags(node: SceneNode) -> PrimitiveFlags:
    flags = PrimitiveFlags.NONE
    if node.collidable:
        flags |= PrimitiveFlags.COLLIDABLE
    if node.visible:
        flags |= PrimitiveFlags.VISIBLE
    return flags


def _topological_order(nodes: list[SceneNode]) -> list[SceneNode]:
    """Order nodes so parents are processed before their children."""
    by_id = {n.id: n for n in nodes}
    visited: set[str] = set()
    ordered: list[SceneNode] = []

    def visit(node: SceneNode) -> None:
        if node.id in visited:
            return
        if node.parent_id is not None and node.parent_id in by_id:
            visit(by_id[node.parent_id])
        visited.add(node.id)
        ordered.append(node)

    for n in nodes:
        visit(n)
    return ordered


def export_scene(graph: SceneGraph, config: ExportConfig = ExportConfig()) -> ExportResult:
    """Export a SceneGraph into a Fla-SL-CustomObjects SchematicObjectDataList."""
    root_id = 1000
    next_id = 1001
    blocks: list[SchematicBlockData] = []
    handoff_entries: list[HandoffEntry] = []

    ordered_nodes = _topological_order(graph.nodes)
    world_mats: dict[str, np.ndarray] = {}
    rigid_mats: dict[str, np.ndarray] = {}
    node_to_object_id: dict[str, int] = {}

    for node in ordered_nodes:
        local_mat = np.asarray(node.local_matrix_blender, dtype=float)
        if node.parent_id is not None and node.parent_id in world_mats:
            world_mat = world_mats[node.parent_id] @ local_mat
            parent_rigid = rigid_mats[node.parent_id]
            parent_obj_id = node_to_object_id[node.parent_id]
        else:
            world_mat = local_mat
            parent_rigid = np.eye(4, dtype=float)
            parent_obj_id = root_id

        world_mats[node.id] = world_mat
        world_pos_b, world_rot_b, world_scale_b = _decompose_blender_world(world_mat)
        node_rigid = _make_rigid_matrix(world_pos_b, world_rot_b)
        rigid_mats[node.id] = node_rigid

        rel_rigid_b = _invert_rigid_matrix(parent_rigid) @ node_rigid
        rel_pos_u, rel_rot_u, _ = blender_matrix_to_unity_trs(rel_rigid_b)

        if node.kind == NodeKind.EMPTY or (node.kind == NodeKind.MESH and node.mesh is None):
            obj_id = next_id
            next_id += 1
            node_to_object_id[node.id] = obj_id
            blocks.append(
                make_empty_block(
                    name=node.name,
                    object_id=obj_id,
                    parent_id=parent_obj_id,
                    position=rel_pos_u,
                    rotation=rel_rot_u,
                    scale=Vec3(1.0, 1.0, 1.0),
                    static=node.static,
                )
            )
            continue

        if node.kind == NodeKind.LIGHT and node.light is not None:
            obj_id = next_id
            next_id += 1
            node_to_object_id[node.id] = obj_id
            light = node.light
            color_str = linear_rgba_to_sl_color(light.color_linear)
            blocks.append(
                make_light_block(
                    name=node.name,
                    object_id=obj_id,
                    parent_id=parent_obj_id,
                    position=rel_pos_u,
                    rotation=rel_rot_u,
                    light_type=light.light_type,
                    color=color_str,
                    unity_intensity=light.intensity,
                    unity_range=light.range_m,
                    shadow_type=light.shadow_type,
                    spot_angle=light.spot_angle if light.light_type == LightType.SPOT else None,
                    static=node.static,
                )
            )
            continue

        if node.kind == NodeKind.TEXT and node.text is not None:
            obj_id = next_id
            next_id += 1
            node_to_object_id[node.id] = obj_id
            text_data = node.text
            color_str = linear_rgba_to_sl_color(text_data.color_linear)
            rel_full_b = _invert_rigid_matrix(parent_rigid) @ world_mat
            _, _, rel_scale_u = blender_matrix_to_unity_trs(rel_full_b)
            blocks.append(
                make_text_block(
                    name=node.name,
                    object_id=obj_id,
                    parent_id=parent_obj_id,
                    position=rel_pos_u,
                    rotation=rel_rot_u,
                    scale=rel_scale_u,
                    text=text_data.text,
                    color=color_str,
                    font_size=text_data.font_size,
                    align=text_data.align,
                    display_size=text_data.display_size,
                    static=node.static,
                )
            )
            continue

        if node.kind == NodeKind.MESH and node.mesh is not None:
            mesh = node.mesh
            cls_res = classify_mesh(mesh, world_scale_blender=world_scale_b, override=node.override)
            color_str = _material_to_sl_color(mesh.material)
            flags = _node_flags(node)

            # Offset world position by local mesh center if vertices are not centered at origin
            center_world_b = world_mat[:3, :3] @ cls_res.local_center_blender + world_mat[:3, 3]
            centered_rigid_b = _make_rigid_matrix(center_world_b, world_rot_b)
            rel_centered_b = _invert_rigid_matrix(parent_rigid) @ centered_rigid_b
            prim_pos_u, prim_rot_u, _ = blender_matrix_to_unity_trs(rel_centered_b)

            if not cls_res.is_complex and cls_res.primitive_type is not None:
                obj_id = next_id
                next_id += 1
                node_to_object_id[node.id] = obj_id
                ux, uy, uz = cls_res.unity_scale
                blocks.append(
                    make_primitive_block(
                        name=node.name,
                        object_id=obj_id,
                        parent_id=parent_obj_id,
                        position=prim_pos_u,
                        rotation=prim_rot_u,
                        scale=Vec3(max(ux, MIN_SCALE), max(uy, MIN_SCALE), max(uz, MIN_SCALE)),
                        primitive_type=cls_res.primitive_type,
                        color=color_str,
                        flags=flags,
                        static=node.static,
                    )
                )
                continue

            # Complex mesh handling
            if config.complex_mode == ComplexMode.BBOX:
                obj_id = next_id
                next_id += 1
                node_to_object_id[node.id] = obj_id
                ux, uy, uz = cls_res.unity_scale
                blocks.append(
                    make_primitive_block(
                        name=node.name,
                        object_id=obj_id,
                        parent_id=parent_obj_id,
                        position=prim_pos_u,
                        rotation=prim_rot_u,
                        scale=Vec3(max(ux, MIN_SCALE), max(uy, MIN_SCALE), max(uz, MIN_SCALE)),
                        primitive_type=PrimitiveType.CUBE,
                        color=color_str,
                        flags=flags,
                        static=node.static,
                    )
                )
            elif config.complex_mode == ComplexMode.DECOMPOSE:
                group_id = next_id
                next_id += 1
                node_to_object_id[node.id] = group_id
                blocks.append(
                    make_empty_block(
                        name=node.name,
                        object_id=group_id,
                        parent_id=parent_obj_id,
                        position=rel_pos_u,
                        rotation=rel_rot_u,
                        scale=Vec3(1.0, 1.0, 1.0),
                        static=node.static,
                    )
                )

                scaled_verts = np.asarray(mesh.vertices, dtype=float) * np.abs(np.asarray(world_scale_b, dtype=float))
                scaled_mesh = MeshData(scaled_verts, mesh.faces, mesh.material)
                parts = decompose_complex_mesh(scaled_mesh, max_primitives=config.max_primitives_per_complex)
                for part_idx, part in enumerate(parts):
                    part_id = next_id
                    next_id += 1
                    part_mat_b = _make_rigid_matrix(part.center_blender, part.rotation_blender_3x3)
                    part_pos_u, part_rot_u, _ = blender_matrix_to_unity_trs(part_mat_b)
                    ex, ey, ez = part.extents_blender
                    part_scale_u = Vec3(max(float(ex), MIN_SCALE), max(float(ez), MIN_SCALE), max(float(ey), MIN_SCALE))
                    blocks.append(
                        make_primitive_block(
                            name=f"{node.name}_Part_{part_idx:02d}",
                            object_id=part_id,
                            parent_id=group_id,
                            position=part_pos_u,
                            rotation=part_rot_u,
                            scale=part_scale_u,
                            primitive_type=part.primitive_type,
                            color=color_str,
                            flags=flags,
                            static=node.static,
                        )
                    )
            else:
                # AI_HANDOFF mode: emit Empty placeholder + HandoffEntry
                placeholder_id = next_id
                next_id += 1
                node_to_object_id[node.id] = placeholder_id
                blocks.append(
                    make_empty_block(
                        name=node.name,
                        object_id=placeholder_id,
                        parent_id=parent_obj_id,
                        position=rel_pos_u,
                        rotation=rel_rot_u,
                        scale=Vec3(1.0, 1.0, 1.0),
                        static=node.static,
                    )
                )
                world_pos_u, world_rot_u, _ = blender_matrix_to_unity_trs(node_rigid)
                ux, uy, uz = cls_res.unity_scale
                handoff_entries.append(
                    HandoffEntry(
                        placeholder_object_id=placeholder_id,
                        node_id=node.id,
                        name=node.name,
                        world_position_unity=world_pos_u,
                        world_rotation_unity=world_rot_u,
                        world_bounds_unity=Vec3(max(ux, MIN_SCALE), max(uy, MIN_SCALE), max(uz, MIN_SCALE)),
                        dominant_color_hex=color_str,
                        triangle_count=int(len(mesh.faces)),
                        suggested_budget=min(config.max_primitives_per_complex, 150),
                    )
                )

    schematic_name = _sanitize_schematic_name(graph.name)
    schematic = SchematicObjectDataList(root_object_id=root_id, blocks=blocks)
    return ExportResult(
        schematic_name=schematic_name,
        schematic=schematic,
        handoff_entries=handoff_entries,
    )
